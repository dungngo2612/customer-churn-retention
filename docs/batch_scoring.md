# M2.4 — Batch scoring, job và worker trong tiến trình

Base: `origin/feat/ml-explain@19d0838`. Giữ nguyên Prediction v1 và use case
`modules/scoring/use_cases.py::score_dataset`. Trên base chưa có
`workflows/score_dataset.py`; file mới này chỉ điều phối job/chunk, không train,
calibrate hoặc sửa code dataset/retention/dashboard của thành viên khác.

## Quyết định và các port

Chọn **SQLite** (stdlib `sqlite3`) cho cả `JobRepository` và `PredictionStore`:
`job-status` trong một lần gọi CLI khác vẫn đọc được trạng thái. Không thêm dependency.
`SQLiteScoringStore` dùng một database chung để chuyển trạng thái/publication atomically;
`BEGIN IMMEDIATE` bảo vệ create-if-absent và claim, kể cả qua hai connection khác nhau.
Một partial unique index chỉ cho một job PENDING/RUNNING/SUCCEEDED trên mỗi key.

Các port ở `modules/scoring/batch_ports.py`:

| Port | Adapter hiện tại | Trách nhiệm khi thay adapter |
| --- | --- | --- |
| JobRepository | SQLiteScoringStore | Create-or-get/claim atomic, tiến độ đơn điệu, trạng thái và publication |
| PredictionStore | SQLiteScoringStore | Stage theo chunk, chống trùng, chỉ đọc kết quả SUCCEEDED |
| DatasetReader | CsvDatasetReader | Tra dataset_id/path, đếm dòng và đọc iterator chunk |
| JobQueue | InProcessJobQueue | Enqueue/dequeue tham chiếu job, không chứa dữ liệu khách |

`BundleScoringModels` resolve đúng thư mục `<bundle-root>/<model_version>` và kiểm
metadata khớp version. Không có alias `latest`, không quét bundle mới nhất. Workflow
nhận predictor factory qua dependency injection và kiểm version trước mỗi chunk.
Bundle version và dataset ID được coi là immutable; thêm thư mục bundle mới không
đổi job đã tạo. Thay nội dung cùng version/path không phải cơ chế cập nhật được hỗ trợ.

## Job, trạng thái, idempotency

`ScoringJob` là dataclass frozen: `job_id`, `dataset_id`, `model_version`, `chunk_size`,
`total_rows`, `processed_rows`, `status`, `error`, `created_at`, `started_at`,
`finished_at`, `lease_seconds`, `lease_expires_at`; có `idempotency_key` và cấu hình `explain_limit`, `explain_customer_ids`.
Timestamps là ISO-8601 có timezone; `scoring_run_id = job_id` trong Prediction v1.

```text
PENDING -> RUNNING -> SUCCEEDED
                  -> FAILED
```

Chuyển trạng thái ngoài sơ đồ raise `InvalidJobTransition`, bao gồm
SUCCEEDED/FAILED -> RUNNING. Không SUCCEEDED trước khi xử lý đủ dòng và staging đủ
Prediction. Tiến độ chỉ tăng trong RUNNING. `processed_chunks` là số chunk đã xử lý
thành công, tính từ processed_rows/chunk_size theo adapter CSV có chunk cố định.

Key = SHA-256 của UTF-8 JSON `[dataset_id, model_version]`, separator `,`/`:` và
`ensure_ascii=False`; JSON framing tránh nhập nhằng khi nối chuỗi.

- Cùng key PENDING/RUNNING chưa hết lease/SUCCEEDED: trả job cũ với nguyên cấu hình đầu tiên.
  Đổi chunk_size/explain_limit không tạo lại job. PENDING được enqueue lại an toàn;
  queue loại trùng reference đang chờ.
- Cùng key FAILED: tạo job_id/scoring_run_id mới, đọc lại từ đầu. Không resume chunk.
- Delivery lại job RUNNING/SUCCEEDED/FAILED: trả trạng thái hiện tại, không chạy lại.
  Claim chuyển PENDING -> RUNNING atomic. Thu hồi lease xảy ra khi gửi lại yêu cầu
  tạo job cùng key, không phải khi chỉ delivery lại message job_id cũ.
- Primary key `(job_id, customer_id)` tương ứng `(scoring_run_id, customer_id)`;
  vị trí dòng cũng unique theo job. ID trùng trong/cross chunk làm job FAILED,
  không dùng upsert để che mất dữ liệu trùng.
- ID mặc định CLI là đường dẫn CSV đã resolve. Có thể đặt `--dataset-id` cho ID
  nghiệp vụ. Caller phải giữ ID và dữ liệu immutable. Đếm dòng phát hiện thay đổi số
  dòng giữa tạo/chạy job; không checksum nội dung cùng số dòng trong bản này.

## Chunk và thất bại giữa chừng

CSV reader chỉ giữ từng chunk trong bộ nhớ; bước đếm dòng dùng iterator CSV.
Chuẩn hoá `customerID` thành `customer_id`, giữ ID dạng chuỗi (kể cả số 0 đầu),
loại cột nhãn `Churn`; không thay thế bước làm sạch dataset của Hiếu. Nếu có cả hai
cột ID, header trùng, ID null/rỗng/trùng hoặc schema không hợp lệ thì không chấm thành công.
Mỗi chunk dùng use case scoring cũ, xác suất phải hữu hạn và trong `[0,1]`.

Prediction/explanation được **stage trong transaction của từng chunk** vào database.
Sau stage thành công mới cập nhật processed_rows. Trong suốt PENDING/RUNNING/FAILED,
các phương thức đọc kết quả raise `ResultsNotReady`; không xuất bản một batch dở dang.
Kết thúc kiểm số Prediction staging = total_rows và processed_rows = total_rows,
rồi commit SUCCEEDED. Đây là điểm publication duy nhất trong cùng SQLite database.
Consumer phải đọc qua port, không đọc trực tiếp bảng staging.

Lỗi predict/SHAP/đọc CSV/ghi chunk được bắt trong worker: chuyển RUNNING -> FAILED
qua máy trạng thái và xoá staging (kể cả explanation qua FK cascade) trong cùng
transaction. Giữ processed_rows cuối cùng để chẩn đoán, không coi là số kết quả đã
publish. Lỗi vật lý khiến database không ghi được trạng thái phải được caller xử lý.

Lỗi mới lưu trong job và lỗi JSON do CLI bắt chỉ giữ tên loại exception cùng thông
báo cố định `batch scoring failed`; không nội suy exception message, input mẫu của
Pydantic, customer ID hay giá trị feature. Lỗi lease vẫn là `lease expired`.
Chẩn đoán chi tiết từ exception không được lưu hoặc in ra qua các trường error này.

## Lease và thu hồi job bị bỏ rơi

`DEFAULT_LEASE_SECONDS = 300`; CLI cấu hình bằng `--lease-seconds`, service nhận
`lease_seconds` và `clock` có thể thay bằng đồng hồ giả. Duration được lưu với job,
không thay khi submit trùng key. Claim gán `lease_expires_at = now + lease_seconds`;
sau mỗi chunk stage và cập nhật tiến độ thành công, gia hạn lại cùng duration.
Chọn duration lớn hơn thời gian xử lý chunk (kể cả SHAP). Không có heartbeat nền:
chunk chạy quá hạn bị FAILED, không được hồi sinh lease hay publish kết quả.

Khi `create_job` cùng key gặp RUNNING: nếu deadline > now thì trả job cũ. Nếu deadline
<= now hoặc NULL, transaction chuyển RUNNING -> FAILED với chính xác `lease expired`,
xoá toàn bộ staging/explanation rồi cho phép tạo job_id mới và đọc CSV từ đầu.
Create-or-get kiểm lại điều kiện dưới `BEGIN IMMEDIATE`, tránh hai submit đồng thời
tạo hai job mới. Worker cũ bị chặn stage/renew/publish theo trạng thái và lease;
không ghi đè error của job đã thu hồi. Query status chỉ đọc, không tự thu hồi job.

Mở database cũ tự thêm cột nullable `lease_expires_at`, giữ nguyên dữ liệu và cấu trúc
JSON còn lại; job cũ không có deadline được xử lý như đã hết lease. Không dùng staging
cũ cho job mới. Public port vẫn chỉ trả kết quả của SUCCEEDED.

## SHAP có giới hạn

`explain_limit=0` mặc định: không import SHAP từ bridge, không mở background train.
Với N > 0, tính từng khách trong tối đa N khách đầu; hoặc lọc bằng danh sách
`--explain-customer-id` lặp lại. Danh sách phải unique, không dài quá N; thiếu ID
được yêu cầu khiến job FAILED. Không tự tính explanation cho toàn bộ lô.
N lớn hơn số khách chỉ có thể tạo tối đa số khách thực tế.

Bridge tái dùng `ml/explainability` M2.3, background chỉ từ `--background-train`
có tên resolve `train.csv`, seed 42, tối đa 200 dòng. Caller vẫn phải đảm bảo
provenance train, không đổi tên val/test thành train. Khách cần giải thích có thể
thuộc val/dataset scoring; không đưa họ vào background. Không mở test.csv.
Kết quả theo contract `ModelExplanation`: log-odds, gộp one-hot, giữ bản chi tiết;
**SHAP mô tả mô hình, không phải nhân quả**. Phạm vi M2.3 giữ nguyên: logistic không
class weight/không calibration; CatBoost yêu cầu SHAP báo chưa hỗ trợ và job FAILED.

## Điểm tích hợp với Hiếu và Cường

Đã **chỉ đọc** `origin/feature/data-infrastructure-ci@e6fa0ce` (forsaken1800), không
import/copy migration, ORM, repository hay use case của nhánh chưa merge:

- Migration `001_create_dataset_metadata_table.py`, revision `001_create_dataset_metadata`,
  bảng `dataset_metadata`: `dataset_id`, `filename`, `total_rows`, `total_columns`,
  `features_s3_uri`, `labels_s3_uri`, `has_labels`, `created_at`.
- Repository có `get_by_id(dataset_id)`. Khi tích hợp, viết adapter DatasetReader
  dùng metadata để resolve features URI; không truyền labels URI vào predictor.
- Cần Hiếu thống nhất mapping: migration dùng dataset_id String(36), ORM String(64);
  ORM/repository dùng `dataset_id/features_s3_uri/labels_s3_uri`, trong khi upload use case
  đang khởi tạo `id/features_path/labels_path`. CSV adapter ở đây độc lập với các tên
  chưa thống nhất đó; không sửa hoặc phụ thuộc code trên nhánh của Hiếu.
- Điểm tích hợp với Cường: tên task/routing là **scoring**; body message chỉ
  `{"job_id":"..."}`. `handle_scoring_message`/`run_worker_once` ở
  `entrypoints/worker/tasks/scoring.py`. UI dùng interface job/status ở trên;
  status terminal SUCCEEDED/FAILED, error khi FAILED; chỉ hiển thị Prediction khi
  SUCCEEDED. Khoá chống trùng chỉ gồm dataset_id + model_version.
- Yêu cầu SHAP thêm sau khi job SUCCEEDED không đổi job cũ; dùng màn hình/API M2.3
  hoặc thiết kế task explanation riêng khi phối hợp, không ngầm phá idempotency.

## Thay bằng queue/Postgres thật và giới hạn bản hiện tại

`score-batch` tạo/enqueue job rồi gọi `run_worker_once` ngay **trong cùng tiến trình**,
đợi kết thúc và in JSON job. Đây không phải dịch vụ worker nền. `job-status` có thể
chạy riêng với cùng `--state-db`. CLI trả exit 1 khi FAILED hoặc đầu vào/lưu trữ lỗi.
`job-status` gọi use case `get_job_status(job_id, jobs)` trong
`workflows/score_dataset.py` qua port `JobRepository`, không gọi storage trực tiếp
từ CLI. Use case chỉ đọc: không mở CSV/bundle/background, không enqueue/chạy worker
hoặc thu hồi lease. Job không tồn tại trả JSON error đã lọc và exit 1.
Queue trong bộ nhớ mất khi tiến trình kết thúc; không có durable outbox/heartbeat nền.
PENDING còn trong SQLite có thể resubmit để enqueue lại. Crash cứng lúc RUNNING để
lại staging không publish; gửi lại yêu cầu sau hạn lease sẽ thu hồi và chấm từ đầu.
Không có exactly-once execution: worker cũ có thể còn tính toán sau khi hết lease,
nhưng không được publish. SQLite là storage local; chưa dùng cho nhiều host.

Khi thay adapter: dùng Postgres cho cả hai storage port với transaction publication
chung, partial unique key và claim atomic; queue thật vẫn chỉ mang job_id. Thêm outbox
để không mất enqueue, ack sau khi đọc kết quả terminal; triển khai cùng ngữ nghĩa
lease/recovery, fencing worker cũ và atomic cleanup khi thay storage. Không cần sửa workflow
cho SQL/broker, nhưng adapter mới phải đáp ứng cùng cam kết atomic/visibility.
Các file __init__.py worker/task_queue/database 0 byte trên base được giữ nguyên;
chỉ thêm __init__.py cho hai package mới `database/scoring` và `worker/tasks`.

## Chạy trong thư mục tạm

Từ gốc repo sau khi commit (Bash/Zsh):

```sh
BATCH_REPO="$PWD"
BATCH_TMP="$(mktemp -d)"
git archive HEAD | tar -x -C "$BATCH_TMP"
cd "$BATCH_TMP"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$BATCH_TMP/matplotlib"
export MLFLOW_TRACKING_URI="sqlite:///$BATCH_TMP/mlflow.db"
python3 scripts/make_splits.py --raw-churn-path data/telco_churn.csv \
  --splits-path artifacts/splits/dataset_splits.csv \
  --train-out data/processed/train.csv --val-out data/processed/val.csv
python3 -m churn_retention.entrypoints.cli.train --config configs/train_config.yaml \
  --model-type logistic --no-class-weight --seed 42 \
  --train-path data/processed/train.csv --val-path data/processed/val.csv \
  --out-dir artifacts/bundles
MODEL_VERSION="$(python3 -c 'from pathlib import Path; print(next(Path("artifacts/bundles").iterdir()).name)')"
python3 -m churn_retention.entrypoints.cli score-batch --dataset data/processed/val.csv \
  --model-version "$MODEL_VERSION" --bundle-root artifacts/bundles \
  --chunk-size 200 --explain-limit 5 --background-train data/processed/train.csv \
  --state-db scoring.sqlite3 > score.json
JOB_ID="$(python3 -c 'import json; print(json.load(open("score.json"))["job_id"])')"
python3 -m churn_retention.entrypoints.cli job-status --job-id "$JOB_ID" --state-db scoring.sqlite3
```

Đối chiếu với chấm một lần, giữ nguyên thứ tự/ID và dùng dung sai số thực `1e-12`:

```sh
python3 - <<'PY'
from datetime import UTC, datetime
import json
from pathlib import Path
import numpy as np
from churn_retention.adapters.database.scoring.csv_reader import CsvDatasetReader
from churn_retention.adapters.database.scoring.sqlite import SQLiteScoringStore
from churn_retention.adapters.predictor.batch_models import BundleScoringModels
from churn_retention.modules.scoring.use_cases import score_dataset

job = json.load(open("score.json"))
frame = next(CsvDatasetReader().chunks(job["dataset_id"], job["total_rows"]))
predictor = BundleScoringModels(Path("artifacts/bundles")).predictor_for(job["model_version"])
full = score_dataset(job["dataset_id"], lambda _: frame, predictor,
                     lambda: job["job_id"], lambda: datetime.now(UTC))
with SQLiteScoringStore("scoring.sqlite3") as store:
    rows, explanations = store.predictions(job["job_id"]), store.explanations(job["job_id"])
assert [p.customer_id for p in rows] == [p.customer_id for p in full.predictions]
error = float(np.max(np.abs(np.array([p.churn_probability for p in rows]) -
                           np.array([p.churn_probability for p in full.predictions]))))
assert error <= 1e-12 and len(explanations) == 5
print("rows/chunks/seconds/status:", len(rows), job["processed_chunks"], job["elapsed_seconds"], job["status"])
print("explanations:", len(explanations), "max probability difference:", error)
PY
cd "$BATCH_REPO"
rm -r -- "$BATCH_TMP"
```

## Kiểm tra trong phạm vi

```sh
PYTHONPATH=src python3 -m pytest tests/test_batch_scoring.py tests/test_score_dataset.py \
  tests/test_scoring_contracts.py tests/test_scoring_api.py -q
```

Ruff check, ruff format --check và mypy chạy trên 13 file Python mới/sửa: jobs và
batch_ports; database/scoring (3 file); in_process queue; batch_models; workflow;
worker/tasks (2 file); CLI batch_scoring và __main__; test_batch_scoring.

```bash
BATCH_FILES=(
  src/churn_retention/modules/scoring/jobs.py
  src/churn_retention/modules/scoring/batch_ports.py
  src/churn_retention/adapters/database/scoring/__init__.py
  src/churn_retention/adapters/database/scoring/sqlite.py
  src/churn_retention/adapters/database/scoring/csv_reader.py
  src/churn_retention/adapters/task_queue/in_process.py
  src/churn_retention/adapters/predictor/batch_models.py
  src/churn_retention/workflows/score_dataset.py
  src/churn_retention/entrypoints/worker/tasks/__init__.py
  src/churn_retention/entrypoints/worker/tasks/scoring.py
  src/churn_retention/entrypoints/cli/batch_scoring.py
  src/churn_retention/entrypoints/cli/__main__.py
  tests/test_batch_scoring.py
)
python3 -m ruff check "${BATCH_FILES[@]}"
python3 -m ruff format --check "${BATCH_FILES[@]}"
python3 -m mypy "${BATCH_FILES[@]}"
```

Không chạy full pytest. Tests dùng SQLite tạm và model tối thiểu, kiểm trạng thái,
atomic claim/create, idempotency, chunk/progress, xác suất hữu hạn/biên 0/1,
failure không publish, ID trùng, version cố định, message chỉ job_id, bounded SHAP,
CLI status và tương đương chấm một lần. Các test scoring v1 giữ nguyên.
Kiểm tra sau ba bản vá: **102 passed**; ruff check/format và mypy sạch trên 6 file
Python có thay đổi. Regression tests kiểm lỗi load model, đọc CSV, predict, SHAP,
stage và CLI không lưu/in customer ID hoặc giá trị feature trong exception message;
job-status dùng use case cho cả bốn trạng thái và job không tồn tại, không đọc input,
chạy worker hay thu hồi lease.

## Kết quả đã chạy thật

Đo ngày 2026-10-29 trong archive tạm với mã M2.4, trên baseline logistic của
`origin/feat/ml-explain@19d0838`, không class weight và không calibration.
`make_splits.py` tạo 4930 dòng train và 1056 dòng val; CLI train báo val ROC-AUC
0.8454, PR-AUC 0.6309. Không đọc test.csv; không commit CSV, bundle, database hay log tạm.

- Model: `logistic-20261029T135734-seed42`.
- Job: `c8d8cfc4-7c24-40d8-8d81-1b02b7636af4`; score-batch và job-status riêng đều SUCCEEDED.
- 1056/1056 dòng, 6 chunk `[200, 200, 200, 200, 200, 56]`, 1056 Prediction, đúng 5 explanation.
- Thời gian `elapsed_seconds` do score-batch đo: **2.266021 giây**, không gồm thời gian
  import Python trước khi vào command handler.
- Thứ tự/ID khớp chấm một lần; chênh xác suất lớn nhất **1.1102230246251565e-16**,
  nhỏ hơn dung sai 1e-12. Không tuyên bố bằng nhau từng bit giữa các kích thước ma trận.
- Background của cả 5 explanation không trùng ID với val; không có SHAP cho các khách còn lại.

Phần làm tạm: CSV/path registry, SQLite local và queue volatile/worker đồng bộ trong
tiến trình; chưa có service worker/broker/Postgres, durable outbox, heartbeat nền hoặc
endpoint HTTP job status. Các điểm tích hợp phía trên không kéo code Hiếu/Cường vào nhánh này.
