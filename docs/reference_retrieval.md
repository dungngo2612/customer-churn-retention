# Train-only customer reference retrieval

Bước 5.2 chỉ xây dựng reference artifact và tìm khách hàng tương tự. Không tạo
prompt dự đoán, gọi LLM, thay đổi Logistic Regression hoặc tích hợp Streamlit.
Các thư viện hiện có đáp ứng toàn bộ chức năng; không thêm dependency.

## Build offline

```bash
PYTHONPATH=src python3 -m churn_retention.workflows.build_reference \
  --train data/processed/train.csv \
  --splits artifacts/splits/dataset_splits.csv \
  --bundle artifacts/bundles/logistic-20261018T130159-seed42 \
  --out artifacts/references/telco-train-v1.zip
```

Hoặc gọi `build_reference(train_path, bundle_path, splits_path, output_path)`.
Có thể truyền `reference_version`; mặc định version được tạo từ hash các nguồn,
format và phiên bản thư viện. Cùng nguồn và runtime tạo cùng version và ZIP bytes.

Builder đọc các nguồn một lần vào bộ nhớ, kiểm tra:

- Đúng 19 feature theo `feature_schema.json` của bundle, cộng `customerID`, `Churn`.
- ID không rỗng/trùng; tập ID khớp **toàn bộ train IDs** trong split manifest.
  Unknown, validation/test và thiếu train IDs đều bị từ chối.
- Nhãn chỉ là chuỗi `No` hoặc `Yes`, giữ nguyên trong reference; không mã hóa vào
  vector. ID cũng không xuất hiện trong input của preprocessor.
- Bundle là Logistic Regression với fitted `ColumnTransformer`, đúng tên/thứ tự
  feature, numerical imputer/scaler và categorical one-hot encoder.
- Numerical hữu hạn, không âm; `SeniorCitizen` là 0/1; các cột integer phải nguyên.
  Categorical phải thuộc category đã fit. Không tự dùng mã unknown của encoder.
- `TotalCharges` None/NaN/chuỗi trắng chỉ được thay bằng 0 khi `tenure=0`;
  nếu tenure dương thì báo lỗi. Các missing value khác bị từ chối.

Chỉ gọi `transform`, không gọi `fit`, `fit_transform`, `predict_proba` hoặc huấn
luyện classifier. Source CSV, split manifest và bundle không bị sửa.

Artifact ZIP chứa:

| Entry | Nội dung |
| --- | --- |
| `manifest.json` | Format/version, số dòng/chiều, schema, hashes, split counts và runtime |
| `records.json` | Feature gốc đã chuẩn hóa, ID nội bộ, nhãn đúng của từng khách hàng |
| `matrix.npy` | Ma trận float64 numerical/one-hot, không có object/pickle data |
| `splits.csv` | Bản sao mapping ID/split; chỉ metadata của holdout, không có feature/nhãn holdout |
| `bundle/model.pkl` | Bản sao nguyên byte fitted pipeline gốc; chỉ sử dụng preprocessor |
| `bundle/feature_schema.json` | Schema chính thức nguyên byte |
| `bundle/metadata.json` | Metadata model gốc nguyên byte |
| `bundle/effective_config.yaml` | Có khi bundle nguồn chứa file này |

Rows được xếp theo ID; labels, feature gốc và matrix dùng chung thứ tự.
Checksum SHA-256 kiểm tra từng payload, bundle và split; dataset hash là hash byte
của CSV nguồn. File tạm được ghi hoàn chỉnh, fsync và kiểm tra bằng Store trước
khi hard-link sang output. Không overwrite kể cả khi builder khác vừa publish.
File staging bị xóa sau thành công/thất bại. Local filesystem phải hỗ trợ hard link.

## Load một lần, retrieve nhiều lần

```python
import pandas as pd

from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.adapters.retrieval.nearest_neighbors import SimilarCustomerRetriever

store = ReferenceStore.load("artifacts/references/telco-train-v1.zip")
retriever = SimilarCustomerRetriever(store, k=5)

# Ví dụ lấy query từ dữ liệu có sẵn; không cần biết Churn của query.
row = pd.read_csv("data/processed/train.csv").iloc[0].to_dict()
matches = retriever.retrieve(row)
examples = [match.to_prompt_data() for match in matches]
```

Store kiểm checksum trước khi unpickle, từ chối format/runtime không tương thích,
đối chiếu schema, số dòng/chiều, membership train và toàn bộ matrix/record alignment
một lần khi load. Có thể pin `expected_reference_version`, `expected_bundle_sha256`,
`expected_dataset_sha256`. Sklearn phải đúng phiên bản; NumPy phải cùng major.

Store giữ records, fitted preprocessing và ma trận trong RAM; dữ liệu trả ra dùng
mapping/read-only array để tránh thay đổi nhầm. Mỗi query chỉ transform DataFrame
`1 × 19`, không đọc CSV/ZIP, kiểm hash file hoặc transform lại toàn bộ train.

`retrieve(query, k=None, customer_id=None)`:

- Query phải có đủ 19 feature. Numeric string hợp lệ được chuyển về kiểu numerical.
- `Churn` của query được bỏ qua hoàn toàn. ID có thể truyền riêng hoặc bằng
  `customerID`/`customer_id`; các ID được truyền phải nhất quán.
- Dùng `NearestNeighbors(metric="euclidean", algorithm="brute", n_jobs=1)`.
  `fit` của NearestNeighbors chỉ index matrix đã có, không refit preprocessing.
- K mặc định 5; K phải là Python integer dương, không chấp nhận boolean.
  K quá lớn được giảm còn số khách khả dụng; chỉ còn chính khách đó thì trả `[]`.
- Loại reference trùng ID của query. Các khách khác ID nhưng cùng feature vẫn là
  các bản ghi riêng; không tự gộp nhãn có thể khác nhau.
- Khoảng cách tăng dần; bằng khoảng cách thì xếp theo ID để deterministic cả tại
  ranh giới K. Tính tất cả khoảng cách trong RAM để xử lý tie chính xác ở quy mô này.

Mỗi `SimilarCustomer` có `features`, `churn`, `distance`, `customer_id` nội bộ.
`to_prompt_data()` chỉ trả `{"features": {...}, "Churn": "Yes/No"}`; không có ID,
distance hoặc encoded vector. Không serialize toàn bộ object làm prompt data.
Euclidean distance không phải xác suất churn. Chức năng này không tạo prompt text.

## Kiểm thử và giới hạn

```bash
PYTHONPATH=src python3 -m pytest tests/test_reference_retrieval.py \
  tests/test_reference_retrieval_integration.py -q
PYTHONPATH=src python3 -m pytest tests --ignore=tests/test_reference_retrieval.py \
  --ignore=tests/test_reference_retrieval_integration.py -q
python3 -m ruff check src/churn_retention/adapters/retrieval \
  src/churn_retention/workflows/build_reference.py tests/test_reference_retrieval*.py
python3 -m ruff format --check src/churn_retention/adapters/retrieval \
  src/churn_retention/workflows/build_reference.py tests/test_reference_retrieval*.py
PYTHONPATH=src python3 -m mypy src/churn_retention/adapters/retrieval \
  src/churn_retention/workflows/build_reference.py tests/test_reference_retrieval*.py
```

Unit tests có fixture nhỏ, tách biệt dữ liệu dự án. Chỉ transformer fixture mới được
fit khi dựng test; trong phần chạy Builder/Store/Retriever, fit bị cấm và kiểm tra.
Không huấn luyện classifier của fixture. Integration tests dùng CSV/bundle thực
đang có và skip rõ ràng nếu thiếu asset local; không thay bằng dữ liệu giả.
Không có LLM API request trong các test.

Chỉ nạp artifact/bundle local đáng tin cậy: sklearn pickle có thể thực thi code;
checksum bảo vệ integrity, không phải chữ ký chống sửa manifest độc hại.
Manifest xác minh reference thuộc train; lịch sử fit của bundle gốc vẫn dựa vào
provenance đáng tin cậy của bundle, vì metadata cũ chưa lưu toàn bộ fit IDs.
Không có cập nhật online: khi nguồn thay đổi, build artifact mới và nạp store mới.
K/metric hiện là baseline kỹ thuật, chưa chứng minh cải thiện chất lượng dự đoán churn.
