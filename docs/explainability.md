# M2.3 — Giải thích logistic bằng SHAP

Phạm vi: bundle logistic **không class weight, không calibration**, cùng cấu trúc
`model.pkl`, `feature_schema.json`, `metadata.json` của scoring. CatBoost **chưa hỗ trợ**
trong M2.3: API và UI báo lỗi rõ ràng; không thay bằng predictor giả. Pipeline được bọc
calibrator hoặc logistic có class weight cũng bị từ chối.

**SHAP mô tả mô hình, không phải nhân quả**.

## Cách tính và đơn vị

`explain_global(bundle, background_df)` và
`explain_one(bundle, row_df, background_df)` nằm trong
`src/churn_retention/ml/explainability/`. Hai hàm chỉ gọi `transform` trên preprocessor
đã fit trong bundle; không fit lại imputer, scaler, encoder hay classifier.

Với dữ liệu đã preprocessing `z`, logistic dự đoán
`log_odds = intercept + Σ coefficient[j] × z[j]` và
`probability = sigmoid(log_odds)`. `shap.LinearExplainer` dùng
`shap.maskers.Independent` trên dữ liệu đã preprocessing; đầu ra là **log-odds**:

```text
base_value = trung bình log-odds trên background
SHAP[j] = coefficient[j] × (z[j] − trung bình background[j])
base_value + tổng contribution = log-odds dự đoán
sigmoid(log-odds dự đoán) = predict_proba của bundle
```

`output_type = "log_odds"`, `calibration_applied = "original"`. Contribution không có
đơn vị điểm phần trăm xác suất. Dương/âm nghĩa là tăng/giảm log-odds churn so với
background; chuyển tổng log-odds qua sigmoid mới được xác suất. Không dùng cùng loại
output này để mô tả xác suất sau calibration trong tương lai.

## Background và cách gộp one-hot

- Background **chỉ lấy từ train.csv**, lấy mẫu `min(200, len(train))` dòng với
  `random_state=42`, không hoàn lại. Truyền `max_samples=200` cho masker để SHAP
  không âm thầm giảm còn mặc định 100 dòng.
- Giữ ID trong đầu vào để xuất danh sách `background.customer_ids`; chỉ các cột
  trong `feature_schema` được đưa qua model. Không đưa Churn hay ID vào features.
- API nhận DataFrame nên người gọi phải bảo đảm nguồn train. Nó từ chối
  `attrs["source_split"] != "train"` hoặc cột `split` có dòng ngoài train;
  không thể chứng minh nguồn dữ liệu từ DataFrame không có metadata. UI chỉ mở
  đường dẫn đã resolve có tên `train.csv` (kể cả kiểm tra đích symlink);
  đổi tên một file khác thành train.csv không phải là bằng chứng nguồn train.
- Không đọc `test.csv`. Một khách val được dùng để minh hoạ local explanation,
  không được bổ sung vào background hoặc dùng để fit/tune.
- Mapping từ cột đã biến đổi về feature gốc lấy từ **encoder và ColumnTransformer
  đã fit**, không cắt tên ở dấu `_`. Tên feature/category có `_` vẫn đúng.
- Với mỗi dòng, cộng SHAP các one-hot của cùng feature thành contribution gốc.
  Global importance là `mean(abs(tổng SHAP theo feature trên từng dòng))` trên
  mẫu nền; không cộng các mean tuyệt đối của từng one-hot. Tie-break theo tên
  feature để thứ tự ổn định. Giữ cả `detailed_contributions`/`detailed_importance`.

Kết quả phụ thuộc mẫu nền, mô hình và các feature tương quan. Global importance ở
đây mô tả 200 dòng mẫu nền, không khẳng định là thống kê trên toàn bộ khách hàng.
Tổng đóng góp đúng về mặt mô hình không chứng minh tác động của một can thiệp giữ chân.

## Contract và màn hình

Contract `ModelExplanation` trong `contracts/http/explanations.py` mở rộng
`ExplanationList`, tái dùng từng record `Explanation` đã có trong `scoring.py`.
Contract cũ giữ nguyên. Contract mới xuất key `contributions` khi dùng
`model_dump(by_alias=True)` / `model_dump_json(by_alias=True)`; đồng thời có
`output_type`, `base_value`, `log_odds`, `probability`, `model_version`,
`feature_schema`, `detailed_contributions`, `calibration_applied` và `background`.
Thiếu trường bắt buộc, đơn vị sai, customer/model không khớp, hoặc sai đẳng thức
cộng tính/sigmoid đều bị từ chối. Dung sai cộng tính là `1e-6`.

```sh
PYTHONPATH=src python3 -m streamlit run apps/streamlit/scoring_page.py
```

Chấm điểm một CSV có `customer_id`, rồi mở “Giải thích mô hình và khách hàng”.
Trang giải thích dùng chính object bundle và bản chụp dataset vừa scoring thành công
trong session; không tải một model khác. Nhập đường dẫn train.csv, chọn khách và
bấm “Tính SHAP”. Màn hình có global bar chart mean |SHAP|, local bar chart có dấu,
bảng tăng/giảm log-odds, base value, log-odds, xác suất và JSON chứa cả chi tiết one-hot.
`data/processed/val.csv` dùng tên ID `customerID`; nếu dùng làm dataset scoring,
hãy tạo bản CSV tạm đổi cột đó thành `customer_id` theo contract scoring hiện tại.
Không đổi dữ liệu train hoặc schema bundle.

Logic chọn khách, chuẩn bị bảng và lưu context là hàm thuần trong
`entrypoints/streamlit/explanation_views.py`; kiểm bằng unit test, không viết test
khởi chạy Streamlit. Chưa kiểm thử giao diện bằng trình duyệt.

## Lệnh chạy thật có thể tái lập

Chạy từ gốc repo sau khi commit. Archive và mọi CSV/bundle/MLflow đều ở thư mục tạm.
Script splits chỉ xuất train/val theo manifest có sẵn; không mở test.csv.

```sh
EXPLAIN_REPO="$PWD"
EXPLAIN_TMP="$(mktemp -d)"
git archive HEAD | tar -x -C "$EXPLAIN_TMP"
cd "$EXPLAIN_TMP"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$EXPLAIN_TMP/matplotlib"
export MLFLOW_TRACKING_URI="sqlite:///$EXPLAIN_TMP/mlflow.db"
python3 scripts/make_splits.py \
  --raw-churn-path data/telco_churn.csv \
  --splits-path artifacts/splits/dataset_splits.csv \
  --train-out data/processed/train.csv --val-out data/processed/val.csv
python3 -m churn_retention.entrypoints.cli.train \
  --config configs/train_config.yaml --model-type logistic --no-class-weight \
  --train-path data/processed/train.csv --val-path data/processed/val.csv \
  --seed 42 --out-dir artifacts/bundles
python3 - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.ml.explainability import explain_global, explain_one
from churn_retention.ml.training.bundle import load_bundle

bundle = load_bundle(next(Path("artifacts/bundles").iterdir()))
train = pd.read_csv("data/processed/train.csv")
train.attrs["source_split"] = "train"
val = pd.read_csv("data/processed/val.csv")
global_result = explain_global(bundle, train)
local = ModelExplanation.model_validate(explain_one(bundle, val.iloc[[0]], train))
features = [column["name"] for column in bundle.feature_schema]
logit = float(bundle._pipeline.decision_function(val[features].iloc[[0]])[0])
error = abs(local.base_value + sum(c.contribution for c in local.contributions) - logit)
probability_error = abs(1 / (1 + np.exp(-logit)) - local.probability)
overlap = set(global_result["background"]["customer_ids"]) & set(val.customerID.astype(str))
assert error <= 1e-6 and probability_error <= 1e-6 and not overlap
print("train/val:", len(train), len(val), "background:", global_result["background"]["rows"])
print("top5:", global_result["feature_importance"][:5])
print("customer:", local.customer_id, "base:", local.base_value,
      "log_odds:", logit, "probability:", local.probability)
print("additivity error:", error, "sigmoid error:", probability_error, "val overlap:", len(overlap))
PY
cd "$EXPLAIN_REPO"
rm -r -- "$EXPLAIN_TMP"
```

Đã đo ngày 2026-10-29 với baseline main `e26c373`, sklearn 1.9.1, SHAP **0.53.0**
(dependency có sẵn `shap>=0.53,<1`, không sửa pyproject): train/val = 4930/1056,
19 feature gốc → 45 cột, background 200 dòng và 0 ID trùng val.
Train CLI báo val ROC-AUC 0.8454, PR-AUC 0.6309. Top 5 mean |SHAP| (log-odds):

| Feature | Mean absolute SHAP |
| --- | ---: |
| tenure | 1.0876233649 |
| Contract | 0.5163961681 |
| TotalCharges | 0.5081728101 |
| InternetService | 0.4488909323 |
| MonthlyCharges | 0.3778163105 |

Khách val `5129-JLPIS`: base `-1.7121662193045242`, tổng contribution
`1.5645895172288626`, log-odds `-0.14757670207566165`, xác suất
`0.4631726384657812`. Sai số cộng tính `1.1102230246251565e-16`; sai số sigmoid `0.0`.
Model version của lượt đo: `logistic-20261029T105443-seed42` (timestamp thay đổi mỗi lượt train).
Đây là số đã chạy thật; không đọc test set, không đưa bundle/CSV tạm vào Git.

## Kiểm tra giới hạn phạm vi

```sh
PYTHONPATH=src python3 -m pytest tests/test_explainability.py \
  tests/test_explanation_contracts.py tests/test_explanation_views.py \
  tests/test_scoring_contracts.py -q
```

Ruff check, ruff format --check và mypy chạy trên các file Python đã thêm/sửa của
module explainability, contract explanations, helper/page Streamlit, BundlePredictor
và ba file test mới, bằng các lệnh sau (Bash hoặc Zsh):

```bash
EXPLAIN_FILES=(
  src/churn_retention/ml/explainability/__init__.py
  src/churn_retention/ml/explainability/linear.py
  src/churn_retention/contracts/http/explanations.py
  tests/test_explanation_contracts.py
  tests/test_explainability.py
  src/churn_retention/entrypoints/streamlit/__init__.py
  src/churn_retention/entrypoints/streamlit/explanation_views.py
  src/churn_retention/adapters/predictor/bundle_predictor.py
  apps/streamlit/scoring_page.py
  apps/streamlit/pages/1_Explain.py
  tests/test_explanation_views.py
)
python3 -m ruff check "${EXPLAIN_FILES[@]}"
python3 -m ruff format --check "${EXPLAIN_FILES[@]}"
python3 -m mypy "${EXPLAIN_FILES[@]}"
```

Không chạy full pytest. Test dựng logistic nhỏ ngay trong test,
kiểm cộng tính, sigmoid, mapping one-hot, thứ tự ổn định, seed/cap/provenance background,
không fit lại và từ chối model ngoài phạm vi. Contract scoring cũ được chạy để kiểm
tương thích. UI chỉ kiểm hàm thuần.
