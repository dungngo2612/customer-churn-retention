# customer-churn-retention

Main sử dụng **LLM/RAG v2 và Dashboard của Cường** để hỗ trợ giữ chân khách hàng.
Provider Qwen/Groq và Gemini, model, router và timeout được giữ nguyên.
Xác suất LLM chưa hiệu chuẩn; campaign, can thiệp và ROI hiện là mô phỏng.

## Bản tự train đã được bảo toàn

Toàn bộ mã Logistic Regression/CatBoost, tuning, calibration, classifier bundle,
SHAP và CLI cũ được giữ tại nhánh [codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02),
từ commit `10d72b9559754d6ecc363a8e2210c619551f0e47`. Nhánh đã được push.
Main không còn chức năng train/load/predict classifier hoặc fallback về model bundle.
Artifact classifier local hiện có được giữ nguyên trên máy và không còn được tracking.
Các báo cáo kiểm thử theo bước cũ là lịch sử; hướng dẫn chạy main hiện tại nằm
trong README này và [báo cáo tích hợp Dashboard](docs/cuong_dashboard_llm.md).

## Cài đặt

Dùng CPython 3.12–3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -c constraints/rag-v2.txt -e ".[dev]"
```

NumPy/scikit-learn vẫn cần cho preprocessing, retrieval và metrics dùng chung;
fit scaler/encoder trên reference train là xây RAG, không phải train classifier.
Không còn dependency CatBoost, SHAP, MLflow hay extra `training`.
`constraints/rag-v2.txt` giữ NumPy/Pandas/scikit-learn đúng phiên bản của artifact
và golden đã duyệt. Loader kiểm tra phiên bản thư viện; không tự nâng rồi nạp
artifact cũ hoặc tái tạo golden để bỏ qua sai khác.

## Dashboard chính

Chạy từ repository root:

```bash
PYTHONPATH=src python3 -m churn_retention.dashboard.demo
```

Launcher chỉ bind `127.0.0.1:8501`, dùng mock mặc định và database demo riêng.
Không tự inference trên load/rerun. Chọn 1–5 ID, xác nhận rồi Submit qua Queue/Worker.
Dashboard có bốn tab: kinh tế, hành vi/LLM reason, Customer 360, ngân sách/campaigns.
Lưu campaign và xuất CSV là mô phỏng, chưa thực thi ưu đãi thật.
Live API cần opt-in server và xác nhận rõ ràng; không có lựa chọn classifier.
Xem [hướng dẫn tích hợp](docs/cuong_dashboard_llm.md) và
[phạm vi demo một admin](docs/group5_single_admin_security.md).

## RAG artifact và app Predict độc lập

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit_llm/app.py
```

Cấu hình artifact v2 qua `CHURN_LLM_REFERENCE_ARTIFACT` và hash pin được phê duyệt
`CHURN_LLM_REFERENCE_SHA256`. Khi cài từ wheel hoặc chạy ngoài checkout, dùng đường
dẫn tuyệt đối. Artifact private được cung cấp riêng, không nhúng trong wheel.
Default trong checkout: `artifacts/references/telco-train-v2.zip`;
pin: `c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8`.

Build offline từ dữ liệu reference và manifest hiện có, với output mới:

```bash
PYTHONPATH=src python3 -m churn_retention.workflows.build_reference_v2 \
  --train data/processed/train.csv \
  --splits artifacts/splits/dataset_splits.csv \
  --out artifacts/references/custom-v2.zip
```

Builder fit preprocessing RAG, không nhận classifier bundle. Không ghi đè output.
Giữ dữ liệu/splits local hiện có; chỉ chạy scripts chuẩn bị dữ liệu khi chủ động cần.
Xem [artifact v2](docs/reference_artifact_v2.md), [retrieval](docs/reference_retrieval.md)
và [app độc lập](docs/llm_standalone_app.md).

## API dataset và kiểm thử

Cấu hình `DATABASE_URL` cho database riêng (có thể dùng `.env.example` làm mẫu), rồi:

```bash
python3 -m alembic upgrade head
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src
python3 -m pytest
python3 -m build
```

API có `/health`, upload/list/detail/quality của `/datasets`. Không còn API/CLI
scoring classifier, lệnh train/tune/calibrate, hoặc trang Explain SHAP.
Batch 1–5 ID của LLM được giữ trong Dashboard/ledger, không chuyển sang hệ batch cũ.

Regression strict cần artifact/golden private local đã pin:
`CHURN_GOLDEN_REQUIRE_ASSETS=1 python3 -m pytest`.
CI dùng PostgreSQL tạm và tests mock/offline; không dùng database sản xuất hoặc API trả phí.
[Báo cáo tách classifier và kết quả kiểm chứng](docs/cuong_dashboard_llm.md#tách-classifier-khỏi-main).
