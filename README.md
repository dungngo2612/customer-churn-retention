# customer-churn-retention

Dự án dự đoán khách hàng rời bỏ và hỗ trợ giữ chân. Dashboard của Cường là giao diện tích hợp chính, dùng LLM/RAG v2; có thêm luồng ML truyền thống, batch scoring, quản lý dataset và các công cụ đánh giá. Kết quả LLM chưa hiệu chuẩn; chiến dịch và ROI hiện là mô phỏng.

Trạng thái task, branch, stash và bằng chứng kiểm tra được cập nhật trong [báo cáo tích hợp](docs/cuong_dashboard_llm.md#rà-soát-tích-hợp-main-ngày-2026-11-02).

## Cài môi trường

Sử dụng CPython 3.12–3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,training]"
cp .env.example .env
```

Điền `DATABASE_URL` trong `.env` khi cần kết nối PostgreSQL.

## Phân công thư mục

| Phụ trách | Thư mục | Phạm vi |
| --- | --- | --- |
| Người 1 | `src/churn_retention/models/`, `explainability/` | Huấn luyện, dự đoán, SHAP |
| Người 2 | `src/churn_retention/data/`, `evaluation/` | Xử lý dữ liệu, đặc trưng, kiểm chứng, phân tích lỗi |
| Người 3 | `src/churn_retention/retention/`, `api/`, `dashboard/`, `storage/` | CLV, chiến lược giữ chân, tích hợp, giao diện, lưu trữ |
| Cả nhóm | `src/churn_retention/contracts/`, `notebooks/`, `tests/`, `data/`, `artifacts/`, `docs/` | Schema chung và tài nguyên dùng chung |

Người 3 chủ trì tích hợp các phần của nhóm.

## Streamlit LLM + RAG độc lập

Chạy từ repository root:

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit_llm/app.py
```

Ứng dụng chỉ đăng ký trang Predict, sử dụng Reference Artifact v2 với hash pin
hiện có và không khởi chạy Scoring/Explain. Cấu hình artifact, khóa provider và
giới hạn được mô tả trong [hướng dẫn app độc lập](docs/llm_standalone_app.md).

### Dashboard demo một quản trị viên

```bash
PYTHONPATH=src python3 -m churn_retention.dashboard.demo
```

Chỉ `http://127.0.0.1:8501`, mock mặc định và SQLite demo riêng. Không có
authentication hoặc cách ly phiên. Không dùng tunnel/reverse proxy/public deploy.
Xem [phạm vi bảo mật demo](docs/group5_single_admin_security.md).


## API và kiểm tra

```bash
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
python3 -m churn_retention.entrypoints.cli --help
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src
python3 -m pytest
```

Trước khi dùng API dataset, cấu hình `DATABASE_URL` rồi chạy
`python3 -m alembic upgrade head`. API có `/health` và `/datasets`.
`calibrate`, `tune`, `score-batch`, `job-status` là các lệnh CLI riêng.
Golden strict cần các artifact private local đã pin; không commit các artifact đó.
CI kiểm tra migration trên PostgreSQL riêng. Không dùng database sản xuất cho tests.
