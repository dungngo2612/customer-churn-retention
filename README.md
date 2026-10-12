# customer-churn-retention

Khung dự án dự đoán khách hàng rời bỏ và hỗ trợ giữ chân. Hiện chỉ có cấu trúc và cấu hình, chưa có logic ứng dụng.

## Cài môi trường

Sử dụng CPython 3.12–3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
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
