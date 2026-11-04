# customer-churn-retention

Dashboard hỗ trợ giữ chân khách hàng, dùng **LLM/RAG v2** (Groq/Gemini) kết hợp
phân tích rời bỏ và lập kế hoạch chiến dịch. Provider Qwen/Groq và Gemini, model,
router và timeout được giữ nguyên. Xác suất LLM chưa hiệu chuẩn; campaign và ROI hiện là mô phỏng.

---

## ▶ Khởi chạy nhanh (dành cho thành viên nhóm)

Thành viên có quyền truy cập repo private chỉ cần:

### 1. Điều kiện tiên quyết (Prerequisites)
- Máy đã cài sẵn **Python 3.12, 3.13 hoặc 3.14**.
- Kết nối mạng để tải các gói dependency trong lần đầu chạy.

### 2. Lệnh duy nhất để khởi chạy
```bash
git clone https://github.com/dungngo2612/customer-churn-retention.git
cd customer-churn-retention
bash run.sh
```

**Kết quả:**
- Script tự tạo môi trường ảo `.venv`, cài đặt dependency khóa phiên bản theo `constraints/rag-v2.txt`.
- Tự động nạp cấu hình dùng chung và API key tài khoản free của nhóm từ `.env.shared`.
- Kiểm tra toàn vẹn SHA256 của RAG artifact (`telco-train-v2.zip`).
- Khởi tạo database và chạy migration schema qua Alembic.
- Mở Dashboard tại **http://127.0.0.1:8501** với **chế độ LLM thật đã sẵn sàng**. Không cần sửa file, không cần tạo `.env`, không cần nhập key thủ công.

---

## Các tuỳ chọn bổ sung của `run.sh`

| Lệnh / Tuỳ chọn | Mục đích |
|---|---|
| `bash run.sh` | Chạy mặc định với LLM thật (Groq primary + Gemini fallback) |
| `bash run.sh --mock` | Chạy offline chế độ mô phỏng (không gọi API ngoài, tiết kiệm quota) |
| `bash run.sh --port 8502` | Khởi chạy trên cổng chỉ định thay vì 8501 |
| `bash run.sh --setup-only` | Chỉ tạo virtualenv, cài dependency, chạy migrations mà không mở web |
| `bash run.sh --help` | Xem toàn bộ danh sách tham số |

---

## Ghi đè cấu hình cá nhân (Tuỳ chọn)

Nếu bạn muốn sử dụng API key hoặc cơ sở dữ liệu riêng, hãy tạo file `.env.local` tại thư mục gốc dự án (file này được `.gitignore` bảo vệ, không bao giờ được commit):

```bash
# .env.local — ghi đè cấu hình cá nhân (tuỳ chọn)
GROQ_API_KEY=<key_groq_rieng_cua_ban>
GEMINI_API_KEY=<key_gemini_rieng_cua_ban>
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/churn_db
```

---

## Hướng dẫn cài đặt Python (nếu máy chưa có)

| Hệ điều hành | Lệnh cài đặt Python 3.12 |
|---|---|
| **macOS** | `brew install python@3.12` (hoặc tải installer từ [python.org](https://www.python.org/downloads/)) |
| **Linux (Ubuntu/Debian)** | `sudo apt update && sudo apt install -y python3.12 python3.12-venv` |
| **Windows** | Cài đặt [WSL2](https://learn.microsoft.com/vi-vn/windows/wsl/install) rồi chạy lệnh của Ubuntu |

---

## Các thành phần đã tích hợp sẵn trong repo

| Thành phần | Đường dẫn | Ghi chú |
|---|---|---|
| Cấu hình dùng chung & API keys | `.env.shared` | Tracked trong repo private, nạp tự động bởi `run.sh` |
| RAG v2 Reference Artifact | `artifacts/references/telco-train-v2.zip` | Checksum SHA256 được ghim và tự động xác thực |
| Dataset phân chia & huấn luyện RAG | `artifacts/splits/dataset_splits.csv` | Bảng phân chia train/test/val |
| Dữ liệu thô | `data/telco_churn.csv` | Bộ dữ liệu khách hàng Telco |
| Khóa phiên bản dependency | `constraints/rag-v2.txt` | Cố định đúng phiên bản numpy/pandas/scikit-learn |
| Database migrations | `migrations/` & `alembic.ini` | Đồng bộ schema tự động qua `alembic upgrade head` |

---

## Nền tảng đã kiểm chứng & Giới hạn

- ✅ **macOS (Apple Silicon M-series, Intel):** Đã kiểm chứng hoạt động hoàn hảo với Python 3.12 và Python 3.14.
- ✅ **Linux (Ubuntu 22.04 / Debian 12):** Đã kiểm chứng môi trường bash, venv, alembic và Streamlit.
- ⚠ **Windows (Native PowerShell/cmd):** Chưa kiểm chứng native; khuyến nghị sử dụng môi trường WSL2 (Ubuntu).

---

## Bảo toàn bản tự train

Toàn bộ mã Logistic Regression, CatBoost, tuning, calibration, classifier bundle, SHAP và CLI cũ đã được lưu trữ an toàn tại nhánh:
[codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02).
Main tập trung vào LLM/RAG v2 kết hợp Dashboard điều hành giữ chân khách hàng.
