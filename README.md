# customer-churn-retention

Dashboard hỗ trợ giữ chân khách hàng, dùng **LLM/RAG v2** (Groq/Gemini) kết hợp
phân tích churn và kế hoạch chiến dịch. Provider Qwen/Groq và Gemini, model,
router và timeout được giữ nguyên. Xác suất LLM chưa hiệu chuẩn; campaign và ROI hiện là mô phỏng.

---

## ▶ Chạy nhanh (dành cho thành viên nhóm)

**Yêu cầu duy nhất:** Python 3.12, 3.13 hoặc 3.14 đã cài sẵn.

```bash
git clone https://github.com/dungngo2612/customer-churn-retention.git
cd customer-churn-retention
bash run.sh
```

Dashboard sẽ mở tại **http://127.0.0.1:8501** — dùng chế độ mock (không tốn quota API).

### Chạy với LLM thật (Groq + Gemini)

```bash
bash run.sh --live
```

Nhập `ENABLE LIVE API` khi được hỏi để xác nhận. API key của nhóm đã được cấu hình
sẵn trong `.env.shared` (được quản lý trong repo private).

### Cài đặt thủ công (nếu cần)

```bash
python3 -m pip install virtualenv       # nếu chưa có
bash run.sh --setup-only                # chỉ cài môi trường
```

### Ghi đè cấu hình cá nhân (tuỳ chọn)

Tạo `.env.local` tại gốc repo (không được commit) để ghi đè bất kỳ biến nào từ `.env.shared`:

```bash
# .env.local — không commit file này
GROQ_API_KEY=<key_của_bạn>
GEMINI_API_KEY=<key_của_bạn>
```

---

## Cài đặt Python

| Hệ điều hành | Cách cài Python 3.12 |
|---|---|
| macOS | `brew install python@3.12` hoặc [python.org/downloads](https://www.python.org/downloads/) |
| Ubuntu/Debian | `sudo apt install python3.12 python3.12-venv` |
| Windows | Cài [WSL](https://docs.microsoft.com/windows/wsl/install) rồi dùng hướng dẫn Ubuntu |

---

## Tham số run.sh

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--live` | mock | Bật live LLM API (gọi Groq/Gemini thật) |
| `--port PORT` | 8501 | Cổng Streamlit |
| `--state-dir DIR` | tạo tạm | Thư mục lưu trạng thái demo |
| `--setup-only` | — | Chỉ cài môi trường, không khởi động |
| `--help` | — | Hướng dẫn |

---

## Những gì đã được tích hợp sẵn

| Thành phần | Nơi lưu | Ghi chú |
|---|---|---|
| API keys Groq + Gemini | `.env.shared` | Key tài khoản free của nhóm |
| Artifact RAG v2 | `artifacts/references/telco-train-v2.zip` | SHA256 được pin và kiểm tra khi khởi động |
| Dependency lock | `constraints/rag-v2.txt` | Giữ NumPy/Pandas/scikit-learn đúng phiên bản |
| Database demo | Tạo tự động | SQLite tạm tạo mỗi lần chạy, không commit |

---

## Nền tảng đã kiểm chứng

- ✅ macOS (Intel, Apple Silicon) — Python 3.12
- ✅ Ubuntu 22.04 — Python 3.12
- ⚠ Windows — chạy qua WSL (chưa test native)

---

## Cấu trúc chính

```
run.sh                          # Script khởi chạy một lệnh
.env.shared                     # Cấu hình nhóm (API keys, artifact path)
.env.local                      # Ghi đè cá nhân — KHÔNG commit
artifacts/references/
  telco-train-v2.zip            # Artifact RAG v2 (được track trong repo)
src/churn_retention/
  dashboard/                    # Giao diện Streamlit 5 workspace
  modules/scoring/              # LLM predictor, router, prompt
  adapters/llm/                 # Groq, Gemini provider
```

---

## Bản tự train đã được bảo toàn

Toàn bộ mã Logistic Regression/CatBoost, tuning, calibration, classifier bundle,
SHAP và CLI cũ được giữ tại nhánh
[codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02).
Main không còn chức năng train/load/predict classifier.

---

## API dataset và kiểm thử

```bash
python3 -m alembic upgrade head
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
python3 -m pytest
```

CI dùng PostgreSQL tạm và tests mock/offline; không dùng database sản xuất.
Xem [báo cáo tích hợp Dashboard](docs/cuong_dashboard_llm.md) và
[phạm vi demo một admin](docs/group5_single_admin_security.md).

---

## RAG artifact và app Predict độc lập

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit_llm/app.py
```

Pin SHA256: `c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8`.
Xem [artifact v2](docs/reference_artifact_v2.md), [retrieval](docs/reference_retrieval.md)
và [app độc lập](docs/llm_standalone_app.md).
