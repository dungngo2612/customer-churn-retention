# RAG reference data / split audit

Phạm vi main sau tách: schema dữ liệu và membership của RAG preprocessing/reference.
Báo cáo training Logistic/CV cũ được bảo toàn ở [codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02); main không còn trainer.

Source: `data/telco_churn.csv`, 7043 dòng; split manifest
`artifacts/splits/dataset_splits.csv`, seed 42. Train 4930 (1308 churn), val 1056
(280 churn), test 1057 (281 churn). ID duy nhất, không rỗng; mỗi ID thuộc đúng một split.

SHA-256 theo byte chuẩn hóa CRLF/CR sang LF, không parse/reserialize:

- Source: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- Splits: `bced80937bf69798aa314d03bd35d59d6c14134a593997d3e3dd8c7fe8b6512c`.

Schema canonical của RAG nằm ở `adapters/retrieval/rag_preprocessor.py`:
19 feature, loại customerID/Churn khỏi đầu vào fit/transform. Fit preprocessing chỉ
nhận đúng train IDs; builder reference đọc nhãn để lưu few-shot examples sau đó.
Không train classifier, không fit trên val/test, không dùng nhãn query để retrieval.

`notebooks/explore_and_split.py`, `scripts/make_splits.py`, `scripts/prepare_splits.py`
được giữ vì phục vụ dữ liệu RAG. Không chạy ghi lại các dữ liệu local trong đợt tách.
`tests/test_leakage_audit.py` giữ inventory schema, seed, row/churn counts, membership,
stratification và LF checksums. Giới hạn: dữ liệu snapshot không chứng minh feature
đã có sẵn tại mọi thời điểm dự đoán hoặc hiệu quả kinh doanh ngoài thực tế.
