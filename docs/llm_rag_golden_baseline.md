# Golden RAG: oracle v1, runtime v2

Golden đã được ghi trước refactor, không được sinh lại hoặc sửa trong đợt tách này.
Code capture/loader classifier v1 nằm ở [codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02).

- Public: `tests/fixtures/llm_rag_v1_golden.json`, không có raw IDs/query vectors.
- Private local: `artifacts/golden/llm-rag-v1-private.json`, mode 0600, không commit.
- Oracle local: `artifacts/references/telco-train-v1.zip`; chỉ đọc records/matrix/manifest
  bằng JSON/NPY. Không mở bundle/model.pkl hoặc deserialize classifier.
- Runtime được kiểm tra: artifact v2 với external pin hiện có.
- Helpers: `scripts/verify_rag_golden.py`; không còn CLI capture golden.

`tests/test_llm_rag_golden.py` load v2 dưới guard cấm fitting và network; giữ so sánh
19 feature, 45 cột transformed, toàn bộ matrix/hash/row order, tám query vectors,
K=5 IDs/distances/tie-breaking, missing values, cấu trúc và hash đầy đủ của prompt.
Snapshot preprocessing của v2 vẫn phải khớp oracle cũ. Test private so sánh vectors/IDs
trực tiếp và xác nhận private file bị Git ignore. Hash public fixture được khóa.

```bash
CHURN_GOLDEN_REQUIRE_ASSETS=1 python3 -m pytest tests/test_llm_rag_golden.py tests/test_reference_v2.py tests/test_rag_preprocessor.py
python3 scripts/verify_rag_preprocessor.py
python3 scripts/verify_reference_v2.py
```

Không dùng output của code đang kiểm tra để thay oracle expected. CI không có assets
private sẽ skip rõ ràng các kiểm tra đó; local strict-assets phải có đủ file.
