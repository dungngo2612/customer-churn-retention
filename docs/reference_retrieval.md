# RAG v2: preprocessing và retrieval

Main chỉ dùng `ReferenceStore` v2. Loader/builder v1 dựa vào classifier bundle đã
được gỡ; bản lịch sử nằm ở [codex/archive-self-training-2026-11-02](https://github.com/dungngo2612/customer-churn-retention/tree/codex/archive-self-training-2026-11-02).

## Build offline

```bash
PYTHONPATH=src python3 -m churn_retention.workflows.build_reference_v2 \
  --train data/processed/train.csv \
  --splits artifacts/splits/dataset_splits.csv \
  --out artifacts/references/custom-v2.zip
```

`build_reference_v2(train_path, splits_path, output_path)` fit RAG preprocessing
chỉ trên toàn bộ train IDs trong manifest. Fit này không học classifier và không
sử dụng Churn làm đầu vào. CSV reference vẫn giữ nhãn No/Yes để tạo few-shot examples.
Không ghi đè CSV, manifest hoặc output đã có. Vocab/numerical state được lưu bằng JSON.

- Đúng 19 feature; ID/Churn không đi vào vector hoặc query prompt.
- ID không rỗng/trùng, khớp toàn bộ reference train IDs; không nhận val/test IDs.
- Numerical hữu hạn, không âm; SeniorCitizen 0/1; tenure nguyên.
- TotalCharges missing chỉ thành 0 khi tenure=0; categorical phải thuộc vocab.
- Artifact chứa schema, JSON preprocessing/fit provenance, records.jsonl, matrix.npy,
  splits và manifest hashes; không có model.pkl hoặc classifier.
- Runtime load artifact một lần, kiểm tra external pin, payload checksum và alignment,
  rồi chỉ transform query. Không fit lại, không đọc CSV hoặc classifier khi request.

Retriever dùng khoảng cách Euclidean, mặc định K=5; tie-break theo customer ID,
loại chính customer query và clamp K theo số reference hợp lệ. Nhãn không mã hóa
vào vector. Prompt chỉ nhận features và nhãn tham chiếu, không nhận ID/distance.
Xem [định dạng v2](reference_artifact_v2.md) và [runtime](llm_runtime_v2.md).

## Regression

```bash
python3 -m pytest tests/test_reference_retrieval.py tests/test_reference_retrieval_integration.py tests/test_reference_v2.py tests/test_rag_preprocessor.py tests/test_llm_rag_golden.py
```

Fixtures tổng hợp dùng RAG v2. Kiểm tra dữ liệu thật đối chiếu ma trận, query vectors,
neighbor IDs/distances và full prompt hashes với oracle v1 bất biến. Chỉ đọc JSON/NPY
của oracle; không deserialize classifier hoặc tái tạo golden từ output hiện tại.
