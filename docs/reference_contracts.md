# Neutral RAG contracts

`src/churn_retention/contracts/reference.py` giữ Scalar, ChurnLabel, FeatureColumn,
StoredRecord, ReferenceRecord, PreparedQuery, SimilarCustomer, validation, JSON
serialization và SHA-256. Module không import adapter, sklearn hoặc training.
`contracts/reference_source.py` là protocol để retriever không phụ thuộc loader.

Prompt Builder và Retriever dùng các contract này. Public `ReferenceStore` alias
loader v2; builder `build_reference_v2` dùng schema của RAGPreprocessor. Không còn
v1 classifier-bundle loader hoặc compatibility exports của `_artifact.py` trên main.

Input validation và projection không đổi: 19 feature, không ID/Churn trong vector;
ID nội bộ được loại khỏi prompt; chỉ reference labels được dùng làm few-shot.
Test `tests/test_reference_contracts.py` giữ kiểm tra contract identity còn hỗ trợ,
validation và import isolation. Phần compatibility riêng cho loader v1 đã được loại.
