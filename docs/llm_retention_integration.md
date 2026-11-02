# MVP Retention/Budget sau LLM + RAG v2

Chỉ tích hợp đến bước 2: accepted PredictionRecord, adapter trung tính và UI mô phỏng.
Không có Campaigns, database persistence, Queue/Worker, batch scoring hay TreeSHAP mới.

## Chạy demo

Từ root của bản sao đã triển khai:

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit_llm/app.py --server.port 8502
```

Mở `http://127.0.0.1:8502/retention` hoặc chọn **Mô phỏng Retention / Budget**.
Trang mặc định dùng **Dữ liệu giả (demo)**: bốn khách hàng hư cấu và xác suất viết tay,
không gọi API, không đọc dataset hoặc artifact. Hai khách đủ điều kiện; một khách đã
rời bỏ và một khách tenure=0 được loại khỏi mô phỏng. Thay ngân sách về 0 để xem A0;
tăng hệ số chi phí liên hệ lên 1000 để xem tình huống mọi khách chọn A0.

Các con số demo không phải dự đoán thực tế hoặc kết quả của Groq/Gemini. Chúng có
origin=synthetic, provider=None, reference_count=0, reference_version=synthetic-demo-v1,
không có hash pin của artifact thật. Demo không ghi đè accepted record của Predict.

Trang Predict giữ luồng LLM có sẵn. Task này chỉ kiểm thử luồng đó bằng mock providers;
không gọi API thật. Không cần nhập/cấu hình API keys để dùng trang demo.

## Record và adapter

- `PredictionRecord` frozen: UUID, UTC timestamp, đúng 19 feature dưới dạng tuple
  chứa scalar, LLM DTO frozen, origin, reference pin và config ID. Snapshot trả ra
  qua `features()` là bản sao; không thể thay đổi record bằng cách sửa dict này.
- Hook sau submit thành công cấp UUID đúng một lần; submit giống nhau lần nữa có UUID
  khác. Rerun/chuyển trang/thay budget không capture hoặc gọi lại LLM. Submit/artifact
  error xóa accepted context hiện hành cùng với kết quả Predict, không tạo record mới.
- `config_id=llm-rag-v2-defaults@7341854` là định danh contract ứng dụng ở baseline,
  KHÔNG phải phiên bản model provider thực tế. DTO hiện không có actual model version.
- Customer ID và observed_active nằm trong CustomerContext riêng, không vào 19 feature,
  prompt hoặc LLM DTO. UI yêu cầu xác nhận trạng thái cho mỗi record mới. Chỉ chọn
  trạng thái khi có căn cứ nghiệp vụ; không suy ra từ churn_prediction.
- Adapter tạo hai DataFrame tương thích engine. Cột Churn chỉ là phép chiếu nội bộ
  từ observed_active (`True -> No`, `False -> Yes`), không phải predicted label,
  không ghi nhãn ngược vào dataset. Unknown status, ID/run trùng hoặc trộn dữ liệu
  synthetic và LLM trong một simulation bị từ chối.
- Engine dùng nguyên probability LLM; không đổi label, threshold, calibration,
  preprocessing, RAG, K=5, prompt hay chính sách Router/Providers.
- Budget/margin/horizon/contact-cost multiplier được validate kiểu, phạm vi và
  finite values; chặn cả tràn số trong economics. Không can thiệp vào công thức.

## Nguồn code Cường

Pin: `origin/demo-test-04@911a60a028cb89507d35fbe584059f30f7ee2da4` (Cuogphi/Cường).

Lấy có chọn lọc `modules/retention/{policy,use_cases,budget}.py` và hai test trung tính
`tests/test_retention.py`, `tests/test_budget.py`; không merge/cherry-pick nhánh.
`docs/retention_source.json` ghi SHA-256 bytes gốc trước khi port. Thay đổi ở phần port
chỉ gồm format/import/type syntax, bỏ unused imports và hai bool keys trùng với 1/0
trong mapping; vẫn chấp nhận cả bool và int. Formulas, ranking/tie-breaking và greedy
allocation được giữ. Đối chiếu source pin với bản port trên 24 kịch bản, mỗi kịch bản
ba bảng evaluate/recommend/allocate, cho kết quả byte-identical sau cùng serialization.
Không dùng helper CLV đã làm tròn để tính lại kết quả vector của engine.

Điều kiện mặc định của các action lấy từ `default_actions()`; bảng giả định UI dùng
chính các action đưa vào engine, không dùng catalog thứ hai. Hệ số chi phí chỉ thay
tham số contact_cost, không thay thuật toán.

## Session, cache và phạm vi

Entrypoint dùng allowlist đúng hai trang: Predict và Retention. Không auto-discovery
legacy pages; không import nguyên Dashboard Cường. `active_scoring_context` giữ nguyên.

Accepted record gắn đúng snapshot đã submit. Bản nháp Predict không thay record.
Khi dùng kết quả LLM, trang mô phỏng kiểm tra reference fingerprint và external pin
so với context đã capture; thiếu/thay artifact hoặc pin thì fail closed. Không load
artifact v1, không nạp model bundle và không tạo prediction để tự phục hồi.

Không thêm cache/persistence dùng chung cho khách hàng. Records/context và demo cohort
chỉ ở session state; kết thúc session mất dữ liệu. Chưa có cohort thật/bulk LLM/history,
CRM integration, lưu campaign hoặc export. Danh tính local-customer là mã cục bộ do UI
cung cấp cho một mô phỏng, không phải ID đã được đối chiếu CRM.

Probability LLM chưa hiệu chuẩn, efficacy/acceptance là giả định, proxy_value không phải
CLV thực tế. Kết quả chỉ là mô phỏng theo giả định, không phải ROI được kiểm chứng.
Greedy là heuristic, không bảo đảm nghiệm tối ưu toàn cục.

## Kiểm thử

```bash
PYTHONPATH=src python3 -m pytest tests/test_retention.py tests/test_budget.py tests/test_llm_retention.py tests/test_llm_retention_page.py tests/test_llm_standalone_app.py -q
```

Các fixture thêm mới đều synthetic, không cần API thật. Test standalone export deployment
chỉ có runtime v2 cần thiết, bỏ bundle/v1/training/SHAP/worker/campaigns; process sạch
chặn imports legacy, classifier construction/unpickle/refit và sockets. Luồng Retention
được kiểm tra qua Router thật với fake providers: status không suy từ label, rerun và
đổi budget không gọi lại LLM, submit mới có UUID mới nhưng prompt cũ không bị thêm
metadata, reference pin thay đổi chặn kết quả cũ. Test A0/rỗng tránh lỗi idxmax cũ.

Full regression bao gồm golden private có sẵn, không cập nhật fixture: matrix 4930×45,
8 query vectors, top-5/thứ tự/distances/tie/self-exclusion và 8 prompt hashes. Private assets
chỉ dùng để kiểm tra regression, không phải dữ liệu demo và không đưa vào Git.

Độc lập ở đây là runtime/import. pyproject.toml vẫn khai báo các dependencies ML chung
của toàn repository; MVP không sửa dependency packaging hay deploy.
