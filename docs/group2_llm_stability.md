# Nhóm 2 — LLM/RAG trong Dashboard Cường

Kết luận: đạt bộ tiêu chí chức năng và độ ổn định bằng kiểm thử offline/mock.
Không coi đây là xác nhận SLA/quota/billing của API thật hoặc phê duyệt deploy.
Không triển khai nhóm 3; Retention/Budget và Campaigns giữ nguyên business logic.

## Phạm vi đã xác minh

- Artifact v2 được load bằng external hash pin có đúng 4.930 reference records,
  matrix 4.930 × 45. Tập ID bằng chính xác processed-train và các dòng split=train;
  không giao với validation/test. Source-train SHA-256 khớp manifest; target không
  dùng khi fit preprocessing. Test không công bố ID/feature khách hàng thật.
- Input model đúng 19 feature; customerID và observed Churn chỉ là metadata nghiệp
  vụ. Đã kiểm tra thiếu từng feature, sai categorical/type, bool trong numeric,
  NaN/Inf, số âm, tenure phân số, SeniorCitizen ngoài 0/1 và missing TotalCharges.
  Missing TotalCharges chỉ được chấp nhận khi tenure=0 theo preprocessing hiện có.
- Output class/probability/reason được validate strict. Dashboard dùng alias
  churn_prob của churn_probability; provider phải Groq/Gemini, reference_count=5
  và reference_version khớp Store. Lưu mode/reference pin/cohort digest riêng.
- Submit đơn và batch 1–5; chặn cả lúc prepare và khi worker đọc lại snapshot.
  Không autoscore CSV, không inference trên đổi tab/budget/bộ lọc hoặc rerun.
- Router/Providers giữ nguyên: Groq timeout90s, Gemini30s, hedge60s, overall120s;
  Groq SDK max_retries=0, Gemini attempts=1. Một logical row có thể gọi cả hai
  provider; tối đa5 rows không có nghĩa chỉ5 provider requests.
- Groq lỗi có Gemini fallback. Cả hai lỗi thì fail rõ ràng, không tạo probability
  giả, không báo COMPLETE. Hedge kiểm tra overlap, cancellation và đóng clients;
  Gemini lỗi trong hedge vẫn chấp nhận Groq nếu Groq còn trả output hợp lệ.
- Queue atomic claim/fencing và request idempotency được giữ; hai Submit đồng thời
  chỉ gọi generation một lần. Không có retry ngoài Router. UNKNOWN sau crash không
  tự replay. Batch fail-fast giữ successful subset, các row còn lại là NOT_RUN.

## Lỗi đã sửa / ranh giới được siết

1. **Snapshot làm tròn số:** pandas.to_json mặc định10 chữ số thập phân làm thay
   giá trị float và có thể làm đổi query vector. Serializer mới dùng native JSON
   trên validated records để round-trip chính xác. Test so sánh từng float và
   vector bằng array_equal. Snapshot/kết quả cũ không được ghi lại hay suy đoán sửa.
2. **Mode không hợp lệ có thể vượt live gate:** Literal là type hint, không phải
   runtime validation. Mode unknown trước đây rơi vào nhánh predict_customer.
   Nay reject mode ngoài mock/live và kiểm tra lại live opt-in trước execution.
3. **Bypass cap qua persisted run:** prepare chặn5 nhưng process trước đây không
   chặn khi run được tạo trực tiếp ở ledger. Worker nay kiểm tra1–5, snapshot hash,
   schema và provenance trước provider invocation. Corrupt run fail closed.
4. **Replay failed run báo job thành công:** process từng trả dict status=FAILED;
   generic worker xem return bình thường là SUCCESS. Nay FAILED/PARTIAL raise safe
   error; chỉ COMPLETE được replay như success idempotent, không generation mới.
5. **Thông tin lỗi quá chung:** thêm allowlisted error codes cho configuration,
   rate limit/quota, timeout/deadline, connection, invalid response, validation và
   provenance. Table dashboard_failures mới chỉ chứa code, không raw SDK message,
   keys, response hoặc prompt; không ALTER/DROP hay rewrite bảng dữ liệu cũ.
6. **Nhãn và consent không nhất quán:** giữ lựa chọn live đã có, nhưng banner,
   header/report/export label lấy mode của kết quả đã lưu, không dùng mode draft.
   Consent reset khi đổi provider/cohort/pin và khi chủ động tạo request mới.
   Campaign export dùng mode của chính campaign. Không sửa thuật toán Campaigns.

Dashboard hiển thị lỗi lần Submit gần nhất và trạng thái từng row qua rerun.
Nếu giữ snapshot thành công cũ, snapshot đó vẫn mang run/mode riêng; không được
trình bày như thành công của Submit vừa lỗi. Không fallback về FakePredictor hay
classifier/model bundle khi LLM hoặc artifact lỗi.

## Kiểm thử

| Kiểm tra | Kết quả |
|---|---|
| Bộ stability mới | 58 passed |
| Dashboard integration + service + standalone + stability trên repository chính | 115 passed |
| Full regression, strict private-assets mode | 1099 passed, 15 warnings như baseline |
| Golden | Matrix/vectors/top5/order/distances/tie/self-exclusion/8 prompt hashes pass, không sửa fixture |
| Import isolation | Standalone và primary Dashboard minimal deployment pass; chặn classifier/bundle/training/v1/refit/SDK thật/network |
| Ruff check/format phạm vi thay đổi | Pass, 9 Python files |
| Mypy normal imports phạm vi thay đổi | Pass, 9 source files |

Các kiểm thử provider dùng real service/Router/provider adapters với fake SDK I/O;
không đổi thời gian cấu hình để chờ thật60/120 giây mà dùng controllable clock.
Kiểm tra kwargs thực tế của SDK và số generation attempts/cleanup. Không gọi API thật.

Full suite chạy trên snapshot có executable/config files khớp mã repository chính,
để các legacy tests ghi SQLite/MLflow không tác động dữ liệu dự án. Các test liên
quan chạy trực tiếp trên repository chính, temp state/cache nằm ngoài repository.
File P1 docs/llm_predictor.md không sửa và không đưa modified content vào test snapshot.
15 warnings là bundle metadata/dirty-tree và SQLAlchemy deprecation có sẵn.
Không tự sửa notebook/docs ngoài phạm vi để làm xanh lint toàn repository.

## Giới hạn còn lại

- Không gọi API thật: chưa xác nhận health/quota/latency/billing với credentials
  thật. Hedge cancellation không bảo đảm provider chưa tính phí.
- Probability chưa hiệu chuẩn; hiệu quả hành động và ROI vẫn là mô phỏng.
- Không mở rộng auth/tenant, worker fleet, deployment hoặc migration dữ liệu cũ.
  Recovery job bị ngắt vẫn cần xác nhận worker đã dừng, không tự gọi API lại.
- Existing failed rows không có error code cũ chỉ được hiển thị lỗi generic;
  không suy đoán hoặc backfill raw errors. Artifact v2 hợp lệ vẫn là điều kiện
  mở prediction flow, không fallback v1/legacy inference.
- API contract LLMChurnPredictor, preprocessing, artifact v2, retrieval K=5,
  prompt, Router/Providers và golden fixtures không đổi.

Không commit, stage, merge, push, reset hoặc rebase. Các thay đổi tồn tại trong
working tree để review; các thay đổi chưa commit ngoài phạm vi được giữ nguyên.
