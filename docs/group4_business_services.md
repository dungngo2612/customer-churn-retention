# Nhóm 4 — Business services của Dashboard Cường

Ngày kiểm tra: 2026-11-02. Repository chính, nhánh `feat/ml-batch-scoring`,
HEAD `7341854d2aedf94103916a3a59b696482c47eb30`.

Kết luận: **ĐẠT trong phạm vi Dashboard hỗ trợ quyết định và chiến dịch mô phỏng**,
các kiểm tra bên dưới đã hoàn tất. Không xác nhận thực thi can thiệp thật,
ROI thực nghiệm, SLA API, worker fleet hay điều kiện deploy.
Không thực hiện nhóm 3, 5, 6 hoặc 7; không gọi API thật.

## Luồng nghiệp vụ đã kiểm tra

1. Dashboard nhận kết quả đã chấp nhận của LLM/RAG v2 từ SQLite ledger. Mỗi
   prediction giữ customer ID, run ID, provider, mode mock/live, reference version
   và external hash pin. Không dùng CatBoost làm predictor chính.
2. `customers.Churn` là trạng thái đã quan sát do người dùng cung cấp; không suy
   từ `churn_prediction`. Chỉ khách Churn=No và tenure>0 đủ điều kiện Retention.
   Tab Churn hiển thị tỷ lệ đã quan sát trong snapshot, không phải tỷ lệ dự đoán
   hoặc ước lượng cho toàn bộ tập train. Mẫu demo được ghi rõ dữ liệu giả.
3. Executive Summary/Customer 360 dùng các giả định kinh tế hiện hành:
   - V = MonthlyCharges gốc × horizon × contribution margin (proxy value).
   - C = contact_cost + acceptance_rate × offer_cost.
   - r = acceptance_rate × efficacy; B = p × r × V; ΔEV = B − C.
   - p* = C/(r×V), không xác định khi mẫu số bằng 0.
   - Chỉ chọn can thiệp có ΔEV>0; đúng hòa vốn chọn A0. Không trừ ưu đãi hai lần.
   - ROI mô phỏng = tổng ΔEV/tổng C ×100%; C=0 hiển thị N/A, không báo ROI=0%.
   - Customer 360 hiển thị label LLM, probability, reason, provider/reference và
     observed Churn riêng; không tạo SHAP cho LLM. Danh sách khuyến nghị này chỉ
     gồm khách đủ điều kiện; tab 02 vẫn hiển thị kết quả của toàn snapshot.
4. Budget dùng greedy heuristic hiện có, không tuyên bố tối ưu toàn cục. Mỗi
   khách tối đa một hành động; thứ tự và tie-breaking giữ nguyên. Brute-force trên
   tập nhỏ dùng đối chiếu. Hành động miễn phí có lợi vẫn có thể chọn với budget=0.
5. Campaign lưu snapshot bất biến, liên kết đúng run/cohort/predictor/reference.
   Dashboard tính lại allocations và summary từ kết quả đã lưu trước khi chấp nhận
   lưu campaign, hoàn toàn không gọi predictor. Chấp nhận PARTIAL chỉ trên subset
   SUCCESS của run; không dùng FAILED/UNKNOWN/NOT_RUN như prediction thành công.
6. Đọc lịch sử/render CSV không ghi trạng thái. Bấm tải mới chuyển EXPORTED;
   CSV escape công thức spreadsheet trong text, không đổi customer ID trong DB.
7. Queue/Worker dùng SQLite atomic claim và fenced completion. Retry có giới hạn
   với transient error cho handler được đăng ký, permanent error không retry.
   Dashboard LLM tiếp tục max_retries=0; fallback/hedge chỉ nằm trong Router cũ.
   Không tự gọi LLM khi đổi tab, ngân sách, bộ lọc, rerun hay hồi phục job.

## Lỗi phát hiện và sửa

Không phát hiện P0 trong phạm vi kiểm tra này. P1/P2 dưới đây có bằng chứng từ
source thực tế và test regression; không phải khẳng định lỗi đã xảy ra trên dữ liệu thật.

| Mức | Lỗi trước sửa / tác động | Cách sửa và bằng chứng hiện tại |
|---|---|---|
| P1 | Customer 360 bấm phê duyệt chỉ `st.success`, không ghi hay thực thi; thông báo thành công giả | `dashboard/app.py:2048`: disable riêng nút thực thi, giải thích chưa áp dụng; không xây hệ thống thực thi ngoài phạm vi. Test UI xác nhận disabled. |
| P1 | Cùng idempotency key nhưng payload/type/retry policy khác được trả job cũ, có thể gắn kết quả sai request | `adapters/task_queue/queue.py:153`: kiểm tra canonical payload, type và retry policy, cả đường đua INSERT; khác request bị từ chối. |
| P1 | Recovery cập nhật queue và run ở hai transaction; lỗi giữa chừng để hai trạng thái lệch nhau | `adapters/database/dashboard_runs.py:277`: một BEGIN IMMEDIATE fence cả job/run/outcome. Test trigger SQL gây lỗi xác nhận rollback toàn bộ. |
| P1 | Lưu lại cùng chiến dịch tạo UUID mới; tầng Dashboard nhận dataset/run/results rời rạc, chưa ràng buộc nội dung kết quả với run | `entrypoints/streamlit/dashboard_llm_ui.py:399`: xác minh provenance, tính lại kết quả/summary; content intent hash + UUID5, SQLite immutable comparison chống race. Test bấm nhiều lần, 3 luồng, đổi customer/probability/run/pin/summary. |
| P2 | NaN/Inf/bool, tenure âm/phân số và overflow kinh tế có thể lọt vào public engine | `modules/retention/use_cases.py`, `policy.py`, `budget.py`, `modules/campaigns/models.py`: validate số hữu hạn/phạm vi, từ chối invalid budget trước tính toán/lưu. Đã tái hiện NaN action cost và benefit vô hạn trước sửa. |
| P2 | Budget UI tối thiểu1000; so ngân sách có epsilon1e-9 cho phép can thiệp chi phí rất nhỏ ở B=0 | UI cho0; greedy/brute-force cộng và so Decimal theo giá trị chi phí biểu diễn, bỏ epsilon. Test tiny cost5e-10 với B=0 không được chọn; không đổi ưu tiên greedy. |
| P2 | ROI chi phí0 báo0%, và báo cáo gọi lợi ích theo margin là doanh thu | ROI N/A khi không có chi phí; báo cáo gọi đúng lợi ích giữ chân kỳ vọng. Test UI can thiệp miễn phí, ngân sách0. |
| P2 | Ghi lại status có thể làm EXPORTED lùi SIMULATED; retry thành công vẫn giữ error cũ; nhánh worker error trả object stale khi mất fence | Repository chỉ cho tiến trạng thái; SUCCESS xóa error; mọi nhánh worker kiểm tra kết quả complete_claim. Tests retry/transient/permanent, exhausted và stale completion. |
| P2 | Campaign CSV trực tiếp từ use case chưa escape formula; lỗi lưu DB lên UI chưa được xử lý an toàn | Dùng chung `safe_export_frame`; lỗi lưu hiển thị safe error và giữ snapshot cũ. Test text `=`, whitespace-prefix, CR/LF/tab và SQL trigger. |
| P2 | Recovery chỉ nhìn5 job gần nhất, và xác nhận worker đã dừng có thể còn khi đổi job | Liệt kê mọi RUNNING job, chọn từng job, reset consent theo job/token. Test job cũ ngoài danh sách recent vẫn được tìm thấy. |

Snapshot precision/provenance và fail-closed của nhóm2 được giữ. Không refit,
không sửa preprocessing/artifact/retrieval K=5/prompt/Router/Providers/golden.

## Toàn vẹn và khôi phục SQLite

- Không ALTER/DROP bảng, không backfill, không sửa database dữ liệu hiện có.
- Ledger/outcome FK giữ nguyên; begin_attempt nay cần customer thuộc cohort và run RUNNING.
- Campaign có PRIMARY KEY, immutable comparison trong BEGIN IMMEDIATE; cùng intent
  có ID tất định, các concurrent inserts không ghi đè snapshot. Status chỉ tiến
  DRAFT→SIMULATED→EXPORTED. Intent thay đổi cần campaign mới; tên trùng được phép
  nếu run/scenario/nội dung khác, không dùng tên làm identity.
- Cross-database run–campaign được kiểm tra ở service Dashboard trước lưu; không
  có distributed transaction hay FK xuyên hai SQLite DB. Run/outcomes là snapshot
  bền vững, UI không xóa chúng. Generic legacy campaign API vẫn dùng được độc lập;
  không được xem API đó như đường bypass kiểm tra provenance của Dashboard LLM.
- Recovery cần operator xác nhận worker đã dừng. CALLING→UNKNOWN, giữ SUCCESS,
  giữ phần chưa chạy NOT_RUN; job/run cùng terminal. Nếu run đã COMPLETE trước khi
  worker chết, sửa trạng thái job từ kết quả bền vững, không dự đoán lại.
- Trigger gây lỗi INSERT campaign: không có row nửa chừng, bỏ trigger rồi retry
  tạo thành công. Trigger gây lỗi UPDATE job khi recovery: job/run/outcome cùng
  rollback. Stale worker không thể ghi đè UNKNOWN hoặc terminal job.
- Chưa thử power-loss/OS filesystem corruption, backup restore thực tế, hoặc DB
  do nhiều process không tuân thủ service này sửa trực tiếp.

## Kết quả kiểm thử

Kết quả cuối cùng (mock/offline, không gọi API thật):

| Kiểm tra | Kết quả |
|---|---|
| Bộ test nghiệp vụ mới `test_group4_business_services.py` | 71 passed |
| Integration/nhóm2/Retention/Budget/Campaigns/standalone trên repo chính | 218 passed |
| Full regression strict private-assets | 1.170 passed, 15 warnings, 403,94s; không failed/skipped |
| Ruff check / format check | Pass, 13 file Python trong phạm vi |
| Mypy follow-imports normal | Pass, 13 source files |
| Golden nhóm2 / RAG v2 | Pass trong full suite strict-assets, fixture không đổi |
| Import isolation | Pass trong bộ targeted: clean minimal Dashboard và standalone, không bundle/v1/classifier/training/SHAP/provider thật |
| Browser smoke | Đủ4 tab; label1 nhưng observed No; budget0→cost0, ROI N/A; lưu và xuất campaign tổng hợp→EXPORTED |

Bộ test mới gồm kiểm tra công thức độc lập, ngưỡng hòa vốn, negative/NaN/Inf,
thứ tự ID đảo, zero/free actions, ngân sách nhỏ, tính tất định, đối chiếu brute
force, concurrency, persistence/restart, idempotency, rollback, orphan/provenance,
CSV, retry/fencing/recovery và UI edge cases. Full regression chạy trên bản sao
có mã/config tương đương repo chính vì legacy tests có ghi MLflow/SQLite; targeted
chạy trực tiếp main với temp state/cache ngoài repo. Socket outbound bị chặn;
credentials loại khỏi môi trường test. Artifact thật chỉ được đọc cho golden;
không đưa customer IDs thật hoặc API keys vào tài liệu/fixture mới.

## Chạy demo và kiểm tra thủ công đã thực hiện

Preview đang chạy từ repository chính tại <http://127.0.0.1:51415/>.
Database riêng ở `/private/tmp/churn-group4-_6rkpegq/preview-state/`.
4 khách hàng tổng hợp, mock provider, outbound network bị chặn, live bị tắt.
Browser đã kiểm tra snapshot COMPLETE; churn observed25%; DEMO-001 prediction1,
p0.76 nhưng observed No; B=0, cost0, ROI N/A; lưu campaign rồi tải CSV,
xác minh EXPORTED bằng cả UI và SQLite. Không dùng dữ liệu này làm kết quả live.

Lệnh entrypoint hiện hành: `PYTHONPATH=src python3 -m streamlit run src/churn_retention/dashboard/app.py`.
Lệnh này dùng cấu hình server hiện có; preview offline của nhiệm vụ được launch
qua wrapper tạm, tắt keys/live và trỏ state ra ngoài repository.

## File thay đổi trong nhiệm vụ này

Sửa12 file có sẵn, giữ các phần không thuộc nhóm4:

- `src/churn_retention/modules/retention/policy.py`
- `src/churn_retention/modules/retention/use_cases.py`
- `src/churn_retention/modules/retention/budget.py`
- `src/churn_retention/modules/campaigns/models.py`
- `src/churn_retention/modules/campaigns/use_cases.py`
- `src/churn_retention/adapters/database/campaigns/repository.py`
- `src/churn_retention/adapters/database/dashboard_runs.py`
- `src/churn_retention/adapters/task_queue/queue.py`
- `src/churn_retention/contracts/jobs/models.py`
- `src/churn_retention/entrypoints/worker/worker.py` (generic dispatch/fencing; không sửa legacy scoring handler/store)
- `src/churn_retention/entrypoints/streamlit/dashboard_llm_ui.py`
- `src/churn_retention/dashboard/app.py`

Tạo2 file: `tests/test_group4_business_services.py` và tài liệu này.
Các file nhóm2/7.7 có trước được giữ, chỉ nối thêm xử lý nghiệp vụ trong các file
liệt kê. File P1 `docs/llm_predictor.md` không đọc nội dung modified ra báo cáo,
không sửa và không copy modified content vào test snapshot.

## Giới hạn còn lại và kết luận

- Can thiệp thực tế, gửi ưu đãi/campaign ra hệ thống ngoài, và đo outcome/ROI
  thực nghiệm chưa có; nút phê duyệt thực thi được khóa rõ ràng. Campaign ở đây
  là snapshot mô phỏng và danh sách CSV, không đồng nghĩa khách đã nhận ưu đãi.
- Campaign simulation hiện chạy đồng bộ trong UI/use case. Enum
  CAMPAIGN_SIMULATION và cơ chế đăng ký handler có sẵn; test dùng handler tổng
  hợp để kiểm tra retry. Không tự thêm daemon/async campaign orchestration.
- Không thêm worker heartbeat/lease, auth/tenant, deploy, backup pipeline hay
  multi-node HA. Không khẳng định exactly-once API billing khi process chết;
  UNKNOWN không được tự replay. Retry không có backoff scheduler trong nhóm4.
- Greedy vẫn là xấp xỉ; xác suất LLM và các hệ số hiệu quả chưa hiệu chuẩn.
- Layout4 KPI trên viewport rất hẹp còn chật; không mở rộng sang nhóm giao diện
  responsive ngoài phạm vi. Các control/luồng vẫn được kiểm tra bằng AppTest
  và browser; không tuyên bố đã hoàn thành toàn bộ kiểm thử thiết bị.
- CSV dành cho spreadsheet có thể thêm apostrophe bảo vệ ở text bắt đầu bằng công thức; DB giữ ID gốc. Consumer máy cần hiểu quy ước export này.
- 15 warnings đến từ bundle metadata/dirty tree và SQLAlchemy deprecation có sẵn.
- Kiểm tra live API của nhóm1 không được lặp lại trong nhiệm vụ này.

Nhóm4 đạt các tiêu chí business services của ứng dụng hỗ trợ quyết định trong
phạm vi trên; không coi trạng thái đó là đủ điều kiện thực thi chiến dịch thật
hoặc triển khai production.

## Git safety

Branch và HEAD trước/sau giữ nguyên như đầu báo cáo; refs và index không đổi,
index không có staged files. Đối chiếu SHA-256 xác nhận **2.155 file có sẵn ngoài
phạm vi không đổi**, bao gồm dữ liệu/artifacts/bundle/golden, API keys và file P1.
167 executable/config files của snapshot regression khớp mã repository chính.
Legacy scoring handler và IdempotentPredictionStore trong worker không đổi.
Quét pattern secrets trong12 file sửa và2 file mới không có hit.
Working tree tiếp tục dirty với thay đổi có sẵn và các file nhóm4; không dọn,
ghi đè hoặc stage các thay đổi đó. Các temp DB/log/cache nằm ngoài repository.

Không commit, stage, push, merge, reset, rebase hay đổi nhánh. Bản ghi hash đầu/cuối,
logs và bằng chứng browser nằm trong `/private/tmp/churn-group4-_6rkpegq/`.
