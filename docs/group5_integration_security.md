# Nhóm 5 — Integration và Application Security

Ngày: 2026-11-02. Repository chính: `customer-churn-retention`.
Branch/HEAD: `feat/ml-batch-scoring` / `7341854d2aedf94103916a3a59b696482c47eb30`.

## Kết luận

**CHƯA ĐẠT tiêu chí bảo mật toàn hệ thống cho nhiều người dùng.**
Phần tích hợp/offline và các sửa lỗi cục bộ đã triển khai, có kết quả test bên dưới.
Còn P1 về authentication, object authorization/tenant isolation và quyền dùng live
API theo người dùng. Người dùng đã xác nhận yêu cầu nhiều người dùng và chỉ yêu
cầu báo cáo phần auth/tenant để quyết định riêng; không tự xây kiến trúc đó.

Không có P0 được xác nhận trong phạm vi này. Test pass không phải bằng chứng đã
đạt multi-user security. Không deploy, không gọi API thật, không thực thi ưu đãi;
không triển khai nhóm 3, 6 hoặc 7.

## Phạm vi end-to-end

Đã kiểm tra luồng thực:

CSV bytes → strict CSV/schema validation → 19 cleaned features → RAG v2 K=5 →
LLMChurnPredictor/Prompt Builder/Router (provider mock) → Queue/Worker → SQLite
ledger → Dashboard4 tab → Retention/Budget → immutable Campaign → CSV export.

- Đơn1 và batch5, primary thành công, primary lỗi→fallback, cả hai lỗi và PARTIAL.
  Không báo success giả; các row chưa chạy vẫn NOT_RUN. PARTIAL chỉ dùng subset
  SUCCESS để tính kinh tế/campaign; FAILED không tạo probability giả.
- Natural prediction identity hiện là PRIMARY KEY `(run_id, customer_id)`;
  không có một prediction UUID riêng. Run ID, request key và campaign UUID riêng
  biệt; test đối chiếu ID/row order qua CSV, ledger, prediction và campaign.
- Request idempotency/concurrency: hai submit trùng chỉ có một job/một generation;
  persisted cap1–5 vẫn được kiểm tra tại worker. Retry LLM ngoài Router bằng0.
- Recovery/fencing/rollback của nhóm4 được kiểm thử lại; UNKNOWN không tự replay,
  stale worker không ghi đè terminal state. PRAGMA integrity_check=ok và không
  có foreign_key_check violations trên state tổng hợp sau workflow/concurrency.
- Tab/budget/rerun/campaign/export không gọi predictor. Churn đã quan sát giữ
  riêng, không biến `churn_prediction=1` thành observed Churn=Yes.
- Kết quả mock luôn ghi mode mock; provider route không được hiểu là API thật.

## Lỗ hổng/điểm yếu đã sửa

| Mức | Bằng chứng trước sửa | Thay đổi và kiểm chứng |
|---|---|---|
| P1 | Một response hợp lệ về schema nhưng `reason` chứa key giả đang cấu hình đã được lưu vào SQLite; tái hiện offline, không phải key/API thật | Thêm `contracts/application_security.py:19`, reject configured credentials và các định dạng token dễ nhận biết tại service/input/output/persistence/read/export. Không sửa text rồi coi là response thành công; fail closed. Test kiểm tra error/log/UI/DB/CSV không chứa key giả. |
| P1, tùy quyền thư mục cha | File SQLite mới có mode0644 dưới umask022; OS user khác có thể đọc nếu truy cập được thư mục cha | `adapters/database/local_sqlite.py:13`: file0600, kiểm tra regular file/owner/link count, từ chối symlink/hardlink/parent cho người khác ghi; connection timeout và FK bật. Tighten quyền file cũ do cùng owner mở; không rewrite records. Không thay quyền database gốc trong nhiệm vụ này vì tests/demo dùng DB tạm. |
| P2 | CSV thừa một trường được pandas tự dùng làm index, âm thầm bỏ trường thay vì từ chối; đã tái hiện | `entrypoints/streamlit/dashboard_llm_service.py:94`: csv.reader strict, kiểm tra số field từng row, NUL, lỗi quote/field-size, row cap trước pandas; giữ2MB/1000rows và batch5. |
| P2 | Group4 đã escape campaign CSV, nhưng filtered recommendations và built-in dataframe download còn mang text chưa escape, gồm reason | Dùng `safe_export_frame` cho mọi text column và tất cả các đường export liên quan. Bản sao an toàn không đổi ID/probability trong ledger, selector hay phép join. Test formula ở ID/action_name/reason/provider, whitespace/CR/LF/tab. |
| P2 | Corrupt state/archive hoặc lỗi callback có thể thoát ra exception UI; chưa có safe boundary đồng nhất | `dashboard_llm_ui.py:143`, `dashboard/app.py:770` và callback mark_exported: static safe error, giữ DB, không xóa/tạo lại dữ liệu tự động. Test corrupt bytes và corrupt JSON, giữ nguyên nội dung gốc. |
| P2 / hardening | Filename chỉ là metadata nhưng chưa normalize; control characters trong ID chưa chặn rõ | Basename/length/control validation; không mở filename/path do khách gửi. SQL-looking ID vẫn chỉ là parameter, không trở thành SQL/path/instruction. Positive-control test .env access bị chặn trước open. |
| P2 / hardening | Text động từ action/campaign có thể vào HTML/Markdown formatting | Escape action labels/HTML, dùng thông báo/download label cố định thay vì nội suy tên campaign. Customer ID được escape và reason dùng st.text. Không khẳng định đã có browser XSS exploit; tests xác nhận payload không đi vào executable HTML. |

Các credential guards chỉ đọc environment được cấu hình, không đọc `.env`.
Token matching có boundary để không chặn nhầm ID thông thường như risk-customer-…;
không serialize NaN/NumPy numeric để quét secrets, nên không làm đổi preprocessing.
Các thông báo từ chối không kèm giá trị bị từ chối, prompt, key hay raw SDK error.

## P1 còn mở — cần quyết định kiến trúc riêng

### 1. Shared archive không có owner/tenant authorization

Bằng chứng:

- `dashboard_llm_ui.py:76`: runtime_path là environment/config toàn process.
- `dashboard_llm_ui.py:406`: mọi session dùng cùng campaign repository.
- `dashboard_llm_ui.py:496`/`:500`: get/list campaign không kiểm tra principal.
- Campaign/run/job schema chưa có owner/tenant identity để enforce quyền.
- Probe tổng hợp có hai run riêng cho A và B, AppTest session B mới nhìn thấy
  campaign A trong dropdown. `multiuser-evidence.json` ghi true; không dùng thông
  tin khách hàng thật. Đây là bằng chứng thiếu isolation, không phải test pass
  chứng nhận authorization.

File0600 chỉ bảo vệ theo OS account; mọi browser user của cùng Streamlit process
vẫn dùng chung server account. UUID, session_state, CORS và XSRF không thay thế auth.

### 2. Live API flag không phải quyền của người dùng

`dashboard_llm_ui.py:225` dùng CHURN_DASHBOARD_ALLOW_LIVE/CHURN_DASHBOARD_PREVIEW
cấp server; checkbox là consent cho request, không phải identity/role authorization.
Test đã xác nhận server opt-in, preview gate, stale consent reset và revocation
đều có hiệu lực. Nhưng nếu mở server nhiều người dùng và bật live thì các user
không được phân quyền riêng, chưa có quota/cost budget theo user/tenant.

Không tự thêm user table, tenant IDs, auth proxy hoặc di chuyển archive cũ vì cần
quyết định ownership, chính sách quyền và migration dữ liệu. Phần này đã dừng.

Những quyết định cần chốt trong nhiệm vụ riêng:

1. Nguồn identity tin cậy (SSO/OIDC hoặc tài khoản nội bộ) và boundary tenant.
2. Quyền read/export/recover/run-live theo owner/role; kiểm tra ở service/worker,
   không chỉ UI; không tin owner_id do client tự gửi.
3. Bind run/prediction/job/campaign với principal/tenant và chính sách migration
   archive hiện có; chưa gán lại ownership cho dữ liệu cũ trong bước này.
4. Quota/cost/concurrency limits theo identity cho live requests và audit trail.

## Prompt injection và privacy

- Không sửa Prompt Builder/instructions/schema hoặc thuật toán retrieval.
- Category attack trong query bị validator từ chối trước Router. Metadata ID mang
  SQL/instruction vẫn không vào prompt. Query chỉ có19 feature, không có observed
  Churn, customer ID hay provider key.
- References chỉ có feature+Churn; không có neighbor ID, distance, vector hoặc
  held-out ID. Mock capture kiểm tra đúng5 examples và đúng schema cho từng example.
- Artifact tổng hợp chứa string `</system>`, backticks và END_CUSTOMER_DATA_JSON
  trong category: builder encode vào JSON, không tạo raw tag/delimiter mới.
- Response chứa synthetic private neighbor ID bị predictor validator từ chối và
  fallback; private identifier không được persist vào reason. Cơ chế có sẵn được
  kiểm tra, không đổi predictor để ép kết quả.
- Artifact bị append bytes không khớp external pin bị từ chối, không fallback v1.
  Tập reference tổng hợp bằng chính xác split=train, không gồm held-out val/test.
- Golden/nhóm2 kiểm tra artifact baseline4930×45 và8 query/prompt cases; fixtures,
  preprocessing và artifact gốc không đổi. Security attacks/demo chỉ dùng dữ liệu
  tổng hợp. Baseline/private assets chỉ phục vụ kiểm tra offline tương đương hiện
  có; không gửi API hoặc đưa record/ID thật vào report/fixture mới.

Mock chứng minh framing/validation, **không chứng minh semantic immunity của LLM
thật**. Các feature của5 train references vẫn là nội dung gửi provider theo thiết
kế RAG nếu operator gọi live; bỏ IDs không đồng nghĩa đã anonymize hoàn toàn.
API keys không được đưa vào prompt; không có tool thực thi lệnh/đọc file từ LLM.

## Kiểm thử và kết quả

| Kiểm tra | Kết quả cuối |
|---|---|
| Bộ security/e2e mới | 73 passed (trong full suite) |
| Security + Prediction Service (lượt trước test bổ sung cuối) | 95 passed |
| Targeted tích hợp nhóm1/2/4/standalone/security | 258 passed |
| Full regression strict-assets, chạy tuần tự | 1.243 passed, 15 warnings, 572,50s; không failed/skipped |
| Golden / import isolation | Pass, strict-assets; matrix/vectors/top5/tie/self-exclusion và8 prompt hashes không đổi |
| Ruff check / format | Pass, 13 Python files |
| Mypy follow-imports normal | Pass, 13 source files |
| Browser smoke | Pass4 tab, synthetic ref8/customer4, budget0, campaign→EXPORTED |
| Multi-user authorization probe | FAIL theo yêu cầu isolation: B thấy campaign A; P1 còn mở |

Lượt full đầu chạy cùng các kiểm tra khác gặp AppTest timeout30s tại
`test_dashboard_preserves_tabs_persistence_campaigns_and_no_inference_on_rerun`;
đã ngắt lấy traceback (378 passed,1failed). Test đó pass trong targeted258.
Không tăng timeout hoặc sửa assertion để che lỗi; chạy lại toàn bộ tuần tự để
kiểm chứng: lượt cuối1.243 passed, timeout không tái hiện. Không tính lượt bị ngắt là full regression pass; đây vẫn là giới hạn cần lưu ý khi chạy QA dưới tải song song.

Full suite chạy trong bản sao riêng với executable/config files khớp main để các
legacy tests ghi MLflow/SQLite không đụng dữ liệu chính. Targeted tests dùng repo
chính nhưng mọi state/cache/temp nằm ngoài repo. Credentials thật bị loại khỏi
environment test, socket outbound bị chặn; provider/SDK I/O giả. Các test
standalone/minimal deployment được bổ sung đúng hai runtime helper modules;
không bỏ import barrier hoặc bớt assertions.

## Browser và giới hạn vận hành

Preview: <http://127.0.0.1:54407/> — mã repository chính, chỉ bind loopback,
reference v2 tổng hợp8 dòng,4 customer input tổng hợp, mock provider, live tắt,
outbound socket chặn, telemetry tắt, HTTP upload cap2MB. State riêng:
`/private/tmp/churn-group5-ng86zt9m/preview-state/`.

Đã mở đủ4 tab, xác minh prediction1 nhưng observed Churn=No, reason MOCK, budget0
và ROI N/A; lưu rồi tải CSV, xác nhận EXPORTED bằng UI và SQLite; DB mode0600.
Run/job/outcome counts không tăng khi đổi tab/budget/export; restart preview bằng mã cuối và reload browser vẫn giữ1 run,1 job,4 outcomes và campaign EXPORTED, không generation mới.

Cần phân biệt cấu hình preview với lệnh Streamlit bất kỳ: installed defaults
server.address=None (server fallback0.0.0.0), upload200MB, CORS/XSRF=true. App chặn
CSV2MB sau nhận bytes; transport cap cần được cấu hình khi chạy server. Không đổi
cấu hình triển khai toàn repository hoặc public networking trong nhóm5. Không
coi mặc định CLI là an toàn để chia sẻ multi-user. CORS/XSRF giữ nguyên, không tắt
để làm browser test pass.

Các giới hạn khác:

- Không thử API thật/quota/billing/live SLA; không kiểm tra secrets trong debug
  logs của SDK bên ngoài hoặc dump process/crash ở môi trường triển khai thật.
- Credential guard không phải DLP tổng quát, không nhận diện mọi encoding hay
  historical secret không còn cấu hình. Không quét/publish nội dung `.env`.
- SQLite không mã hóa at-rest; không chống host admin/same-UID process; Windows
  ACL chưa được chứng nhận. Không tự sửa DB hỏng, restore backup hay backfill data.
- Crash recovery cần operator xác nhận worker đã dừng; UNKNOWN không auto-repeat;
  không khẳng định exactly-once billing sau crash/power loss.
- Campaign chỉ là mô phỏng và CSV, không gửi ưu đãi. Không sửa nghiệp vụ nhóm4
  hoặc khai triển thực thi nhóm3; không đổi layout nhóm6/deploy nhóm7.

## File tạo/sửa

Tạo4 file:

- `src/churn_retention/contracts/application_security.py`
- `src/churn_retention/adapters/database/local_sqlite.py`
- `tests/test_group5_integration_security.py`
- `docs/group5_integration_security.md`

Sửa10 file có sẵn:

- `src/churn_retention/entrypoints/streamlit/dashboard_llm_service.py`
- `src/churn_retention/entrypoints/streamlit/dashboard_llm_ui.py`
- `src/churn_retention/entrypoints/streamlit/llm_prediction_service.py`
- `src/churn_retention/adapters/database/dashboard_runs.py`
- `src/churn_retention/adapters/database/campaigns/repository.py`
- `src/churn_retention/adapters/task_queue/queue.py`
- `src/churn_retention/modules/campaigns/use_cases.py`
- `src/churn_retention/dashboard/app.py`
- `tests/test_dashboard_llm_integration.py` (minimal deployment file inventory)
- `tests/test_llm_standalone_app.py` (minimal deployment file inventory)

Không sửa preprocessing, RAG artifact, retrieval K=5, prompt, golden fixtures,
Router/Providers, classifier/training/batch/SHAP implementations, pyproject hoặc
file P1 `docs/llm_predictor.md`. Không dùng payload như chỉ dẫn thực thi.

## Git safety

HEAD/branch/refs/index trước và sau không đổi; index không có staged files.
170 executable/config files khớp snapshot regression cuối. Đối chiếu hash xác
nhận2.190 file có sẵn ngoài phạm vi không đổi, bao gồm dữ liệu, artifacts/bundles,
golden, `.env`, `pyproject.toml`, Router/Providers và file P1.

Ngoại lệ quan sát trong lúc làm việc: `.idea/workspace.xml` thay đổi ngoài các
lệnh sửa của nhiệm vụ;87 Git objects mới (30 blob,57 tree) xuất hiện. Không đọc
nội dung các object, không xóa hoặc khôi phục metadata này. HEAD/refs/index vẫn
khớp baseline; không có lệnh tạo commit/stage nào được chạy. Không suy đoán tác
nhân tạo object. Số object là quan sát tại lần đối chiếu, không phải thao tác Git
của nhiệm vụ.

`.env` không được track; chỉ `.env.example` đã có trong tracked files. Không đọc
nội dung `.env` ra output/report. Quét pattern secrets trong các file tạo/sửa
không có hit; không coi đó là kiểm toán secrets toàn lịch sử hoặc file P1 được
bảo vệ. Working tree vẫn dirty với thay đổi có sẵn và14 file trong phạm vi này.

Không stage/commit/push/merge/reset/rebase/đổi nhánh. Logs, hash inventory,
probe multi-user và screenshot nằm ở `/private/tmp/churn-group5-ng86zt9m/`.
