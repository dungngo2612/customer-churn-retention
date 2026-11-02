# Nhóm 5 — Bảo mật demo một quản trị viên (phương án A)

Ngày: 2026-11-02. Branch `feat/ml-batch-scoring`, HEAD
`7341854d2aedf94103916a3a59b696482c47eb30`.

## Phạm vi và kết luận

Kết luận kiểm chứng cuối cùng: **ĐANG XÁC MINH REGRESSION CUỐI**.

Dashboard Cường là ứng dụng trung tâm; LLM/RAG v2 là dịch vụ dự đoán hỗ trợ.
Đây là demo cục bộ cho **một quản trị viên tin cậy trên máy được kiểm soát**.
Run ledger, queue và campaign archive dùng chung giữa các cửa sổ/phiên trình duyệt
của quản trị viên đó. Không có tài khoản người dùng, authentication, RBAC hoặc tenant.
Session state, UUID, checkbox, CORS/XSRF và loopback không được coi là identity.

P1 về hai người dùng xem chung campaign trong báo cáo nhóm 5 trước vẫn đúng nếu
mở ứng dụng cho nhiều người. Quyết định phương án A thay đổi phạm vi được hỗ trợ;
không phải đã sửa tenant isolation. Không được tái sử dụng kết luận này để deploy
Internet, LAN, multi-user hoặc bật public reverse proxy.

## 1. Audit các đường truy cập

| Đường khởi động/truy cập | Kết quả audit và kiểm soát |
|---|---|
| `src/churn_retention/dashboard/app.py` | Entrypoint Dashboard chính, bốn tab. Trước đây dùng `streamlit run` không chỉ rõ host; mặc định framework có thể bind wildcard. Nay guard chạy trước import dịch vụ và trước đọc ledger/archive. Runtime không được launcher kích hoạt sẽ chỉ báo lỗi an toàn, không mở prediction controls hoặc lịch sử. |
| `python -m churn_retention.dashboard.demo` | Launcher được hỗ trợ. Kiểm tra cấu hình trước tạo state/mở socket; buộc địa chỉ IPv4 `127.0.0.1`, port rõ ràng, không tự dò port khác. |
| `.streamlit/config.toml` | Thêm default loopback, CORS/XSRF, upload2MB, static user-file serving off, telemetry off, exception UI chi tiết off. Đây là default theo cwd, không phải lớp bảo vệ duy nhất. Launcher vẫn enforce khi chạy từ cwd khác. |
| CLI/env/config override | Không có tùy chọn host của launcher. `0.0.0.0`, `::`, LAN address, hostname khác, base URL proxy và SSL config không phù hợp bị từ chối. Đọc rõ `STREAMLIT_SERVER_ADDRESS`/`STREAMLIT_BROWSER_SERVER_ADDRESS` vì programmatic bootstrap không xử lý các biến này giống CLI. Các flag bảo vệ được overlay cố định trước start. |
| Lệnh `streamlit run ...dashboard/app.py` trực tiếp | Không còn là lệnh demo hợp lệ, kể cả khi cố mở listener bằng flag ngoài launcher: guard không phục vụ dữ liệu. `.streamlit` chỉ là default; không tuyên bố có thể ngăn người có quyền chạy Python tự mở một web server khác. |
| `apps/streamlit_llm/app.py`, `apps/streamlit/scoring_page.py`, trang Explain, `main.py` | Entrypoint phát triển/legacy riêng, không được launcher mới đăng ký hay import. Không khởi động song song các entrypoint này trong demo được chứng nhận. Không sửa implementation legacy. |
| HTTP `/`, static framework, `/_stcore/health`, upload và `/media/...` download/export | Toàn bộ ASGI app được bọc, không chỉ UI: peer phải là IPv4 loopback, Host phải đúng `127.0.0.1:<port>`, Origin nếu có phải đúng HTTP origin đó. Duplicate Host, Origin khác hoặc Forwarded/X-Forwarded-* bị chặn. Bảo vệ cũng áp dụng các route framework khác nếu có. |
| WebSocket `/_stcore/stream` | Cùng peer/Host checks, bắt buộc đúng Origin; thiếu/sai Origin bị chặn trước Streamlit. CORS/XSRF framework giữ bật. |
| Reverse proxy/tunnel | Không có cấu hình proxy/tunnel/deployment của Dashboard được tìm thấy trong các entrypoint, scripts, docker/config hiện hành. Kiểm tra tiến trình không thấy ngrok/cloudflared/nginx/caddy/httpd hoặc SSH forwarding; ssh-agent không phải tunnel. Không tạo public URL hay proxy. |

Môi trường kiểm chứng: macOS/Python3.14, Streamlit1.66.0 (Starlette/Uvicorn).
Adapter kiểm tra backend/factory thực được server gọi, không chỉ sửa re-export.
Nếu API server nội bộ Streamlit không còn tương thích, launcher từ chối khởi động;
không fallback sang lệnh Streamlit trực tiếp. Backend Tornado cũ chưa được hỗ trợ.
Không thay dependency trong pyproject.

Không có home `.streamlit/config.toml`, home/repo `secrets.toml` trong lần audit này.
Không đọc hoặc in nội dung `.env`/API keys. Không có Dashboard listener cũ đang mở
trước nhiệm vụ; các listener hệ thống khác không được coi là Dashboard và không bị sửa.

## 2. Các rủi ro đã được kiểm soát bằng kỹ thuật

- **Public binding vô tình:** default loopback + launcher từ chối cấu hình public
  trước side effect; flags cố định; guard từ chối đường chạy bypass. Kiểm tra bằng
  subprocess CLI thật, không chỉ monkeypatch config.
- **DNS rebinding/cross-site WebSocket/proxy headers:** middleware bao toàn bộ app,
  exact Host/Origin/peer checks, reject forwarding headers, giữ CORS/XSRF.
- **Live API tự bật:** launcher luôn ghi đè inherited ALLOW_LIVE thành0 và PREVIEW=1
  theo mặc định, bỏ provider keys khỏi process mock. `--enable-live` cần nhập chính
  xác `ENABLE LIVE API` tại terminal. Sau đó UI vẫn mặc định chọn mock và cần chọn
  live + xác nhận snapshot + Submit. Environment live flag một mình không đủ.
  Không gọi live API trong nhiệm vụ này.
- **Trộn dữ liệu production:** không nhận CHURN_DASHBOARD_STATE từ environment.
  Mỗi lần mở mặc định có thư mục demo tạm mới0700; `--state-dir` chỉ nhận thư mục
  rỗng mới hoặc thư mục riêng0700 có marker demo đúng. Không tự mở/import/migrate
  DB cũ. Ledger/queue ở state.sqlite3, campaign archive ở sibling demo riêng.
  Existing SQLite guards từ chối symlink/hardlink/owner sai; file0600, FK bật.
- **Chi phí/rerun:** giữ max5 tại UI, adapter và persisted worker boundary. Không
  thay Router/retry/hedging. Tab/budget/campaign/export không gọi prediction lại.
- **Data export/secret/error:** giữ toàn bộ Group5 credential guards, strict CSV,
  safe CSV cho mọi text column, HTML escaping, static error boundaries. Không
  đưa exception/config paths lên UI; upload name chỉ là basename metadata.
  Không dùng raw customer filenames để mở file. Không expose thư mục state qua
  static serving. Không đưa key vào prompt, ledger, campaign hoặc export.

Không thay preprocessing, RAG v2, retrieval K=5, Prompt Builder, Router/Providers,
Retention/Budget, reference artifact, dataset, model bundle hay golden fixtures.

## 3. Shared archive và mọi đường dữ liệu

`runtime_path()` trỏ đến demo ledger đã được launcher chọn; `_repo()` dùng campaign
SQLite sibling trong cùng private directory. Mọi phiên của một operator dùng chung
repository, không có phép lọc theo session như giả authentication.

Đã kiểm tra lại reserve/process/get/progress/frames của run store, enqueue/claim/
complete/fail/recovery của queue, create/get/list/export/mark_exported của campaign,
các dataframe download, filtered CSV và lịch sử campaign trên Dashboard. Không thêm
đường đọc DB production, không sửa transaction/recovery/economic algorithms.

Idempotency, lease/fencing, partial failure, rollback, immutable snapshots và join
run/customer/campaign vẫn do các service và SQLite constraints kiểm soát. Không
biến `churn_prediction` thành `Churn` đã quan sát. Campaign chỉ mô phỏng, không gửi
ưu đãi hay thông báo thật.

## 4. Bằng chứng mạng và browser

Preview từ **source repository chính**: <http://127.0.0.1:54407/>.
Không phải deployment. Có4 khách hàng giả, provider mock và một artifact v2 tổng
hợp8 reference do test helper hiện có tạo trong thư mục tạm. Đây không phải tập
reference sản xuất4930 khách; golden riêng kiểm tra artifact baseline thật offline.

- `lsof -nP -iTCP:54407 -sTCP:LISTEN`: chỉ `TCP 127.0.0.1:54407 (LISTEN)`, IPv4;
  không `*:54407`, `0.0.0.0` hoặc IPv6 wildcard.
- Kết nối tới cùng port bằng địa chỉ LAN của máy: refused.
- HTTP root và health200; 16 request Host/Origin/forwarding sai403.
- WebSocket không Origin và Origin ngoài localhost403; đúng local Origin101.
- Kiểm tra bằng `tests/single_admin_network_probe.py`; probe không Submit/callLLM.
- Browser smoke đủ4tab: Executive Summary, churn/reason, Customer360, Budget/
  Campaign. Chỉ hiển thị MOCK; banner nêu archive dùng chung, không có identity.
- Đổi ngân sách0 → lưu campaign tổng hợp → export → trạng thái EXPORTED.
- Trước/sau thao tác và restart launcher:1run,1job,4outcomes,1campaign; không có
  generation mới do tab/rerun/budget. Cả2DB integrity_check=ok, FK violations0,
  file0600. Không có fake success hoặc live API.

## 5. Kiểm thử

ĐANG XÁC MINH: cập nhật kết quả cuối sau full regression.

Lần full đầu:1300passed/1failed. Lỗi môi trường bản sao thiếu.git khiến test
`test_internal_vectors_and_ids_match_without_exposing_them` không chạy được
`git check-ignore`; không phải mismatch vector/neighbor. Đã bổ sung bản sao metadata
Git vào môi trường tạm, giữ nguyên assertions/private fixture và chạy lại toàn bộ.
Một lần khởi tạo lại nhầm cwd đã bị dừng ở collect:0test chạy; kiểm tra SHA xác
nhận tất cả protected data/source ngoài phạm vi không thay đổi.

Các business AppTest dùng mocked Runtime không có listener, nên fixture/probe
mock transport guard riêng; không tắt guard production để né test. Có test AppTest
mới xác nhận raw entrypoint không hiện file uploader/button/tab khi thiếu launcher.
HTTP/WebSocket/bind được kiểm chứng riêng trên server thật bằng socket và browser.
Full suite chạy trong bản sao ngoài repository, network connect bị chặn, provider
keys bị gỡ; MLflow/training test outputs chỉ ghi trong bản sao/tạm.

## 6. Rủi ro còn lại và điều kiện vận hành

**Không có authentication:** bất kỳ process hoặc người nào dùng được loopback trên
máy này đều có thể là cùng quản trị viên đối với server. Một OS user khác có thể
truy cập loopback dù không đọc được file0600. Không có session isolation, logout,
per-user quotas hoặc object authorization. Không dùng máy shared/untrusted, remote
browser service, port forwarding, Docker published port, SSH tunnel hoặc proxy.

Middleware chặn header forwarding/Origin thông thường; không chứng minh chống một
process cục bộ cố ý relay rồi viết lại cả Host và Origin. Không thể xác thực nguồn
người dùng từ loopback. Máy/OS account phải được quản trị viên kiểm soát, khóa màn
hình, không chạy tunnel/relay hoặc phần mềm không tin cậy. Không sửa host firewall
hay các dịch vụ khác trong nhiệm vụ này. Nếu không đáp ứng boundary đó: **không
vận hành demo; kết luận này không áp dụng**.

Dùng dữ liệu tổng hợp cho demo. Không tải dữ liệu khách thật, training/test records
hoặc private goldens lên UI/export. Các kiểm tra credential không phải DLP tổng
quát và không chứng minh semantic prompt-injection immunity. Mặc định mock không
gửi dữ liệu ra provider; việc bật live là quyết định chủ động có thể gửi19feature
và5reference theo thiết kế RAG, phát sinh chi phí. Chưa kiểm chứng live ở bước này.

SQLite chưa mã hóa at rest và không bảo vệ khỏi chính OS owner/admin. Không tự xóa
corrupt DB hoặc replay UNKNOWN jobs; dừng và điều tra. Backup/restore/power-loss,
Windows ACL và backend Streamlit khác chưa được chứng nhận. Kết thúc demo phải
đóng server; không để tiếp tục chạy trên môi trường chuyển sang multi-user/public.

Lệnh được hỗ trợ (artifact v2 và external hash pin phải hợp lệ theo cơ chế có sẵn):

```bash
PYTHONPATH=src python3 -m churn_retention.dashboard.demo
```

Chỉ mở `http://127.0.0.1:8501/`; không dùng hostname khác. Có thể dùng `--port`
để chọn port local khác. `--state-dir` là thư mục demo riêng tuyệt đối, không phải
đường dẫn DB production; không dùng thư mục có dữ liệu khác hoặc tự tạo marker cho
DB cũ. Nếu cấu hình không an toàn, port bận, adapter không tương thích hoặc không
xác minh được state thì dừng, không đổi sang wildcard/direct entrypoint.

## 7. File thay đổi và Git safety

Tạo6file:

- `.streamlit/config.toml`
- `src/churn_retention/dashboard/demo.py`
- `src/churn_retention/entrypoints/streamlit/demo_security.py`
- `tests/test_single_admin_security.py`
- `tests/single_admin_network_probe.py`
- `docs/group5_single_admin_security.md`

Sửa bổ sung6file hiện có, giữ các thay đổi trước nhiệm vụ:

- `src/churn_retention/dashboard/app.py`: guard trước truy cập dữ liệu, lệnh chạy.
- `src/churn_retention/entrypoints/streamlit/dashboard_llm_ui.py`: banner shared
  single-admin archive; không sửa business calculation.
- `tests/test_dashboard_llm_integration.py`: test fixture mock transport-only guard
  của AppTest; thêm module bảo vệ vào minimal deployment inventory.
- `tests/dashboard_llm_probe.py`: AppTest isolation process mock transport-only guard.
- `README.md`: lệnh demo an toàn và phạm vi.
- `docs/cuong_dashboard_llm.md`: thay lệnh Dashboard cũ, giải thích state demo.

ĐANG XÁC MINH: cập nhật Git/hash cuối. Không stage/commit/push/merge/reset/rebase;
không thay branch hoặc HEAD. Không sửa docs/llm_predictor.md. Dữ liệu demo/test
được tạo bên ngoài repository; không thêm private artifacts/golden/secrets vào Git.
