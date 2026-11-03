# Trung tâm điều hành giữ chân khách hàng

Bản cải tạo trên `codex/dashboard-redesign`, từ main đã xác minh tại
`c497806a29bbdf5b55d88a1b0d2849bc30a6cabb`. Chưa push/merge vào main.
Stack Streamlit và toàn bộ service nghiệp vụ, LLM/RAG được giữ nguyên.
Điều hướng dùng `st.navigation`/`st.Page`; file trang nằm trong `dashboard/views`
để không kích hoạt cơ chế tự khám phá `pages/` cũ khi mở trực tiếp URL trang con.
Mỗi trang kiểm tra launcher trước khi đọc dữ liệu.

## Bản đồ chức năng trước / sau

| Chức năng đã khảo sát | Vị trí cũ | Vị trí mới | Service/use case giữ lại | Trạng thái bảo toàn |
|---|---|---|---|---|
| Upload CSV và validation | Sidebar | Khách hàng & dữ liệu → Nhập dữ liệu | read_csv, validate_frame | Không inference; lỗi không thay dữ liệu đã chọn |
| Dữ liệu tổng hợp mẫu và tải mẫu | Sidebar | Khách hàng & dữ liệu → Nhập dữ liệu | synthetic_frame | Chọn chủ động; ghi rõ dữ liệu giả |
| Tìm/lọc/sắp xếp khách hàng | Tab 3 | Khách hàng & dữ liệu → Danh sách | Snapshot đã validation | Phân trang 10/25/50; ID ổn định |
| Hồ sơ 19 thuộc tính | Tab 3 | Danh sách, kết quả phân tích, phương án | Customer ID / saved frames | Không ghép điểm của nguồn dữ liệu khác |
| Dự đoán đơn / lô 1–5 | Sidebar | Phân tích rời bỏ → Chạy phân tích | DashboardLLMAdapter.prepare, execute | Xác nhận snapshot; idempotency; không auto rerun |
| Mock/live và lỗi provider | Sidebar/banner | Phân tích; Cấu hình & trạng thái | Adapter, router/service hiện có | Mặc định mock; live gate/timeout/retry không đổi |
| Điểm, reason, reference metadata | Tab 2/3 | Phân tích → Kết quả & lý do; hồ sơ | Ledger.frames | Chưa hiệu chuẩn; không dựng nội dung hàng xóm chưa lưu |
| Observed churn theo contract/Internet/support/payment | Tab 2 | Kết quả → Quan sát theo nhóm | Snapshot quan sát, groupby | Không dùng nhãn LLM làm observed Churn; ít dữ liệu dùng bảng |
| Xếp hạng, lọc nguy cơ/gói, CSV đã lọc | Tab 3 | Phân tích → Kết quả & lý do | recommend, safe_export_frame | Bộ lọc trước phân trang; CSV xuất toàn bộ phần đã lọc |
| Lịch sử kết quả / chọn run | Sidebar kỹ thuật | Phân tích → Lịch sử | Queue.list_jobs, Ledger.get/frames | Chỉ đọc lại; không gọi LLM |
| Tham số kinh tế và gói A0–A3 | Sidebar | Kế hoạch → Bước 1 | default_actions / ActionConfig | Cùng mặc định, điều kiện, đơn vị USD; lưu thiết lập trong session |
| Tổng quan kinh tế và báo cáo Markdown | Tab 1 | Tổng quan; Kế hoạch → Bước 2 | recommend, allocation totals | Mô phỏng; ROI chi phí 0 là không xác định |
| Ma trận phương án từng khách | Tab 3 | Hồ sơ → So sánh phương án hợp lệ | evaluate_actions | Không có nút phê duyệt giả; chưa có backend thực thi |
| Phân bổ ngân sách | Tab 4 | Kế hoạch → Bước 2 | select_actions_under_budget | Heuristic không bảo đảm tối ưu; không đổi thuật toán |
| Khảo sát ngân sách, độ nhạy và số khách theo gói | Tab 4 | Bước 2 → Khảo sát ngân sách & độ nhạy | select_actions_under_budget, sensitivity_by_margin | Cùng dữ liệu và giả định; không inference |
| Tạo và lưu campaign | Tab 4 | Kế hoạch → Bước 3 | create_campaign wrapper hiện có | Provenance/summary được service xác minh; idempotent |
| Mở lại, trạng thái, nguồn và CSV campaign | Tab 4 | Chiến dịch → Chiến dịch đã lưu | list/get/export_campaign_csv, mark_exported | Snapshot bất biến; status export chỉ đổi khi bấm tải |
| Queue và tiến độ dòng | Sidebar | Chiến dịch → Công việc xử lý | SqliteTaskQueue, ledger.progress | Hàng đợi kỹ thuật, không phân công nhân viên |
| Recovery công việc bị ngắt | Sidebar | Công việc xử lý → Khôi phục | recover_interrupted_job | Xác nhận worker đã dừng; fencing; không tự gọi lại |

Các biểu đồ kinh tế trùng thông tin được thay bằng summary/bảng đủ chiều rộng.
Không bỏ use case hoặc tạo thêm nguồn lưu trữ. Session chỉ giữ bản nháp/thiết lập;
ledger, queue, campaigns vẫn là nguồn kết quả đã lưu hiện có.

## Điều hướng và cách dùng

1. **Khách hàng & dữ liệu:** upload → xem trước sau validation → Sử dụng dữ liệu này;
   hoặc chọn Dùng dữ liệu mẫu. Không tự chấm toàn tệp.
2. **Phân tích rời bỏ:** chọn một khách/lô tối đa 5, kiểm tra chế độ, xác nhận và chạy.
   Kết quả & lý do mở snapshot đã lưu. Lịch sử cho phép mở lại mà không inference.
3. **Kế hoạch giữ chân:** thiết lập và áp dụng giả định → xem phương án → xác nhận lưu.
4. **Chiến dịch:** mở bản đã lưu, chọn phạm vi rồi tải CSV. Không có gửi email/SMS,
   thực thi ưu đãi hay phân công nhân viên.
5. **Tổng quan:** tóm tắt lần phân tích và phương án đang chọn, có lối tắt tới bước tiếp.

Thiết lập kinh tế đã áp dụng, ID khách đã mở và snapshot phân tích được giữ khi chuyển
trang. Bản nháp form chưa áp dụng chưa thay thiết lập. Đổi dữ liệu không xóa kết quả;
kế hoạch luôn tham chiếu snapshot đã lưu và hiển thị cảnh báo nếu nguồn mới khác.

Chạy bản local từ branch này:

```bash
PYTHONPATH=src python3 -m churn_retention.dashboard.demo --port 55512
```

Launcher giữ loopback, mock mặc định và database demo riêng. Không nhập dữ liệu thật
để thử giao diện; dùng tệp tổng hợp. Font hệ thống hỗ trợ tiếng Việt, không tải font
hoặc tài nguyên ngoài. Main, dữ liệu/artifact/database có sẵn, stash và worktree không đổi.

## Kiểm chứng ngày 2026-11-03

| Kiểm tra thực tế trên bản cuối | Kết quả |
|---|---|
| Full regression, strict config/markers, PostgreSQL tạm và tài nguyên golden local | **1.205 passed, 0 failed, 0 skipped**, 312,73 giây |
| Ruff lint / format toàn repository | Pass; 182 file đúng định dạng |
| Mypy `src` | Pass; 96 file nguồn |
| Alembic trên PostgreSQL tạm riêng | 001 → 002 → head thành công |
| Build sdist / wheel, cài wheel vào thư mục tạm | Pass; đủ năm file `views`, không có entrypoint classifier cũ |
| AppTest từ wheel đã cài | Năm trang mở được, đọc mock run COMPLETE; chặn mạng ngoài và import classifier/provider SDK |
| Git diff whitespace | Pass |

Hai cảnh báo regression là deprecation từ Starlette/httpx và tên HTTP 422; không có
lỗi kiểm thử. Đây là kết quả local CPython 3.14, không phải tuyên bố CI remote đã chạy.
Runner bỏ API keys thật, chặn socket ngoài loopback; PostgreSQL dùng database tạm
trên port 55511. Không gọi API trả phí. Log và bằng chứng chi tiết ở
`/private/tmp/churn-dashboard-redesign` (`regression-release.log/xml`,
`build-release.log`, `installed-smoke.log`, `preservation-release.json`).

Tests cũ được điều hướng sang trang mới và giữ kiểm tra nghiệp vụ/bảo mật. Bộ acceptance
mới kiểm tra mọi ID qua phân trang 10/25/50, đổi bộ lọc, giữ ID/thiết lập/snapshot, batch
đúng thứ tự và idempotency, mở đúng lịch sử/campaign. Guard được kiểm tra cả entrypoint
chính và năm file trang con. Kiểm thử bytes upload hợp lệ/không hợp lệ, partial/failure,
all-A0, ngân sách 0, chi phí 0, campaign lỗi, HTML/CSV injection vẫn được giữ.

### Browser nghiệm thu

Preview đang chạy tại **http://127.0.0.1:55510/** bằng ứng dụng thật trên branch này.
Fixture ngoài repository chỉ cung cấp 64 hồ sơ tổng hợp, ngân sách 20 USD và mở lại
mock run đã lưu; không thay dữ liệu hoặc mặc định của sản phẩm. Launcher thông thường
ở lệnh trên bắt đầu trống, có nút chọn mẫu bốn khách.

Đã thực hiện trong Chrome:

- Đi qua cả năm khu vực, mở hồ sơ, xem lý do, Lịch sử và Công việc xử lý.
- Chuyển danh sách tới trang 4 (KH-0031…KH-0040), tìm KH-0001 và trở về trang 1/1.
- Xác nhận/chạy batch mock năm khách; ledger ghi một run COMPLETE, một job SUCCESS,
  năm outcome thành công. Điều hướng và kiểm tra kích thước không tạo thêm run/job.
- Áp dụng ngân sách 20 USD, xem phương án dùng 18 USD cho hai khách; lợi ích ròng
  71,38 USD, ROI mô phỏng 396,5%. Lưu và mở lại “Giữ chân khách hàng tháng 11”.
- Bấm tải CSV và đọc tệp tải xuống: đúng sáu cột tương thích, hai khách KH-0001 và
  KH-0005, đều A2. Chỉ campaign demo đổi sang EXPORTED; không thực thi can thiệp.
- Kiểm tra 1440×1000, 1280×900, 1024×900 và 390×844. Đo đủ 20 tổ hợp trang/kích thước:
  document rộng bằng viewport; bảng giữ vùng cuộn riêng. Mobile xếp form thành hàng,
  menu thu gọn, summary hai cột. Đã xem trực tiếp ảnh để chỉnh layout và đơn vị.
- Màu chữ caption được trình duyệt áp dụng là `#526079`, opacity 1. Tương phản chữ
  chính/trắng 15,53:1; chữ phụ/nền xám 5,88:1; trắng/indigo 7,73:1;
  cảnh báo/nền vàng 5,95:1. Có focus outline và dùng widget bàn phím gốc.

**Giới hạn kiểm chứng:** trình chọn file Chrome chưa kiểm tra được do extension chưa
có quyền truy cập file URL, và quyền điều khiển native chưa được cấp. Không thay các
quyền đó. Luồng upload hợp lệ/không hợp lệ đã pass bằng AppTest với bytes CSV thật;
browser dùng nút dữ liệu mẫu. Live provider không được gọi trong đợt giao diện này.
Lịch sử/queue vẫn chỉ liệt kê 50 công việc gần nhất theo backend hiện có.

### Ảnh giao diện

Ảnh gốc từ browser, chỉ dùng dữ liệu giả; không dựng lại bằng mockup.

| Khu vực | Desktop 1440px | Mobile 390px |
|---|---|---|
| Tổng quan | [Ảnh](images/dashboard-redesign/overview-1440.jpg) | [Ảnh](images/dashboard-redesign/overview-390.jpg) |
| Khách hàng & dữ liệu | [Ảnh](images/dashboard-redesign/customers-1440.jpg) | [Ảnh](images/dashboard-redesign/customers-390.jpg) |
| Phân tích rời bỏ | [Ảnh](images/dashboard-redesign/analysis-1440.jpg) | [Ảnh](images/dashboard-redesign/analysis-390.jpg) |
| Kế hoạch giữ chân | [Ảnh](images/dashboard-redesign/plan-1440.jpg) | [Ảnh](images/dashboard-redesign/plan-390.jpg) |
| Chiến dịch | [Ảnh](images/dashboard-redesign/campaigns-1440.jpg) | [Ảnh](images/dashboard-redesign/campaigns-390.jpg) |

Ảnh kích thước trung gian: [Tổng quan 1280](images/dashboard-redesign/overview-1280.jpg),
[Kế hoạch 1280](images/dashboard-redesign/plan-1280.jpg),
[Khách hàng 1024](images/dashboard-redesign/customers-1024.jpg),
[Phân tích 1024](images/dashboard-redesign/analysis-1024.jpg),
[Kế hoạch 1024](images/dashboard-redesign/plan-1024.jpg),
[Chiến dịch 1024](images/dashboard-redesign/campaigns-1024.jpg).
Một số lần chụp bắt sai trạng thái compositor của menu khi đổi viewport; ảnh đó
không dùng làm ảnh bàn giao. Số đo DOM đủ 20 tổ hợp được lưu trong log nghiệm thu.

### Đối chiếu phạm vi

Đã đối chiếu toàn bộ bảng kiểm kê phía trên: mỗi chức năng cũ có vị trí mới. Chỉ loại
hai renderer UI cũ không còn nơi gọi; 15 định nghĩa service/view model còn lại trong
`dashboard_llm_ui.py` giống AST của main trước cải tạo. Hash 92 file dữ liệu, artifact,
database và thành phần bảo toàn không đổi. Provider/model/router/timeout, RAG,
Retention/Budget/Campaign/Queue, migrations và golden không sửa.

Main local và remote được đọc lại cùng ở
`c497806a29bbdf5b55d88a1b0d2849bc30a6cabb`. Hai stash và worktree tích hợp giữ nguyên.
Các commit giao diện chỉ nằm trên `codex/dashboard-redesign`; không push hoặc merge.
