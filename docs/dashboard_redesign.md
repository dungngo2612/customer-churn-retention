# Trung tâm điều hành giữ chân khách hàng

Bản cải tạo trên `codex/dashboard-redesign`, từ main đã xác minh tại
`c497806a29bbdf5b55d88a1b0d2849bc30a6cabb`. Chưa push/merge vào main.
Stack Streamlit và toàn bộ service nghiệp vụ, LLM/RAG được giữ nguyên.

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
PYTHONPATH=src python3 -m churn_retention.dashboard.demo --port 55510
```

Launcher giữ loopback, mock mặc định và database demo riêng. Không nhập dữ liệu thật
để thử giao diện; dùng tệp tổng hợp. Font hệ thống hỗ trợ tiếng Việt, không tải font
hoặc tài nguyên ngoài. Main, dữ liệu/artifact/database có sẵn, stash và worktree không đổi.

## Kiểm chứng

Kết quả regression, browser và ảnh được bổ sung sau khi nghiệm thu trên branch.
