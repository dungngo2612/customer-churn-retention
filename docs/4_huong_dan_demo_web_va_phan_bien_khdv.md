# SỔ TAY DEMO WEB CHI TIẾT & DỰ BÁO 15 CÂU HỎI PHẢN BIỆN MÔN KHOA HỌC DỊCH VỤ
*(Chuẩn 100% nhánh `demo-test-04-khdv` / `demo-05` — URL: `http://127.0.0.1:8505`)*

---

## PHẦN 1: GIẢI THÍCH TỪNG THÀNH PHẦN TRÊN WEB & LỜI DẪN KHI DEMO

### A. Đầu trang (Header & Thanh Kịch Bản Kinh Tế)
* **Gồm những gì:** Tiêu đề hệ thống, thanh trạng thái kịch bản đang áp dụng (`Biên lợi nhuận m = 50%`, `Kỳ hạn đánh giá H = 12 tháng`, `7,043 hồ sơ đang nạp`) và thẻ trạng thái mô hình (`CatBoost Bundle — ROC-AUC 84.8% Validation`).
* **Tác dụng & Ý nghĩa:** Giúp người xem (ban giám đốc / cô giáo) biết ngay mọi con số trên cả 4 tab đang được tính dưới **bộ giả định kinh tế nào** và dùng **mô hình AI thật** chứ không phải số liệu cứng.
* **Lời dẫn mẫu:** *"Ở trên cùng là thanh kịch bản kinh tế đang áp dụng — hiện tại là biên lợi nhuận 50%, kỳ hạn 12 tháng trên 7.043 hồ sơ, chạy bằng mô hình CatBoost thật đạt ROC-AUC 84,8%."*

---

### B. Thanh điều khiển bên trái (Sidebar)
1. **`Tham Số Kinh Tế (CLV)` — Thanh trượt `m` (10%–70%) & Chọn `H` (6/12/24 tháng):**
   * **Tác dụng:** Thay đổi giả định tài chính theo thời gian thực để tính $V = \text{MonthlyCharges} \times H \times m$.
   * **Ý nghĩa:** Cho thấy mô hình linh hoạt theo từng thời kỳ kinh doanh của nhà mạng (ví dụ mùa khuyến mãi biên lợi nhuận giảm xuống 30% thì kế hoạch giữ chân tự co lại).
2. **`Cấu Hình Gói Can Thiệp` (`A1`, `A2`, `A3`):**
   * **Tác dụng:** Cho phép chỉnh chi phí liên hệ ($c$), chi phí ưu đãi ($f$), tỷ lệ chấp nhận ($u$) và hiệu lực giảm churn ($e$) của từng gói.
   * **Ý nghĩa:** Doanh nghiệp không bị khóa cứng vào một bộ tham số cố định; khi phòng Marketing đổi giá voucher hoặc đo được tỷ lệ chấp nhận thực tế mới, chỉ cần nhập vào đây là hệ thống tự tính lại toàn bộ chiến dịch.
3. **`Đổi file CSV / Hạ tầng hệ thống`:**
   * **Tải lên file CSV mới:** Kiểm tra cấu trúc cột Telco, chạy lại CatBoost trên tệp khách hàng mới (ví dụ tệp `demo_telco_test_1000.csv` gồm 1.000 khách).
   * **Nút `Chạy & Ghi Nhận Scoring Job` + Ô tích `Tạo lượt chạy mới` + Bảng lịch sử Job (`Job | Số KH | Trạng thái | Retry`):**
     * **Tác dụng:** Đẩy lượt chấm điểm vào hàng đợi lưu vết (`data/task_queue.db`) kèm **khóa chống trùng (`idempotency_key`)**.
     * **Ý nghĩa:** Ngoài đời, khi chấm điểm định kỳ theo tuần/tháng, hệ thống cần lưu vết kiểm toán (ai chạy, file nào, bao nhiêu khách, model nào) và chống bấm nhầm gây ghi đè trùng lặp dữ liệu.
* **Lời dẫn mẫu:** *"Thanh bên trái là nơi chuyên viên vận hành cấu hình tham số kinh tế, thay đổi chi phí các gói A1–A3, và đặc biệt là nạp tệp CSV mới theo chu kỳ tuần/tháng kèm cơ chế lưu vết Task Queue chống chạy trùng."*

---

### C. Tab 1 — `01. Tổng Quan Kinh Tế` (Góc nhìn Giám đốc / Quản lý)
1. **4 thẻ KPI đầu trang:**
   * `Khách Hàng Đang Hoạt Động` (`5,163` Active) · `Đủ Điều Kiện Can Thiệp` (`1,756` KH — `34.0%`) · `Tổng Chi Phí Can Thiệp` (`$23,184` — `$13.2/KH`) · `Lợi Nhuận Ròng Kỳ Vọng` (`+$31,999` — `ROI +138.0%`).
   * **Ý nghĩa:** Trả lời ngay câu hỏi lớn nhất của lãnh đạo: *"Nếu triển khai chiến dịch này thì bỏ bao nhiêu tiền, giữ bao nhiêu người và lời được bao nhiêu?"* — đồng thời chứng minh hệ thống **tự loại bỏ 66% khách không hiệu quả (`A0`)** để tiết kiệm tiền.
2. **Biểu đồ `Cơ Cấu Phân Bổ Hành Động` (`SHARE OF WALLET`) & `Cân Đối Dòng Tiền` (`CASHFLOW BRIDGE`):**
   * **Ý nghĩa:** Trực quan hóa tỷ trọng từng gói `A0, A1, A2, A3` và so sánh **Lợi ích gộp bảo vệ được** với **Chi phí bỏ ra** để thấy rõ phần thặng dư lợi nhuận ròng ($\Delta EV > 0$).
3. **Bảng `So Sánh Hiệu Quả Tài Chính Từng Gói` (`PACKAGE ECONOMICS`):**
   * **Ý nghĩa:** So sánh trực diện hiệu quả giữa `A1`, `A2`, `A3` xem gói nào đóng góp nhiều lợi nhuận nhất.
* **Lời dẫn mẫu:** *"Tab 1 dành cho cấp quản lý nhìn nhanh bức tranh tài chính: trong 5.163 khách đang hoạt động, hệ thống chỉ chọn 1.756 khách thực sự có lời để can thiệp, bỏ ra 23.184 USD và kỳ vọng mang về 31.999 USD lợi nhuận ròng, đạt ROI 138%."*

---

### D. Tab 2 — `02. Hành Vi Churn & Giải Thích Mô Hình` (Góc nhìn Thiết kế Dịch vụ)
1. **4 thẻ KPI thực trạng gốc (`7,043` KH | `26.5%` Churn | `$64.8` Cước TB | `32.4M` Gắn bó TB):**
   * **Lưu ý quan trọng:** 4 số này **cố định theo file CSV đang nạp** (không đổi khi kéo slider `m`, `H`) vì nó mô tả sự thật lịch sử của tập dữ liệu; chỉ đổi khi nạp file CSV khác.
2. **4 biểu đồ EDA (`Contract`, `InternetService`, `TechSupport`, `PaymentMethod`):**
   * **Tác dụng & Ý nghĩa:** Là **bằng chứng dữ liệu** giải thích vì sao nhóm thiết kế đúng 3 gói `A1, A2, A3`:
     * Hợp đồng từng tháng rời mạng `42.7%` (gấp `15.1` lần hợp đồng 2 năm `2.8%`) → thiết kế **`A3`**.
     * Thanh toán thủ công `Electronic check` rời mạng `45.3%` → thiết kế **`A2`**.
     * Thiếu `TechSupport` rời mạng `41.6%` → thiết kế **`A1`**.
     * Cáp quang `Fiber optic` rời mạng `41.9%` (gấp `2.2` lần DSL `19.0%`) → khách cước cao rất nhạy cảm với chất lượng.
3. **Biểu đồ `Top 10 Đặc Trưng Ảnh Hưởng Nhất` (`CatBoost Feature Importances`):**
   * **Ý nghĩa:** Cho thấy mô hình CatBoost đang dựa vào đúng những yếu tố hành vi trọng yếu (`Contract`, `tenure`, `MonthlyCharges`, `TechSupport`...) chứ không học nhiễu.
* **Lời dẫn mẫu:** *"Tab 2 trả lời câu hỏi 'Vì sao nhóm lại thiết kế 3 gói A1, A2, A3?'. Bốn biểu đồ hành vi ở đây chứng minh bằng số liệu thật, và khớp hoàn toàn với Top 10 đặc trưng mà mô hình CatBoost coi trọng nhất ở bên dưới."*

---

### E. Tab 3 — `03. Hồ Sơ Khách Hàng 360°` (Góc nhìn Nhân viên CSKH)
1. **Bộ lọc & Bảng Danh Sách Đề Xuất (`CUSTOMER SEGMENTATION`):**
   * **Tác dụng:** Lọc theo mã khách, gói đề xuất, mức rủi ro và tải file `.CSV` danh sách đang lọc. **Mỗi mã khách hàng chỉ xuất hiện 1 dòng với 1 gói tốt nhất.**
2. **3 cột phân tích chi tiết 1 khách hàng (VD: `4132-KALRO`):**
   * **Cột 1 — `Hồ Sơ Khách Hàng` & Đồng hồ rủi ro:** Hiển thị rủi ro hiện tại (`84.6%`) và vạch xanh rủi ro sau can thiệp (`67.7%`, giảm `16.9%`) kèm thông tin hợp đồng, cước, dịch vụ.
   * **Cột 2 — `Quyết Định Tối Ưu & Điểm Hòa Vốn`:** Hiển thị gói đề xuất (`A3`), lợi nhuận ròng (`+$83.40`), giá trị bảo vệ (`$605.10`), ngưỡng hòa vốn (`p* = 15.7%`) và kết luận vì sao $p > p^*$ thì có lời.
   * **Cột 3 — `Yếu Tố Tác Động Rủi Ro Đến Cá Nhân` (`TreeSHAP`):** Giải thích riêng cho khách hàng này: thanh đỏ (`+`) đẩy rủi ro tăng, thanh xanh (`−`) giúp giảm rủi ro.
3. **`Gợi Ý Hành Động Cho Nhân Viên CSKH` (`CSKH ACTION CARD`):**
   * **Ý nghĩa:** Chuyển ngôn ngữ AI thành 3 ô nghiệp vụ dễ đọc cho tổng đài viên: *1. Nguyên nhân rủi ro chính · 2. Đề xuất & Kênh liên hệ ưu tiên (Tổng đài 1-1 / Zalo OA) · 3. Gợi ý chăm sóc mở rộng*.
4. **`So Sánh Giá Trị Ròng` (`ACTION MATRIX`) & `Phê Duyệt`:**
   * **Ý nghĩa:** Đối chiếu mọi gói hợp lệ (`A0..A3`) của khách hàng đó, tô xanh gắn nhãn **`TỐI ƯU`** cho gói có $\Delta EV$ cao nhất, và cho phép nhân viên bấm phê duyệt.
* **Lời dẫn mẫu:** *"Tab 3 dành cho nhân viên CSKH soi chi tiết từng khách hàng. Ví dụ khách `4132-KALRO` có rủi ro 84,6%. Biểu đồ TreeSHAP chỉ rõ lý do là mới dùng 4 tháng, cước cao 100,85 USD và hợp đồng tháng. Thẻ gợi ý CSKH tóm gọn luôn gói A3 và kênh gọi Outbound 1-1; bên dưới là bảng đối chiếu cả 4 gói để thấy vì sao A3 là phương án tối ưu."*

---

### F. Tab 4 — `04. Tối Ưu Ngân Sách & Chiến Dịch` (Góc nhìn Hoạch định Chiến dịch)
1. **Thanh trượt `Thiết Lập Hạn Mức Ngân Sách` (`KNAPSACK SOLVER`) & `Phân Bổ Ngân Sách` (`TOP ALLOCATIONS`):**
   * **Tác dụng:** Kéo hạn mức ngân sách (VD: `$15,000`), hệ thống tự chọn danh sách khách hàng và gói có hiệu suất **$\Delta EV / C$** cao nhất (`1,679` KH, thu về `+$27,503.41`, `ROI +183.4%`).
2. **`Chốt Kịch Bản Ngân Sách` & `Lịch Sử Chiến Dịch Đã Lưu` (`CAMPAIGN PERSISTENCE`):**
   * **Tác dụng:** Lưu vĩnh viễn kịch bản chiến dịch vào cơ sở dữ liệu (không mất khi tắt web) và xuất file `.CSV` danh sách khách hàng bàn giao cho đội tổng đài.
3. **2 biểu đồ `MARGIN SENSITIVITY` & `ACTION MIX` (`20% – 50%`):**
   * **Ý nghĩa:** Kiểm tra độ bền tài chính — xem khi biên lợi nhuận ($m$) thay đổi từ `20%` lên `50%` thì tổng lợi nhuận ròng, chi phí và cơ cấu các gói `A1, A2, A3` biến động thế nào.
* **Lời dẫn mẫu:** *"Tab 4 giải quyết bài toán thực tế nhất: ngân sách có hạn. Khi em đặt trần 15.000 USD, thuật toán Knapsack ưu tiên các khách có tỷ suất lời trên chi phí cao nhất, giữ được 1.679 khách với ROI 183,4%, lưu lịch sử chiến dịch và xuất ngay file CSV cho tổng đài triển khai."*

---

### G. Ý nghĩa bảng màu trên toàn bộ Web (nếu cô hỏi)
* **Xanh dương (`#3366FF`):** Quy mô nền tảng, điều hướng tab và nút hành động chính.
* **Đỏ hồng (`#FF3D71`):** Rủi ro, cảnh báo rời mạng, yếu tố đẩy nguy cơ tăng.
* **Xanh ngọc (`#00B383`):** Lợi nhuận ròng ($\Delta EV > 0$), gói **`TỐI ƯU`**, yếu tố giúp giảm rủi ro.
* **Vàng cam (`#FFAA00`):** Chi phí can thiệp hoặc trạng thái cần chú ý (thanh toán thủ công).

---

## PHẦN 2: DỰ BÁO 15 CÂU HỎI PHẢN BIỆN CỦA GIẢNG VIÊN & CÁCH TRẢ LỜI CHUẨN

### Nhóm A — Về Dữ liệu & Tính thực tế
1. **Hỏi: *Bộ dữ liệu Kaggle này cũ rồi (2017–2018), ngoài đời hành vi thay đổi theo tuần/tháng thì áp dụng sao được?***
   * **Trả lời:** *"Dạ đúng là dữ liệu benchmark của IBM từ giai đoạn trước, nhóm dùng vì dữ liệu cước thuê bao thật của nhà mạng là dữ liệu mật không thể tiếp cận công khai. Tập này là ảnh chụp tĩnh (Snapshot) tại một kỳ chốt cước. Ngoài thực tế, hệ thống của nhóm đã có sẵn tính năng **nạp file CSV mới theo tuần/tháng** ở thanh bên trái và tách riêng **Model Bundle**, nên doanh nghiệp chỉ cần nạp snapshot mới để chấm điểm và định kỳ retrain lại mô hình theo dữ liệu 3–6 tháng gần nhất để tránh trôi dữ liệu (Data Drift)."*
2. **Hỏi: *Nếu nạp file CSV khách hàng mới của tháng này (chưa biết ai rời mạng nên không có cột `Churn`) thì web có chạy lỗi không?***
   * **Trả lời:** *"Không lỗi ạ. Mô hình CatBoost chỉ cần 19 cột đặc trưng đầu vào để dự đoán xác suất rời mạng ($p$) và tự động bỏ qua cột `Churn` nếu có. Trên Tab 1, 3, 4 mọi tính toán kinh tế đều chạy hoàn toàn bình thường."*
3. **Hỏi: *Tập dữ liệu bị mất cân bằng (chỉ có 26,5% Churn), nhóm đánh giá mô hình bằng Accuracy có bị ảo không?***
   * **Trả lời:** *"Chính vì tập mất cân bằng nên nhóm **không** đánh giá chính bằng Accuracy, mà dùng **ROC-AUC (84,8%)**, **PR-AUC (0,654)** để đo khả năng xếp hạng rủi ro, và **Brier Score (0,134)** để kiểm tra độ hiệu chuẩn xác suất trước khi đưa $p$ vào công thức tính tiền $\Delta EV$."*
4. **Hỏi: *Dữ liệu gốc có 11 dòng trống ở `TotalCharges`, nhóm xử lý thế nào?***
   * **Trả lời:** *"11 dòng đó rơi vào các khách hàng mới hòa mạng (`tenure = 0`) nên chưa phát sinh tổng cước tích lũy. Khi nạp file, hệ thống tự động ép kiểu số và chuẩn hóa về `0.0` để tránh lỗi khi dự báo."*

### Nhóm B — Về Mô hình AI & Giải thích
5. **Hỏi: *Con số ROC-AUC 84,8% hiểu đơn giản là gì?***
   * **Trả lời:** *"Hiểu trực quan là nếu lấy ngẫu nhiên 1 khách đã rời mạng và 1 khách ở lại trong tập kiểm định, mô hình CatBoost xếp khách rời mạng có điểm rủi ro cao hơn trong khoảng **85 trên 100 cặp**."*
6. **Hỏi: *Biểu đồ giải thích ở Tab 2 và Tab 3 giống hay khác nhau?***
   * **Trả lời:** *"Khác nhau về cấp độ ạ. **Tab 2** là `CatBoost Feature Importances` (toàn cục — cho biết trên toàn mô hình cột nào quan trọng nhất để thiết kế chính sách); còn **Tab 3** là `Local TreeSHAP` (cá nhân — giải thích riêng cho 1 khách hàng đang chọn xem yếu tố nào đang kéo rủi ro lên hay xuống)."*
7. **Hỏi: *TreeSHAP báo thiếu `TechSupport` làm tăng rủi ro, vậy tặng `TechSupport` có chắc chắn 100% giữ được khách không?***
   * **Trả lời:** *"Không chắc chắn 100% ạ, vì TreeSHAP chỉ phản ánh **mối tương quan thống kê** của mô hình, không phải quan hệ nhân quả tuyệt đối (ngoài đời khách có thể rời mạng vì chuyển nhà hoặc đối thủ phá giá). Vì vậy hệ thống chỉ đưa ra gợi ý để nhân viên CSKH kiểm tra nhu cầu thật trước khi áp dụng, và trong công thức kinh tế nhóm cũng đã nhân thêm tỷ lệ chấp nhận ($u$) và hiệu lực giảm rủi ro ($e < 100\%$)."*
8. **Hỏi: *Tại sao nhóm chọn CatBoost mà không dùng LLM hay Logistic Regression?***
   * **Trả lời:** *"So với Logistic Regression, CatBoost bắt được các quan hệ phi tuyến và xử lý trực tiếp 15 cột phân loại rất tốt. So với LLM, CatBoost chấm hàng loạt 7.043 khách chỉ mất ~1 giây, 0 đồng phí API, có chỉ số đo chuẩn (`ROC-AUC 84,8%`) và kết quả nhất quán 100%. Hướng LLM phù hợp làm bước mở rộng về sau để viết kịch bản thoại cá nhân hóa cho nhóm VIP."*

### Nhóm C — Về Kinh tế học & Thuật toán Tối ưu
9. **Hỏi: *Tại sao gọi là Proxy CLV ($V = \text{MonthlyCharges} \times H \times m$) mà không tính CLV trọn đời?***
   * **Trả lời:** *"Vì bộ dữ liệu không có chi phí thu hút khách (CAC) hay tỷ lệ chiết khấu dòng tiền dài hạn, nên ước lượng giá trị đóng góp trong kỳ hạn ngắn $H = 12$ tháng (Proxy CLV) sẽ thận trọng, minh bạch và tránh thổi phồng lợi nhuận giả tạo."*
10. **Hỏi: *Các tham số $u$ (tỷ lệ chấp nhận) và $e$ (hiệu lực giảm churn) lấy ở đâu ra? Con số $\Delta EV$ có chắc chắn lời thật không?***
    * **Trả lời:** *"Hiện tại $u, e, c, f$ là **tham số giả định mô phỏng** hợp lý theo nghiệp vụ viễn thông để kiểm chứng mô hình, nên $\Delta EV$ là **lợi nhuận kỳ vọng mô phỏng**. Chính vì vậy nhóm làm sẵn thanh trượt ở sidebar và biểu đồ độ nhạy `MARGIN SENSITIVITY` ở Tab 4 để doanh nghiệp thử nhiều kịch bản; khi triển khai thật cần chạy **A/B Testing** trên nhóm nhỏ để đo đúng $u$ và $e$ thực tế."*
11. **Hỏi: *Tại sao cùng 1 khách hàng, Tab 3 đề xuất gói `A3` mà sang Tab 4 khi đặt ngân sách `$15,000` lại chọn gói `A2`?***
    * **Trả lời:** *"Vì mục tiêu hai màn hình khác nhau ạ. Tab 3 tối ưu riêng 1 người khi **không giới hạn tiền** nên chọn gói có $\Delta EV$ tuyệt đối cao nhất (`A3` lời `+$83,40`). Còn Tab 4 bị **giới hạn tổng ngân sách** nên thuật toán Knapsack ưu tiên theo **tỷ suất lời trên mỗi đô-la chi phí ($\Delta EV / C$)** — gói `A2` tốn chỉ `$9` nhưng lời `$62,68` (hiệu suất `6,96x`, cao hơn `A3` là `4,39x`), nhờ vậy giữ được nhiều khách hơn và tổng lợi nhuận toàn chiến dịch cao hơn."*
12. **Hỏi: *Một khách hàng có thể được đề xuất nhiều gói cùng lúc không?***
    * **Trả lời:** *"Không ạ. Một khách có thể hợp lệ nhiều gói (xem ở bảng `Action Matrix` Tab 3), nhưng trong danh sách chiến dịch mỗi khách hàng chỉ nhận **tối đa 1 gói tối ưu nhất** để tránh gọi điện chồng chéo làm phiền khách và đội chi phí."*

### Nhóm D — Về Vận hành & Khoa học Dịch vụ
13. **Hỏi: *Chức năng `Task Queue` và `Khóa chống trùng (Idempotency Key)` ở thanh bên trái có ý nghĩa gì?***
    * **Trả lời:** *"Giống như app ngân hàng chặn trừ tiền 2 lần khi lỡ bấm đúp, `idempotency_key` gắn với từng bộ `(tên file + số dòng + phiên bản model)` giúp tránh chạy lặp tốn tài nguyên và không ghi đè trùng kết quả vào cơ sở dữ liệu kiểm toán `task_queue.db`."*
14. **Hỏi: *Hệ thống này có tự động gửi SMS hay tự tặng voucher cho khách hàng không?***
    * **Trả lời:** *"Không ạ. Hệ thống tuân thủ nguyên tắc **Human-in-the-loop** (con người ra quyết định cuối): phần mềm ở lớp Backstage tính toán và xuất file `.CSV` + Thẻ gợi ý, còn nhân viên CSKH ở lớp Frontstage sẽ rà soát, gọi điện xác nhận nhu cầu thật rồi mới áp dụng ưu đãi."*
15. **Hỏi: *5 hướng dịch vụ mở rộng (người cao tuổi, gói gia đình, giải trí...) đã có công thức tính tiền $\Delta EV$ chưa?***
    * **Trả lời:** *"Chưa ạ — nhóm minh bạch rõ là 4 gói `A0–A3` đã có đầy đủ công thức kinh tế $\Delta EV$, còn **5 hướng dịch vụ mở rộng** là phần thiết kế dịch vụ (Service Design) khai thác 9 cột dữ liệu còn lại để gợi ý tư vấn thêm ngay trên Thẻ CSKH ở Tab 3; trước khi định giá thương mại cần khảo sát nhu cầu thật của khách hàng."*
