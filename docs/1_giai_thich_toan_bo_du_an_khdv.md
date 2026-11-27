# CẨM NANG GIẢI THÍCH TOÀN BỘ DỰ ÁN (CHUẨN NHÁNH `demo-test-04-khdv` / `demo-05`)

**Tên đề tài:**  
> **“Xây dựng hệ thống dự báo khách hàng có khả năng rời bỏ và tối ưu hóa chiến dịch giữ chân dựa trên giá trị kinh tế (Kèm Web mô phỏng)”**

**Môn học:** Khoa học Dịch vụ (Service Science)

---

## PHẦN 1: BỨC TRANH TỔNG THỂ — DỰ ÁN NÀY GIẢI QUYẾT BÀI TOÁN GÌ?

### 1.1. Hai trụ cột cốt lõi của hệ thống
Dự án kết hợp giữa **Khoa học Dữ liệu (AI/ML)** và **Kinh tế Dịch vụ (CLV & Tối ưu ngân sách)**:

1. **Mô hình Dự báo & Giải thích (CatBoost + TreeSHAP):**
   - Chấm điểm xác suất rời mạng ($p$) hàng loạt cho hàng nghìn khách hàng trong vài giây bằng mô hình **CatBoost** (`catboost-20261018T130222-seed42`, đạt **ROC-AUC 84.8%** trên tập validation gồm 1.056 hồ sơ).
   - Giải thích rõ nguyên nhân ở cả 2 cấp độ: toàn cục trên tập dữ liệu (Tab 2) và từng khách hàng cụ thể bằng **TreeSHAP** (Tab 3).
2. **Mô hình Kinh tế Giữ chân & Phân bổ Ngân sách ($\Delta EV$, $p^*$, Knapsack):**
   - Không phải cứ thấy rủi ro cao là tặng quà. Hệ thống tính **lợi nhuận ròng kỳ vọng ($\Delta EV$)** và **ngưỡng hòa vốn ($p^*$)** cho từng khách hàng.
   - Khi ngân sách bị giới hạn (ví dụ \$15,000), hệ thống dùng thuật toán tối ưu **Knapsack** chọn ra những khách hàng và gói can thiệp (`A1`, `A2`, `A3`) có hiệu suất sinh lời trên mỗi đồng chi phí ($\Delta EV / C$) cao nhất.

> **Thông điệp cốt lõi khi thuyết trình:**  
> *"Hệ thống không chỉ trả lời câu hỏi 'Ai sắp rời mạng?', mà trả lời trọn vẹn 'Tại sao họ rời mạng, có đáng bỏ tiền giữ không, tặng gói nào để có lời, và với ngân sách hiện có thì nên gọi cho ai trước?'"*

---

## PHẦN 2: DỮ LIỆU ĐẦU VÀO & BÀI TOÁN "DỮ LIỆU THEO KỲ"

### 2.1. Tập dữ liệu sử dụng
* **Bộ dữ liệu nền tảng (Benchmark):** **IBM Telco Customer Churn** (`7,043` khách hàng, `21` cột dữ liệu — trong đó có `11` dòng trống ở `TotalCharges` được chuyển đổi an toàn về `0.0` khi nạp).
* **Vì sao dùng bộ dữ liệu này & trả lời thế nào khi cô hỏi "Dữ liệu này cũ rồi / ngoài đời phải lấy theo tuần, tháng?"**
  * Dữ liệu thuê bao thật của các nhà mạng (Viettel, VNPT, MobiFone...) là **dữ liệu mật (thông tin cá nhân khách hàng & bí mật kinh doanh)** nên đồ án môn học dùng bộ benchmark chuẩn của IBM để kiểm chứng giải pháp.
  * **Bản chất dữ liệu:** Là **ảnh chụp tĩnh (Snapshot)** tại một kỳ chốt cước, nhãn `Churn` ghi nhận trạng thái rời mạng trong chu kỳ ngay sau đó.
  * **Kiến trúc thực tế ngoài doanh nghiệp:** Hệ thống **không đóng cứng** vào 7.043 dòng này. Ở thanh bên trái có chức năng **Tải lên file CSV mới** (ví dụ file 1.000 dòng `demo_telco_test_1000.csv`) + **Task Queue (`task_queue.db`)**. Cuối mỗi tuần hoặc kỳ chốt cước hàng tháng, nhà mạng chỉ cần xuất snapshot mới nạp vào là chấm điểm hàng loạt ngay; khi cần cập nhật hành vi mới theo thời gian (tránh trôi dữ liệu) chỉ cần huấn luyện lại `Model Bundle`.

### 2.2. Các nhóm thông tin chính trong 21 cột
* **Nhân khẩu học:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* **Gắn bó & Dịch vụ đang dùng:** `tenure` (số tháng gắn bó), `PhoneService`, `MultipleLines`, `InternetService` (`DSL`, `Fiber optic`, `No`), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
* **Hợp đồng & Thanh toán:** `Contract` (`Month-to-month`, `One year`, `Two year`), `PaperlessBilling`, `PaymentMethod` (`Electronic check`, `Mailed check`, chuyển khoản/thẻ tự động), `MonthlyCharges`, `TotalCharges`, và nhãn `Churn`.

---

## PHẦN 3: CÔNG THỨC KINH TẾ & 3 GÓI CAN THIỆP (`A0`, `A1`, `A2`, `A3`)

### 3.1. Công thức tính Giá trị & Lợi nhuận Ròng
1. **Giá trị bảo vệ khách hàng (Proxy CLV — $V$):**
   $$V = \text{MonthlyCharges} \times H \times m$$
   * $H$: Kỳ hạn đánh giá mặc định **$12$ tháng** (có thể đổi 6, 12, 24 tháng ở sidebar).
   * $m$: Biên lợi nhuận giả định mặc định **$50\%$** (kéo được từ $10\%$ đến $70\%$).
2. **Tổng chi phí kỳ vọng của một gói ($C$):**
   $$C = c + u \cdot f$$
   * $c$: Chi phí liên hệ (gọi điện/nhắn tin, mất kể cả khi khách từ chối).
   * $u$: Tỷ lệ khách chấp nhận gói ưu đãi.
   * $f$: Chi phí ưu đãi/voucher (chỉ mất khi khách đồng ý nhận).
3. **Lợi ích bảo vệ kỳ vọng ($B$) và Lợi nhuận ròng ($\Delta EV$):**
   $$B = V \cdot p \cdot u \cdot e \qquad \text{và} \qquad \Delta EV = B - C$$
   * $p$: Xác suất rời mạng do **CatBoost** dự báo.
   * $e$: Mức giảm xác suất rời mạng sau khi áp dụng gói.
4. **Ngưỡng rủi ro hòa vốn ($p^*$):**
   $$p^* = \frac{c + u \cdot f}{V \cdot u \cdot e}$$
   * Nếu **$p > p^*$** (tức $\Delta EV > 0$): Can thiệp **có lời** → Đề xuất giữ chân.
   * Nếu **$p \le p^*$** (tức $\Delta EV \le 0$): Can thiệp **bị lỗ hoặc hòa vốn** → Xếp vào **`A0 — Không can thiệp`** để bảo toàn ngân sách.

### 3.2. Chi tiết 4 phương án trên giao diện (thông số mặc định)
| Mã gói | Tên phương án | Điều kiện áp dụng | Chi phí liên hệ ($c$) | Chi phí ưu đãi ($f$) | Tỷ lệ chấp nhận ($u$) | Mức giảm churn ($e$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **`A0`** | Không can thiệp | Mặc định cho mọi khách | $\$0$ | $\$0$ | — | — |
| **`A1`** | Hỗ trợ kỹ thuật & bảo mật chủ động | Khách đang dùng Internet nhưng thiếu `TechSupport` hoặc `OnlineSecurity` | $\$2.0$ | $\$15.0$ | $40\%$ | $25\%$ |
| **`A2`** | Tư vấn hợp đồng & thanh toán tự động | Khách hợp đồng `Month-to-month` hoặc thanh toán thủ công | $\$2.0$ | $\$20.0$ | $35\%$ | $40\%$ |
| **`A3`** | Ưu đãi giảm cước cam kết dài hạn | Khách hợp đồng `Month-to-month` | $\$3.0$ | $\$40.0$ | $40\%$ | $50\%$ |

*(Mỗi mã khách hàng ở bảng đề xuất chỉ nhận **1 gói duy nhất** có $\Delta EV$ cao nhất để tránh gọi trùng hoặc tặng chồng chéo ngoài thực tế.)*

---

## PHẦN 4: HƯỚNG DẪN CHI TIẾT TỪNG MÀN HÌNH TRÊN WEB (`http://127.0.0.1:8505`)

### Thanh bên trái (Sidebar)
* **Tham Số Kinh Tế (CLV) & Cấu Hình Gói Can Thiệp:** Kéo thử $m$, $H$ hoặc chỉnh chi phí/hiệu quả của `A1, A2, A3`. Khi kéo, số liệu ở Tab 1, Tab 3, Tab 4 tự cập nhật ngay lập tức.
* **Đổi file CSV / Hạ tầng hệ thống:**
  * **Tải lên file CSV mới:** Kiểm tra đúng cấu trúc cột Telco, chạy lại CatBoost trên file mới.
  * **Chạy & Ghi Nhận Scoring Job:** Đẩy lượt chấm điểm vào **Task Queue (`data/task_queue.db`)** để lưu vết phục vụ kiểm toán. Có **khóa chống trùng (`idempotency_key`)** tránh ghi đè trùng lặp cùng 1 file + model; nếu muốn ép tạo lượt chạy mới khi demo thì tích ô *"Tạo lượt chạy mới (bỏ qua khóa chống trùng cũ)"*.

### Tab 1 — `01. Tổng Quan Kinh Tế`
* **4 thẻ KPI đầu trang:** Tổng tệp khách hàng, số khách đủ điều kiện giữ chân, tổng chi phí can thiệp và **Tổng lợi nhuận ròng kỳ vọng ($\Delta EV$)** cùng **ROI**.
* **Biểu đồ Cơ Cấu Phân Bổ Hành Động (`A0, A1, A2, A3`) & Cân Đối Dòng Tiền:** Cho ban giám đốc thấy tổng tiền bảo vệ được lớn hơn rất nhiều so với chi phí bỏ ra.
* **Bảng So Sánh Hiệu Quả Từng Gói:** Đối chiếu chi phí, lợi ích và $\Delta EV$ trung bình/tổng của từng gói.

### Tab 2 — `02. Hành Vi Churn & Giải Thích Mô Hình`
* **4 thẻ KPI thực trạng gốc:** Tính cố định theo file CSV đang nạp (`7,043` khách, `26.5%` churn, `$64.8` cước TB, `32.4` tháng gắn bó TB) — không bị ảnh hưởng khi kéo slider tài chính.
* **4 biểu đồ EDA hành vi:** Chứng minh bằng số liệu thật tại sao phải thiết kế 3 gói `A1, A2, A3` (ví dụ: *Month-to-month* rời mạng **42.7%**, gấp **15.1 lần** hợp đồng 2 năm **2.8%**; thiếu *TechSupport* rời mạng **41.6%**; *Electronic check* rời mạng **45.3%**).
* **Top 10 Đặc Trưng Ảnh Hưởng Nhất (`CatBoost Feature Importances`):** Xếp hạng các cột mà mô hình CatBoost dựa vào nhiều nhất trên toàn tập dữ liệu.

### Tab 3 — `03. Hồ Sơ Khách Hàng 360°`
* **Bảng lọc & xếp hạng khách hàng:** Mỗi dòng là 1 mã khách hàng duy nhất với gói tốt nhất, $p$, $p^*$, $V, C, B, \Delta EV$.
* **3 cột phân tích chi tiết 1 khách hàng:**
  1. **`Hồ Sơ Khách Hàng`:** Đồng hồ đo rủi ro trước và sau can thiệp (vạch xanh dương) + thông tin gói cước/hợp đồng.
  2. **`Quyết Định Tối Ưu & Điểm Hòa Vốn`:** Hiển thị gói đề xuất, $\Delta EV$, giá trị bảo vệ $V$ và chứng minh vì sao $p > p^*$ thì có lời.
  3. **`Yếu Tố Tác Động Rủi Ro Đến Cá Nhân` (`TreeSHAP`):** Biểu đồ thanh ngang cho biết yếu tố nào làm **tăng rủi ro (+ màu đỏ)** và yếu tố nào giúp **giảm rủi ro (− màu xanh)**.
* **`Gợi Ý Hành Động Cho Nhân Viên CSKH`:** Tóm tắt 3 ô ngắn gọn cho nhân viên tổng đài: *1. Nguyên nhân rủi ro chính*, *2. Đề xuất & Kênh liên hệ ưu tiên (Tổng đài 1-1 hoặc Zalo OA/App)*, *3. Gợi ý chăm sóc mở rộng*.
* **`So Sánh Giá Trị Ròng (Action Matrix)` & `Phê Duyệt`:** So sánh mọi phương án hợp lệ (`A0..A3`), gắn nhãn **`TỐI ƯU`** cho phương án có $\Delta EV$ cao nhất.

### Tab 4 — `04. Tối Ưu Ngân Sách & Chiến Dịch`
* **Thanh trượt Hạn mức ngân sách (`Knapsack Solver`):** Chọn giới hạn tiền (ví dụ $\$15,000$). Hệ thống ưu tiên chọn các khách hàng/gói có hiệu suất $\Delta EV / C$ cao nhất để tối đa hóa tổng lợi nhuận thu về trong hạn mức.
* **Lưu Chiến Dịch & Xuất CSV (`Campaign Persistence`):** Lưu kịch bản đã chốt vào cơ sở dữ liệu lịch sử (không bị mất khi tắt web) và xuất file `.CSV` bàn giao cho bộ phận CSKH gọi điện.
* **Biểu đồ `MARGIN SENSITIVITY` & `ACTION MIX`:** Mô phỏng khi biên lợi nhuận ($m$) thay đổi từ $20\%$ đến $50\%$ thì lợi nhuận ròng và cơ cấu các gói `A1, A2, A3` thay đổi ra sao.

---

## PHẦN 5: Ý NGHĨA MÀU SẮC TRÊN GIAO DIỆN (BẢNG MÀU NGỮ NGHĨA)
* **Xanh dương (`#3366FF`):** Quy mô nền tảng, tab đang chọn và nút hành động chính.
* **Đỏ hồng (`#FF3D71`):** Rủi ro, cảnh báo mất khách, yếu tố làm tăng nguy cơ churn.
* **Xanh ngọc (`#00B383`):** Giá trị giữ chân, lợi nhuận ròng dương ($\Delta EV > 0$), gói **`TỐI ƯU`**, yếu tố giúp giảm rủi ro.
* **Vàng cam (`#FFAA00`):** Chi phí bỏ ra hoặc trạng thái cần lưu ý (thanh toán thủ công, rủi ro trung bình).
