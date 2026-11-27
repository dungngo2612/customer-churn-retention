# CẨM NANG GIẢI THÍCH CHUYÊN SÂU TOÀN BỘ DỰ ÁN (TỪ A ĐẾN Z — CHUẨN NHÁNH `demo-test-04-khdv` / `demo-05`)

**Tên đề tài:**  
> **“Xây dựng hệ thống dự báo khách hàng có khả năng rời bỏ và tối ưu hóa chiến dịch giữ chân dựa trên giá trị kinh tế (Kèm Web mô phỏng)”**

**Môn học:** Khoa học Dịch vụ (Service Science)

---

## PHẦN 1: BỨC TRANH TỔNG THỂ & KIẾN TRÚC HỆ THỐNG

### 1.1. Bài toán thực tế cần giải quyết
Trong các doanh nghiệp dịch vụ thuê bao định kỳ (nhà mạng Internet, Di động, Truyền hình cáp), việc giữ chân khách hàng đối mặt với 3 bài toán lớn:
1. **Nhận diện muộn và thiếu lý giải:** Nếu chỉ biết một con số *"khách này có 80% nguy cơ rời mạng"* mà không biết *vì sao* (do cước cao, do chưa có hỗ trợ kỹ thuật, hay do hợp đồng từng tháng), nhân viên CSKH không biết tư vấn gì khi gọi điện.
2. **Đốt ngân sách không hiệu quả:** Không phải khách nào có nguy cơ rời mạng cao cũng đáng giữ bằng mọi giá. Nếu tặng ưu đãi $\$43$ cho một khách hàng đóng cước thấp (giá trị vòng đời chỉ $\$100$) hoặc khách có rủi ro chưa đủ cao, doanh nghiệp sẽ **bị lỗ ròng ($\Delta EV < 0$)**.
3. **Giới hạn ngân sách chiến dịch:** Ngoài đời, bộ phận CSKH chỉ có một khoản ngân sách cố định mỗi tháng (ví dụ $\$15,000$). Không thể gọi tặng ưu đãi cho toàn bộ vài nghìn khách có rủi ro — cần phải chọn ra **đúng danh sách khách hàng và đúng gói ưu đãi** sao cho tổng lợi nhuận thu về là lớn nhất.

### 1.2. Kiến trúc 5 lớp của hệ thống trên nhánh `demo-test-04-khdv` / `demo-05`
1. **Lớp Dữ liệu & Kiểm tra cấu trúc (`Dataset Validation`):** Nạp tệp CSV chuẩn 21 cột Telco (mặc định `7,043` khách hàng hoặc tệp CSV mới do người dùng tải lên), tự động chuẩn hóa cột `TotalCharges`.
2. **Lớp Dự báo AI (`CatBoost Bundle`):** Mô hình CatBoost đã huấn luyện (`catboost-20261018T130222-seed42`), chấm xác suất rời mạng ($p$) hàng loạt cho hàng nghìn khách hàng trong vài giây.
3. **Lớp Giải thích & Tư vấn Dịch vụ (`Explainability & CSKH Action Card`):**
   - Toàn cục: EDA 4 chiều + `CatBoost Feature Importances` (Tab 2).
   - Cá nhân: `TreeSHAP` từng khách hàng + `Thẻ Gợi Ý Hành Động CSKH` (Tab 3).
4. **Lớp Kinh tế & Tối ưu Ngân sách (`CLV / ΔEV / Knapsack`):** Tính giá trị bảo vệ $V$, ngưỡng hòa vốn $p^*$, lợi nhuận ròng $\Delta EV$ cho từng gói `A0, A1, A2, A3`, và giải bài toán tối ưu ngân sách Knapsack (Tab 1, 3, 4).
5. **Lớp Vận hành & Lưu vết (`Task Queue & Campaign Persistence`):** Lưu vết các đợt chấm điểm vào SQLite (`data/task_queue.db`) kèm khóa chống trùng (`idempotency_key`), lưu lịch sử chiến dịch và xuất file `.CSV` cho đội CSKH triển khai.

---

## PHẦN 2: DỮ LIỆU ĐẦU VÀO — Ý NGHĨA CHI TIẾT TOÀN BỘ 21 CỘT

### 2.1. Nguồn dữ liệu & Quy trình làm sạch
* **Bộ dữ liệu gốc:** **IBM Telco Customer Churn** (`data/raw/telco_churn.csv`), gồm **`7,043` dòng** và **`21` cột**.
* **Xử lý thực tế trên Web:** Trong file gốc, cột `TotalCharges` có `11` dòng bị chuỗi rỗng (ở các khách hàng mới hòa mạng có `tenure = 0`). Khi nạp file vào Dashboard, hệ thống tự động ép kiểu số và chuyển các ô trống đó về `0.0` để mô hình chấm điểm an toàn không bị lỗi.
* **Tập kiểm thử nạp ngoài:** Hệ thống hỗ trợ tải lên bất kỳ file CSV nào đúng cấu trúc Telco (ví dụ `demo_telco_test_1000.csv` gồm `1,000` khách hàng) ngay ở thanh bên trái.

### 2.2. Giải thích chi tiết toàn bộ 21 cột dữ liệu

#### Nhóm A: Định danh & Nhân khẩu học (5 cột)
| STT | Tên cột | Giá trị mẫu | Ý nghĩa thực tế trong dịch vụ viễn thông |
| :---: | :--- | :--- | :--- |
| 1 | `customerID` | `4132-KALRO` | Mã định danh duy nhất của từng thuê bao. |
| 2 | `gender` | `Male` / `Female` | Giới tính chủ thuê bao. |
| 3 | `SeniorCitizen` | `0` / `1` | `1` là khách hàng cao tuổi ($\ge 65$ tuổi), `0` là nhóm tuổi còn lại. |
| 4 | `Partner` | `Yes` / `No` | Có bạn đời / vợ chồng cùng hộ hay không. |
| 5 | `Dependents` | `Yes` / `No` | Có người phụ thuộc (con cái, cha mẹ) hay không (thường ổn định hơn, ít đổi mạng). |

#### Nhóm B: Thời gian gắn bó & Dịch vụ đang sử dụng (10 cột)
| STT | Tên cột | Giá trị mẫu | Ý nghĩa thực tế trong dịch vụ viễn thông |
| :---: | :--- | :--- | :--- |
| 6 | `tenure` | `0` – `72` (tháng) | Số tháng khách hàng đã gắn bó với nhà mạng. Khách mới (`tenure` nhỏ) có rủi ro rời mạng rất cao. |
| 7 | `PhoneService` | `Yes` / `No` | Có sử dụng dịch vụ điện thoại cố định hay không. |
| 8 | `MultipleLines` | `Yes` / `No` / `No phone service` | Có dùng nhiều đường dây điện thoại cùng lúc hay không (gợi ý gói gia đình). |
| 9 | `InternetService` | `DSL` / `Fiber optic` / `No` | Loại đường truyền Internet. Nhóm **Cáp quang (`Fiber optic`)** có tỷ lệ rời mạng cao (`41.9%`). |
| 10 | `OnlineSecurity` | `Yes` / `No` / `No internet service` | Dịch vụ bảo mật trực tuyến. Khách chưa đăng ký (`No`) dễ bỏ mạng hơn. |
| 11 | `OnlineBackup` | `Yes` / `No` / `No internet service` | Dịch vụ sao lưu dữ liệu đám mây. |
| 12 | `DeviceProtection` | `Yes` / `No` / `No internet service` | Dịch vụ bảo vệ/bảo hành thiết bị đầu cuối. |
| 13 | `TechSupport` | `Yes` / `No` / `No internet service` | Dịch vụ hỗ trợ kỹ thuật 24/7. Nhóm chưa có (`No`) rời mạng tới `41.6%`. |
| 14 | `StreamingTV` | `Yes` / `No` / `No internet service` | Dịch vụ truyền hình trực tuyến. |
| 15 | `StreamingMovies` | `Yes` / `No` / `No internet service` | Dịch vụ xem phim trực tuyến theo yêu cầu. |

#### Nhóm C: Hợp đồng, Thanh toán & Nhãn rời mạng (6 cột)
| STT | Tên cột | Giá trị mẫu | Ý nghĩa thực tế trong dịch vụ viễn thông |
| :---: | :--- | :--- | :--- |
| 16 | `Contract` | `Month-to-month` / `One year` / `Two year` | Thời hạn hợp đồng. Nhóm từng tháng (`Month-to-month`) rời mạng **42.7%**, gấp **15.1 lần** hợp đồng 2 năm (**2.8%**). |
| 17 | `PaperlessBilling` | `Yes` / `No` | Nhận hóa đơn điện tử thay vì hóa đơn giấy. |
| 18 | `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Phương thức thanh toán. Thanh toán thủ công (`Electronic check`) có tỷ lệ rời mạng cao (`45.3%`). |
| 19 | `MonthlyCharges` | Số thực (`$18.25` – `$118.75`) | Cước phí hàng tháng (USD) — biến đầu vào trực tiếp để tính **Giá trị bảo vệ ($V$)**. |
| 20 | `TotalCharges` | Số thực (`$0` – `$8,684.8`) | Tổng tiền cước tích lũy từ lúc hòa mạng đến hiện tại. |
| 21 | `Churn` | `Yes` / `No` | Nhãn lịch sử cho biết khách hàng có rời mạng trong kỳ quan sát hay không (có thể vắng mặt ở file khách hàng mới cần dự báo). |

---

## PHẦN 3: MÔ HÌNH AI — CATBOOST & GIẢI THÍCH 2 TẦNG

### 3.1. Vì sao chọn CatBoost thay vì thuật toán khác?
* Trong 21 cột dữ liệu Telco có tới **15 cột phân loại (Categorical)** như `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`...
* **CatBoost** xử lý trực tiếp biến phân loại rất mạnh mà không bị phình số chiều, tốc độ dự báo hàng loạt cực nhanh (chấm 7.043 khách chỉ mất khoảng 1 giây), không phụ thuộc API bên ngoài.
* **Chỉ số kiểm chứng thực tế lưu trong `metadata.json`** (`catboost-20261018T130222-seed42`, huấn luyện trên `4,930` mẫu, kiểm định trên `1,056` mẫu):
  * **ROC-AUC (Validation):** **`0.8476` (~`84.8%`)** — nghĩa là lấy ngẫu nhiên 1 khách rời mạng và 1 khách ở lại, mô hình xếp khách rời mạng có điểm rủi ro cao hơn trong khoảng **85/100 cặp**.
  * **PR-AUC (Average Precision):** `0.6535`
  * **Brier Score:** `0.1344` (độ hiệu chuẩn xác suất tốt, rất quan trọng vì xác suất $p$ đi thẳng vào công thức tính tiền $\Delta EV$).

### 3.2. Phân biệt rõ 2 biểu đồ giải thích ở Tab 2 và Tab 3 (tránh nhầm khi cô hỏi!)
| Cấp độ | Vị trí | Nguồn tính toán | Trả lời câu hỏi gì? |
| :--- | :--- | :--- | :--- |
| **Toàn cục (Global)** | **Tab 2** (`Top 10 Đặc Trưng Ảnh Hưởng Nhất`) | Bảng `feature_importances_` của mô hình CatBoost | *"Nhìn chung trên toàn mô hình, cột thông tin nào quan trọng nhất khi dự báo?"* |
| **Cá nhân (Local)** | **Tab 3** (`Yếu Tố Tác Động Rủi Ro Đến Cá Nhân`) | Giá trị **TreeSHAP** tính riêng cho 1 khách hàng (`ShapValues`) | *"Riêng với khách hàng `4132-KALRO` này, yếu tố nào đang đẩy rủi ro tăng (+) và yếu tố nào kéo giảm (−)?"* |

---

## PHẦN 4: TOÁN HỌC KINH TẾ & VÍ DỤ TÍNH TAY TỪNG CON SỐ

### 4.1. Bộ công thức chuẩn của dự án
1. **Giá trị bảo vệ khách hàng (Proxy CLV — $V$):**
   $$V = \text{MonthlyCharges} \times H \times m$$
2. **Chi phí kỳ vọng ($C$):**
   $$C = c + u \cdot f$$
3. **Lợi ích bảo vệ kỳ vọng ($B$):**
   $$B = V \cdot p \cdot u \cdot e$$
4. **Lợi nhuận ròng kỳ vọng ($\Delta EV$):**
   $$\Delta EV = B - C = V \cdot p \cdot u \cdot e - (c + u \cdot f)$$
5. **Ngưỡng rủi ro hòa vốn ($p^*$):**
   $$p^* = \frac{c + u \cdot f}{V \cdot u \cdot e}$$

### 4.2. Ví dụ tính tay cụ thể (để trả lời nếu cô yêu cầu chứng minh công thức!)
Giả sử xét khách hàng đóng cước **$\text{MonthlyCharges} = \$100.85/\text{tháng}$**, xác suất rời mạng CatBoost dự báo là **$p = 84.6\%$ (`0.846`)**, với thiết lập mặc định **$H = 12$ tháng**, **$m = 50\%$ (`0.50`)**:

* **Bước 1 — Tính Giá trị bảo vệ ($V$):**
  $$V = 100.85 \times 12 \times 0.50 = \$605.10$$
* **Bước 2 — Tính phương án `A3` (Ưu đãi giảm cước cam kết dài hạn):**
  Thông số mặc định: $c = \$3.0$, $f = \$40.0$, $u = 40\%$ (`0.40`), $e = 50\%$ (`0.50`).
  * **Chi phí kỳ vọng ($C$):** $C = 3.0 + 0.40 \times 40.0 = \$19.00$
  * **Lợi ích kỳ vọng ($B$):** $B = 605.10 \times 0.846 \times 0.40 \times 0.50 \approx \$102.40$
  * **Lợi nhuận ròng ($\Delta EV$):** $\Delta EV = 102.40 - 19.00 = +\$83.40$
  * **Ngưỡng hòa vốn ($p^*$):** $p^* = \frac{19.00}{605.10 \times 0.40 \times 0.50} = \frac{19.00}{121.02} \approx 15.7\%$
  * **Kết luận:** Vì **$p = 84.6\% > p^* = 15.7\%$** nên gói `A3` mang lại kỳ vọng lời ròng **$\$83.40$**!

### 4.3. Tại sao Tab 3 chọn `A3` nhưng sang Tab 4 (khi giới hạn ngân sách) lại chọn `A2`?
* **Ở Tab 3 (Không giới hạn ngân sách, tối ưu từng người):** Hệ thống chọn gói có **$\Delta EV$ lớn nhất** (ví dụ `A3` lời `+$83.40`, lớn hơn `A2` lời `+$62.68`).
* **Ở Tab 4 (Giới hạn ngân sách $B$, bài toán Knapsack):** Mục tiêu là **tối đa hóa tổng $\Delta EV$ trên toàn bộ quỹ tiền có hạn**, nên thuật toán ưu tiên theo **hiệu suất $\Delta EV / C$** (tiền lời thu được trên mỗi $1 bỏ ra):
  * Với ví dụ trên: `A3` có $\Delta EV / C = 83.40 / 19.00 = 4.39\text{x}$.
  * Trong khi `A2` ($C = \$9.00$, $\Delta EV = \$62.68$) có hiệu suất tới $62.68 / 9.00 = 6.96\text{x}$!
  * Do đó khi ngân sách eo hẹp (như $\$15,000$), dùng `A2` giúp hệ thống chăm sóc được **nhiều khách hàng hơn** và tổng lợi nhuận toàn chiến dịch cao hơn!

---

## PHẦN 5: HẠ TẦNG TASK QUEUE, KHÓA CHỐNG TRÙNG & CHIẾN DỊCH BẤT BIẾN

1. **Task Queue & Idempotency Key (`data/task_queue.db`):**
   * Khi bấm **"Chạy & Ghi Nhận Scoring Job"**, hệ thống đẩy tác vụ chấm điểm vào hàng đợi SQLite và sinh `idempotency_key = scoring:<tên_file>:<số_dòng>:<model_version>`.
   * Nếu bấm lặp lại trên cùng file & model đã `SUCCESS`, khóa chống trùng ngăn ghi đè trùng lặp vào DB (nếu muốn ép chạy mới để demo, tích ô *"Tạo lượt chạy mới"*).
2. **Campaign Persistence (`IMMUTABLE ARCHIVE`):**
   * Khi chốt kịch bản ở Tab 4 và bấm **"Lưu Chiến Dịch Vào Hệ Thống"**, danh sách phân bổ và giả định tài chính được lưu vĩnh viễn để kiểm toán và tải lại file `.CSV` bất cứ lúc nào.

---

## PHẦN 6: BỘ 8 CÂU HỎI PHẢN BIỆN THƯỜNG GẶP CỦA GIẢNG VIÊN & CÁCH TRẢ LỜI

1. **Cô hỏi: *"Bộ dữ liệu Kaggle này cũ rồi (2017–2018), ngoài đời dữ liệu thay đổi theo tuần/tháng thì hệ thống chạy sao?"***
   * **Trả lời:** Dữ liệu thuê bao thật của nhà mạng là dữ liệu mật nên nhóm dùng bộ benchmark IBM Telco để kiểm chứng phương pháp. Bản chất đây là dữ liệu **ảnh chụp tĩnh (Snapshot)** tại một kỳ chốt cước. Ngoài thực tế, hệ thống đã tách rời **Model Bundle** và có sẵn tính năng **Nạp file CSV mới + Task Queue** ở thanh bên trái: cứ cuối tuần/tháng xuất snapshot mới nạp vào là chấm điểm hàng loạt ngay, và định kỳ retrain lại CatBoost để tránh trôi dữ liệu (Data Drift).
2. **Cô hỏi: *"Mô hình CatBoost của nhóm chính xác bao nhiêu phần trăm? Lấy số liệu đâu?"***
   * **Trả lời:** Đạt **ROC-AUC 84.8%** trên tập validation gồm `1,056` khách hàng (lưu trong `metadata.json` của bundle `catboost-20261018T130222-seed42`).
3. **Cô hỏi: *"Vì sao lại chia đúng 3 gói A1, A2, A3 chứ không phải gói khác?"***
   * **Trả lời:** Căn cứ trực tiếp từ **Tab 2 (EDA)**: Hợp đồng từng tháng rời mạng `42.7%` (gấp `15.1` lần hợp đồng 2 năm `2.8%`) → thiết kế **`A3`**; thanh toán thủ công `Electronic check` rời mạng `45.3%` → thiết kế **`A2`**; thiếu `TechSupport` rời mạng `41.6%` → thiết kế **`A1`**.
4. **Cô hỏi: *"Rủi ro cao nhưng giá trị cước thấp thì xử lý sao?"***
   * **Trả lời:** Hệ thống kiểm tra điều kiện hòa vốn **$p > p^*$** (tức $\Delta EV > 0$). Nếu $\Delta EV \le 0$, hệ thống tự xếp vào **`A0 — Không can thiệp`** để tránh lỗ vốn.
5. **Cô hỏi: *"Con số lợi nhuận ròng $\Delta EV$ có phải tiền lời thật 100% ngoài đời không?"***
   * **Trả lời:** Đây là **giá trị kỳ vọng mô phỏng (Expected Value)** dựa trên giả định biên lợi nhuận $m$, kỳ hạn $H$, tỷ lệ chấp nhận $u$ và mức giảm churn $e$. Do đó nhóm có trang bị các thanh trượt ở sidebar và biểu đồ **Độ nhạy biên lợi nhuận (`MARGIN SENSITIVITY`)** ở Tab 4 để doanh nghiệp thử nghiệm nhiều kịch bản khác nhau trước khi chạy chiến dịch thật.
6. **Cô hỏi: *"Biểu đồ Tab 2 và Tab 3 khác nhau chỗ nào?"***
   * **Trả lời:** Tab 2 là **Global Feature Importances** (xếp hạng tầm quan trọng chung của toàn mô hình CatBoost); Tab 3 là **Local TreeSHAP** (giải thích riêng từng khách hàng: cột nào kéo rủi ro tăng `+`, cột nào kéo giảm `−`) kèm **Thẻ Gợi Ý Hành Động CSKH**.
7. **Cô hỏi: *"Một khách hàng có thể nhận nhiều gói cùng lúc không?"***
   * **Trả lời:** Trong một đợt chiến dịch, mỗi khách hàng chỉ nhận **tối đa 1 gói tối ưu duy nhất** để tránh trùng lặp chi phí và làm phiền khách; toàn bộ các gói hợp lệ được đối chiếu trong bảng `Action Matrix` ở Tab 3.
8. **Cô hỏi: *"Nếu đội nhóm từng làm hướng LLM/RAG, tại sao bản thuyết trình lại chọn CatBoost + Kinh tế?"***
   * **Trả lời:** CatBoost cho tốc độ chấm hàng nghìn thuê bao trong vài giây, có chỉ số đo chuẩn (`ROC-AUC 84.8%`), không tốn phí API và tính TreeSHAP nhất quán 100%. Hướng LLM rất phù hợp làm lớp mở rộng ở tương lai để cá nhân hóa kịch bản thoại cho nhóm khách hàng VIP sau khi CatBoost + Knapsack đã lọc xong danh sách.
