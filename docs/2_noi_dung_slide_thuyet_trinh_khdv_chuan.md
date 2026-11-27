# NỘI DUNG CHI TIẾT 18 SLIDE THUYẾT TRÌNH MÔN KHOA HỌC DỊCH VỤ
*(Chuẩn 100% theo nhánh `demo-test-04-khdv` / `demo-05` — Chú trọng CLV, 4 gói A0–A3, TreeSHAP cho CSKH, Knapsack & Service Blueprint)*

> **Nguyên tắc làm bộ slide này:**
> - **Bỏ sạch** các chi tiết kỹ thuật rườm rà không khớp hoặc ghi bừa.
> - **Đặt trọng tâm đúng môn Khoa học Dịch vụ:** Dữ liệu hành vi (EDA) → Giá trị khách hàng (**Proxy CLV**) → Thiết kế 4 phương án **`A0, A1, A2, A3`** → Kinh tế học giữ chân ($\Delta EV$, $p^*$) → Tối ưu ngân sách (**Knapsack**) → Quy trình phục vụ (**Service Blueprint**) & **5 dịch vụ mở rộng**.
> - Số liệu trên slide **khớp từng con số** với Web demo trên `http://127.0.0.1:8505`.

---

## SLIDE 1: TRANG BÌA (TITLE SLIDE)
* **Tiêu đề chính:** HỆ THỐNG DỰ BÁO RỜI BỎ KHÁCH HÀNG & TỐI ƯU HÓA CHIẾN DỊCH GIỮ CHÂN DỰA TRÊN GIÁ TRỊ KINH TẾ
* **Phụ đề:** Ứng dụng CatBoost, TreeSHAP & Kinh tế học Dịch vụ (CLV – $\Delta EV$ – Knapsack) kèm Web Mô Phỏng
* **Thông tin:** Môn Khoa học Dịch vụ | Tên nhóm & Thành viên | Ngày thuyết trình.
* **Lời nói gợi ý:** *Chào cô và các bạn. Hôm nay nhóm trình bày đề tài xây dựng mô hình dự báo khách hàng rời mạng và tối ưu chiến dịch giữ chân dựa trên giá trị kinh tế, kèm ứng dụng Web mô phỏng cho bộ phận CSKH.*

---

## SLIDE 2: BÀI TOÁN THỰC TẾ & ĐỘNG LỰC NGHIÊN CỨU
* **Nội dung trên slide:**
  * **Thực trạng ngành Viễn thông:** Tỷ lệ rời mạng (Churn) cao gây thất thoát doanh thu định kỳ; chi phí tìm khách mới cao hơn nhiều lần chi phí giữ khách cũ.
  * **3 nút thắt lớn của doanh nghiệp dịch vụ:**
    1. *Dự báo hộp đen:* Biết khách sắp rời mạng nhưng CSKH không rõ nguyên nhân để tư vấn.
    2. *Lãng phí ngân sách:* Tặng ưu đãi tràn lan cho cả khách cước thấp hoặc rủi ro thấp → **lỗ ròng**.
    3. *Ngân sách hữu hạn:* Không thể chăm sóc cùng lúc hàng nghìn thuê bao — phải biết **giữ ai trước, tặng gói nào**.
* **Lời nói gợi ý:** *Nhiều đồ án chỉ dừng ở bước đoán "Ai sắp rời mạng?". Nhưng dưới góc độ Khoa học Dịch vụ, câu hỏi quan trọng hơn là: Có đáng bỏ tiền giữ họ không, giữ bằng gói dịch vụ nào, và với ngân sách có hạn thì ưu tiên ai trước?*

---

## SLIDE 3: MỤC TIÊU & KIẾN TRÚC TỔNG THỂ HỆ THỐNG
* **Nội dung trên slide (Sơ đồ 4 khối nối tiếp):**
  1. **Dữ liệu & Khám phá hành vi (EDA):** Tìm nguyên nhân gốc rễ từ 7.043 thuê bao.
  2. **Dự báo & Giải thích AI (`CatBoost` + `TreeSHAP`):** Chấm xác suất rời mạng ($p$) hàng loạt & chỉ rõ yếu tố tác động từng cá nhân.
  3. **Kinh tế học Giữ chân (`CLV`, `A0–A3`, $\Delta EV$, $p^*$):** Lọc ra khách hàng có lợi nhuận ròng dương.
  4. **Tối ưu Ngân sách & Bàn giao CSKH (`Knapsack` + `.CSV`):** Chia ngân sách giới hạn & xuất danh sách gọi điện cho tuyến đầu.
* **Lời nói gợi ý:** *Hệ thống của nhóm khép kín theo 4 bước liên hoàn, gắn chặt từ phân tích dữ liệu hậu trường đến hành động chăm sóc khách hàng tuyến đầu.*

---

## SLIDE 4: DỮ LIỆU ĐẦU VÀO & BÀI TOÁN DỮ LIỆU THEO KỲ
* **Nội dung trên slide:**
  * **Bộ dữ liệu chuẩn (Benchmark):** IBM Telco Customer Churn (`7.043` thuê bao, `21` thuộc tính).
    * *Nhân khẩu học (5 cột):* `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    * *Dịch vụ & Gắn bó (10 cột):* `tenure`, `InternetService`, `TechSupport`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `StreamingTV/Movies`, `PhoneService`, `MultipleLines`.
    * *Hợp đồng & Cước phí (6 cột):* `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`.
  * **Thực trạng gốc tập dữ liệu:** Tỷ lệ Churn toàn mạng **`26.5%`** (`1.869` KH) | Cước TB **`$64.8/tháng`** | Gắn bó TB **`32.4` tháng**.
  * **Vận hành thực tế theo chu kỳ:** Tập Kaggle là ảnh chụp tĩnh (*Snapshot*) làm benchmark; trên Web nhóm thiết kế sẵn **nạp file CSV mới theo tuần/tháng** để chấm điểm định kỳ.
* **Gợi ý hình ảnh:** Chụp 4 thẻ KPI đầu **Tab 2**.
* **Lời nói gợi ý:** *Vì dữ liệu thuê bao thật của nhà mạng là dữ liệu mật, nhóm dùng bộ benchmark IBM Telco để kiểm chứng mô hình, đồng thời thiết kế chức năng nạp file CSV mới theo kỳ để sẵn sàng chạy định kỳ ngoài thực tế.*

---

## SLIDE 5: KHÁM PHÁ HÀNH VI RỜI MẠNG (EDA) — CĂN CỨ THIẾT KẾ DỊCH VỤ
* **Nội dung trên slide (4 phát hiện chính từ Tab 2):**
  * **Loại hợp đồng (`Contract`):** Hợp đồng từng tháng (`Month-to-month`) rời mạng **`42.7%`**, cao gấp **`15.1` lần** hợp đồng 2 năm (**`2.8%`**) → **Căn cứ tạo gói `A3`**.
  * **Phương thức thanh toán (`PaymentMethod`):** Thanh toán thủ công `Electronic check` rời mạng **`45.3%`**, cao vượt trội thanh toán tự động → **Căn cứ tạo gói `A2`**.
  * **Hỗ trợ kỹ thuật (`TechSupport`):** Khách chưa đăng ký hỗ trợ kỹ thuật rời mạng **`41.6%`** → **Căn cứ tạo gói `A1`**.
  * **Đường truyền (`InternetService`):** Nhóm Cáp quang (`Fiber optic`) rời mạng **`41.9%`**, gấp **`2.2` lần** DSL (**`19.0%`**) → Kỳ vọng cao, nhạy cảm với chất lượng.
* **Gợi ý hình ảnh:** Chụp 4 biểu đồ cột ở **Tab 2**.
* **Lời nói gợi ý:** *Các gói dịch vụ giữ chân của nhóm không phải nghĩ ngẫu nhiên, mà xuất phát trực tiếp từ 4 điểm đau hành vi rõ rệt nhất trên dữ liệu.*

---

## SLIDE 6: MÔ HÌNH DỰ BÁO RỦI RO (`CATBOOST`) & ĐÁNH GIÁ HIỆU NĂNG
* **Nội dung trên slide:**
  * **Lý do chọn CatBoost:** Xử lý trực tiếp 15 cột phân loại (Categorical) trong dữ liệu viễn thông, chấm hàng loạt **7.043 khách chỉ trong ~1 giây**, chạy ổn định không phụ thuộc API ngoài.
  * **Kết quả kiểm định trên tập Validation (`1.056` khách hàng, train `4.930` mẫu):**
    * **ROC-AUC:** **`84.8%`** (`0.8476`) — xếp đúng thứ hạng rủi ro ở ~85/100 cặp khách rời bỏ vs ở lại.
    * **Average Precision (PR-AUC):** **`0.654`**
    * **Brier Score:** **`0.134`** — xác suất $p$ có độ tin cậy tốt để đưa vào công thức tài chính.
  * **Top đặc trưng quan trọng toàn cục (`Feature Importances`):** `Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `TechSupport`...
* **Gợi ý hình ảnh:** Chụp biểu đồ `Top 10 Đặc Trưng Ảnh Hưởng Nhất` ở **Tab 2**.

---

## SLIDE 7: ĐỊNH GIÁ KHÁCH HÀNG — GIÁ TRỊ VÒNG ĐỜI XẤP XỈ (PROXY CLV)
*(Slide trọng tâm theo đúng yêu cầu của bạn)*
* **Nội dung trên slide:**
  * **Vì sao cần CLV trong Khoa học Dịch vụ?** Hai khách hàng cùng có rủi ro rời mạng `80%`, nhưng khách trả `$100/tháng` có giá trị kinh tế khác hẳn khách trả `$20/tháng`.
  * **Công thức Proxy CLV ($V$) — Giá trị bảo vệ trong kỳ hạn $H$ tháng:**
    $$V = \text{MonthlyCharges} \times H \times m$$
    * $\text{MonthlyCharges}$: Cước phí hàng tháng của từng thuê bao (USD).
    * $H$: Kỳ hạn đánh giá giữ chân (mặc định **`12 tháng`**, tùy chỉnh 6 / 12 / 24 tháng trên Web).
    * $m$: Biên lợi nhuận gộp giả định (mặc định **`50%`**, điều chỉnh được từ 10% – 70%).
  * **Ví dụ nhanh:** Khách đóng `$100.85/tháng`, $H = 12$, $m = 50\%$ $\Rightarrow$ Giá trị bảo vệ **$V = \$605.10$**.
* **Lời nói gợi ý:** *Chúng em gọi đây là Proxy CLV — giá trị đóng góp kỳ vọng trong H tháng thay vì phóng đại thành CLV trọn đời, giúp việc định lượng chi phí và lợi nhuận giữ chân thực tế và chặt chẽ hơn.*

---

## SLIDE 8: THIẾT KẾ CHI TIẾT 4 PHƯƠNG ÁN CAN THIỆP (`A0`, `A1`, `A2`, `A3`)
*(Slide trọng tâm — dành hẳn 1 slide rõ ràng cho 4 gói!)*
* **Nội dung trên slide (Bảng chi tiết):**

| Mã | Tên phương án | Điều kiện áp dụng | Chi phí liên hệ ($c$) | Chi phí ưu đãi ($f$) | Tỷ lệ nhận ($u$) | Giảm churn ($e$) | Chi phí kỳ vọng ($C = c + u \cdot f$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | **Không can thiệp** | Mặc định / khi $\Delta EV \le 0$ | $\$0$ | $\$0$ | — | — | **$\$0.00$** |
| **`A1`** | **Hỗ trợ kỹ thuật & bảo mật chủ động** | Có Internet, thiếu `TechSupport` hoặc `OnlineSecurity` | $\$2.0$ | $\$15.0$ | $40\%$ | $25\%$ | **$\$8.00$** |
| **`A2`** | **Tư vấn hợp đồng & thanh toán tự động** | Hợp đồng tháng hoặc thanh toán thủ công | $\$2.0$ | $\$20.0$ | $35\%$ | $40\%$ | **$\$9.00$** |
| **`A3`** | **Ưu đãi giảm cước cam kết dài hạn** | Hợp đồng `Month-to-month` | $\$3.0$ | $\$40.0$ | $40\%$ | $50\%$ | **$\$19.00$** |

* **Nguyên tắc nghiệp vụ:** Mỗi khách hàng chỉ nhận **tối đa 1 gói duy nhất** trong một đợt chiến dịch để tránh gọi trùng và chồng chéo chi phí.

---

## SLIDE 9: MÔ HÌNH KINH TẾ HỌC GIỮ CHÂN ($\Delta EV$ & NGƯỠNG HÒA VỐN $p^*$)
* **Nội dung trên slide:**
  * **Chi phí kỳ vọng:** $C = c + u \cdot f$ *(chỉ mất tiền ưu đãi $f$ khi khách đồng ý nhận với tỷ lệ $u$)*.
  * **Lợi ích kỳ vọng bảo vệ được:** $B = V \cdot p \cdot u \cdot e$
  * **Lợi nhuận ròng kỳ vọng ($\Delta EV$):**
    $$\Delta EV = B - C = V \cdot p \cdot u \cdot e - (c + u \cdot f)$$
  * **Ngưỡng rủi ro hòa vốn ($p^*$):**
    $$p^* = \frac{c + u \cdot f}{V \cdot u \cdot e}$$
  * **Quy tắc ra quyết định:**
    * Nếu **$p > p^*$** ($\Delta EV > 0$): Chọn phương án hợp lệ có **$\Delta EV$ cao nhất** → gắn nhãn **`TỐI ƯU`**.
    * Nếu mọi gói đều **$p \le p^*$** ($\Delta EV \le 0$): Tự động chọn **`A0 — Không can thiệp`** để bảo toàn ngân sách.

---

## SLIDE 10: MINH HỌA TÍNH TOÁN THỰC TẾ TRÊN 1 KHÁCH HÀNG (`4132-KALRO`)
* **Nội dung trên slide:**
  * **Hồ sơ `4132-KALRO`:** Cước **`$100.85/tháng`**, mới dùng **`4 tháng`**, hợp đồng **`Month-to-month`**, **`Fiber optic`**, rủi ro CatBoost **$p = 84.6\%$**.
  * **Giá trị bảo vệ ($H=12, m=50\%$):** $V = 100.85 \times 12 \times 0.50 = \mathbf{\$605.10}$.
  * **So sánh 4 phương án hợp lệ (`Action Matrix`):**
    * `A0`: $C = \$0$, $\Delta EV = \$0.00$
    * `A1`: $C = \$8.00$, $B = \$51.20$, $\Delta EV = +\$43.20$ ($p^* = 13.2\%$)
    * `A2`: $C = \$9.00$, $B = \$71.68$, $\Delta EV = +\$62.68$ ($p^* = 10.6\%$)
    * **`A3` (`TỐI ƯU`)**: $C = \$19.00$, $B = \$102.40$, **$\Delta EV = +\$83.40$** ($p^* = 15.7\%$) → Rủi ro giảm từ **`84.6%`** xuống **`67.7%`** (`-16.9%`).
* **Gợi ý hình ảnh:** Chụp phần `Action Matrix` + `Decision Ticket` ở **Tab 3**.

---

## SLIDE 11: GIẢI THÍCH CÁ NHÂN (`TreeSHAP`) & THẺ HÀNH ĐỘNG CSKH
* **Nội dung trên slide:**
  * **Giải quyết bài toán "Hộp đen AI" cho nhân viên tuyến đầu:**
    * Biểu đồ **`Yếu Tố Tác Động Rủi Ro Đến Cá Nhân` (`TreeSHAP`)**:
      * *Thanh đỏ (+):* Yếu tố làm tăng nguy cơ (`tenure = 4 tháng`, `MonthlyCharges = $100.85`, `Month-to-month`, `Fiber optic`).
      * *Thanh xanh (−):* Yếu tố giúp giảm nguy cơ (`TechSupport = Yes`).
    * **`Thẻ Gợi Ý Hành Động Cho Nhân Viên CSKH`** (3 ô ngắn gọn dễ đọc):
      1. *Nguyên nhân rủi ro chính* (diễn giải tiếng Việt, bỏ ký hiệu kỹ thuật).
      2. *Đề xuất gói & Kênh liên hệ ưu tiên* (`Tổng đài CSKH Outbound 1-1` cho khách rủi ro cao $\ge 65\%$, `Zalo OA / App` cho nhóm còn lại).
      3. *Gợi ý chăm sóc mở rộng* phù hợp hồ sơ.
* **Gợi ý hình ảnh:** Chụp 3 cột Hồ sơ + `Thẻ Gợi Ý Hành Động Cho Nhân Viên CSKH` ở **Tab 3**.

---

## SLIDE 12: TỔNG QUAN KINH TẾ TOÀN MẠNG (KỊCH BẢN MẶC ĐỊNH)
* **Nội dung trên slide (Số liệu thật trên Tab 1 với $m=50\%, H=12$):**
  * **Khách hàng đang hoạt động (Active):** **`5.163`** thuê bao (trong `7.043` bản ghi).
  * **Đủ điều kiện giữ chân ($\Delta EV > 0$):** **`1.756`** khách hàng (**`34.0%`**) — **66% còn lại chọn `A0` để không lãng phí tiền**.
  * **Tổng chi phí can thiệp:** **`$23.184`** (`$13.2/KH`).
  * **Tổng lợi nhuận ròng kỳ vọng ($\Delta EV$):** **`+$31.999`** (**`ROI +138.0%`**).
* **Gợi ý hình ảnh:** Chụp 4 thẻ KPI + biểu đồ *Cân Đối Dòng Tiền* ở **Tab 1**.

---

## SLIDE 13: TỐI ƯU NGÂN SÁCH HỮU HẠN (`KNAPSACK`) & PHÂN TÍCH ĐỘ NHẠY
* **Nội dung trên slide:**
  * **Bài toán thực tế:** Nếu công ty chỉ cấp ngân sách **`$15.000`** (thấp hơn mức `$23.184` cần cho toàn bộ `1.756` khách)?
  * **Giải pháp Knapsack (Xếp hạng theo hiệu suất $\Delta EV / C$):**
    * Ưu tiên gói mang lại nhiều tiền lời nhất trên mỗi `$1` bỏ ra (ví dụ `A2` có hiệu suất `6.96x` cao hơn `A3` `4.39x` → hệ thống linh hoạt chọn `A2` để giữ được nhiều khách hơn).
    * **Kết quả với ngân sách `$15.000`:** Phục vụ **`1.679`** khách hàng, mang về lợi nhuận ròng **`+$27.503`** (**`ROI +183.4%`**).
  * **Phân tích Độ nhạy (`MARGIN SENSITIVITY` & `ACTION MIX`):**
    * Cho thấy khi biên lợi nhuận ($m$) biến động từ `20%` → `50%`, lợi nhuận ròng và cơ cấu các gói `A1 / A2 / A3` thay đổi ra sao → giúp quản lý chủ động ứng phó rủi ro tài chính.
* **Gợi ý hình ảnh:** Chụp thanh ngân sách `$15,000` + 2 biểu đồ `MARGIN SENSITIVITY` & `ACTION MIX` ở **Tab 4**.

---

## SLIDE 14: QUY TRÌNH PHỤC VỤ 4 LỚP (SERVICE BLUEPRINT)
* **Nội dung trên slide (Sơ đồ 4 tầng chuẩn môn KHDV):**
  1. **Customer Actions (Khách hàng):** Sử dụng dịch vụ Internet/ĐT, đóng cước, gặp sự cố hoặc cân nhắc hủy hợp đồng.
  2. **Frontstage (Nhân viên CSKH tuyến đầu):** Nhận danh sách `.CSV` & Thẻ gợi ý CSKH → Gọi Outbound 1-1 / nhắn Zalo OA tư vấn đúng gói `A1/A2/A3`.
  3. **Backstage (Chuyên viên vận hành):** Dùng Web Dashboard 4 Tab để nạp CSV theo kỳ, theo dõi hành vi, duyệt kịch bản ngân sách & xuất chiến dịch.
  4. **Support Processes (Hạ tầng chạy ngầm):** `CatBoost Bundle`, `TreeSHAP`, bộ tính `CLV/Knapsack`, cơ sở dữ liệu lưu vết chiến dịch & lịch sử chạy định kỳ (`SQLite`).

---

## SLIDE 15: ĐỀ XUẤT 5 HƯỚNG DỊCH VỤ CHĂM SÓC MỞ RỘNG (SERVICE DESIGN)
* **Nội dung trên slide:**
  * Khai thác sâu 9 cột dữ liệu còn lại trong hồ sơ để thiết kế dịch vụ cá nhân hóa (tích hợp gợi ý ngay tại Tab 3):
    1. **Đồng hành thuê bao mới (`tenure`):** Kiểm tra đường truyền từ xa & hướng dẫn App Wi-Fi kỳ đầu.
    2. **Hỗ trợ thân thiện người cao tuổi (`SeniorCitizen`):** Ưu tiên gặp trực tiếp tổng đài viên, tài liệu dễ đọc (*Inclusive Design*).
    3. **Gói Gia đình & Đa thiết bị (`Partner`, `Dependents`, `MultipleLines`):** Gộp hóa đơn hộ gia đình tăng gắn kết hệ sinh thái.
    4. **Sao lưu đám mây & Bảo vệ thiết bị (`OnlineBackup`, `DeviceProtection`):** Chuyển từ bán đường truyền sang giải pháp an tâm.
    5. **Giải trí tích hợp trên Cáp quang (`StreamingTV/Movies`, `Fiber optic`):** Đồng sáng tạo giá trị (*Value Co-creation*) thay vì giảm giá trực tiếp.

---

## SLIDE 16: DEMO TRỰC TIẾP TRÊN WEB MÔ PHỎNG (`http://127.0.0.1:8505`)
* **Kịch bản demo mượt trong 3 phút:**
  1. **Tab 1:** Chỉ nhanh 4 KPI tài chính (`+$31,999`, `ROI +138%`) và kéo thử biên lợi nhuận `m` ở sidebar để thấy số liệu nhảy động.
  2. **Tab 2:** Lướt qua 4 biểu đồ EDA chứng minh cơ sở `A1/A2/A3` và Top 10 đặc trưng CatBoost.
  3. **Tab 3:** Chọn khách `4132-KALRO`, chỉ đồng hồ `84.6% → 67.7%`, biểu đồ `TreeSHAP` và `Thẻ Gợi Ý Hành Động CSKH`.
  4. **Tab 4:** Kéo ngân sách `$15,000`, xem danh sách ưu tiên + biểu đồ độ nhạy `20%–50%` + nút xuất file `.CSV` cho CSKH.
  5. *(Điểm cộng)* Mở sidebar nạp thử file `demo_telco_test_1000.csv` (1.000 dòng) để chứng minh hệ thống xử lý dữ liệu mới theo kỳ trong vài giây.

---

## SLIDE 17: HẠN CHẾ HIỆN TẠI & HƯỚNG PHÁT TRIỂN
* **Nội dung trên slide:**
  * **Hạn chế trung thực:**
    * Bộ dữ liệu IBM Telco là ảnh chụp tĩnh benchmark; khi đưa vào nhà mạng thật cần huấn luyện lại trên dữ liệu 3–6 tháng gần nhất để tránh trôi dữ liệu (*Data Drift*).
    * Các tham số $u$ (tỷ lệ chấp nhận) và $e$ (hiệu lực giảm churn) đang là giả định mô phỏng — thực tế cần đo bằng thử nghiệm **A/B Testing / Uplift Modeling**.
  * **Hướng phát triển tương lai:**
    * Tích hợp **A/B Testing** đo hiệu quả can thiệp thực tế ngoài thị trường.
    * Kết nối **LLM** ở lớp hậu xử lý để tự động soạn email/kịch bản trò chuyện cá nhân hóa riêng cho nhóm khách hàng VIP sau khi CatBoost + Knapsack đã lọc xong danh sách.

---

## SLIDE 18: KẾT LUẬN & Q&A
* **Nội dung trên slide:**
  * **3 đóng góp cốt lõi của đồ án:**
    1. Kết hợp **Dự báo (`CatBoost 84.8%`) + Giải thích (`TreeSHAP`)** tốc độ cao cho hàng nghìn khách hàng.
    2. Chuẩn hóa quyết định giữ chân bằng **Kinh tế học (`Proxy CLV`, `A0–A3`, $\Delta EV$, $p^*$, `Knapsack`)**.
    3. Kết nối trọn vẹn góc nhìn **Khoa học Dịch vụ** (*Service Blueprint*, *Thẻ hành động CSKH* & *5 dịch vụ mở rộng*).
  * **Cảm ơn Cô và các bạn đã lắng nghe! — Mời đặt câu hỏi (Q&A).**
