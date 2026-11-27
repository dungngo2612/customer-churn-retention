# LỜI DẪN THUYẾT TRÌNH CHI TIẾT 18 SLIDE — MÔN KHOA HỌC DỊCH VỤ
*(Chuẩn nhánh `demo-test-04-khdv` / `demo-05` — Văn phong nói tự nhiên, mạch lạc, có câu chuyển slide mượt mà)*

> **Gợi ý tốc độ trình bày:** Tổng thời gian lý tưởng **10 – 12 phút** (trong đó phần Slide khoảng 7–8 phút, Demo Web trực tiếp 3–4 phút).

---

### SLIDE 1 — Trang bìa
> "Em xin chào cô và các bạn. Hôm nay nhóm em xin phép trình bày đề tài môn Khoa học Dịch vụ: **Xây dựng hệ thống dự báo khách hàng có khả năng rời bỏ và tối ưu hóa chiến dịch giữ chân dựa trên giá trị kinh tế**, kèm ứng dụng Web mô phỏng hỗ trợ ra quyết định cho bộ phận chăm sóc khách hàng."

---

### SLIDE 2 — Bài toán thực tế & Động lực nghiên cứu
> "Trong ngành dịch vụ thuê bao định kỳ như viễn thông hay Internet, khách hàng rời mạng (**Churn**) gây thất thoát doanh thu liên tục, trong khi chi phí tìm một khách hàng mới tốn kém hơn rất nhiều so với giữ một khách hàng cũ.
>
> Tuy nhiên, khi đi sâu vào thực tế vận hành, nhiều hệ thống hiện nay vướng 3 nút thắt:
> - Thứ nhất là **dự báo hộp đen**: chỉ ra được con số rủi ro nhưng nhân viên CSKH không biết khách đang gặp vấn đề ở đâu để tư vấn.
> - Thứ hai là **lãng phí ngân sách**: nếu cứ thấy rủi ro cao là tặng ưu đãi, thì với những khách cước thấp hoặc rủi ro chưa đủ bù chi phí, công ty sẽ bị lỗ ròng.
> - Thứ ba là **giới hạn ngân sách**: mỗi tháng bộ phận CSKH chỉ có một quỹ tiền nhất định, không thể gọi hết cho hàng nghìn người. Vì vậy, bài toán đặt ra là phải biết **giữ ai trước, tặng gói nào để có lời cao nhất**."

---

### SLIDE 3 — Mục tiêu & Kiến trúc tổng thể hệ thống
> "Để giải quyết 3 nút thắt đó, nhóm xây dựng hệ thống khép kín gồm 4 bước liên hoàn như trên sơ đồ:
> 1. **Bước 1:** Khám phá hành vi khách hàng từ dữ liệu (EDA) để tìm đúng nguyên nhân gốc rễ.
> 2. **Bước 2:** Dùng mô hình **CatBoost** chấm điểm rủi ro hàng loạt và **TreeSHAP** giải thích yếu tố tác động cho từng khách hàng.
> 3. **Bước 3:** Đưa qua **Mô hình Kinh tế Giữ chân** với 4 phương án `A0–A3`, tính **Proxy CLV**, lợi nhuận ròng **$\Delta EV$** và ngưỡng hòa vốn **$p^*$**.
> 4. **Bước 4:** Dùng thuật toán **Knapsack** tối ưu phân bổ ngân sách giới hạn và xuất file `.CSV` bàn giao thẳng cho nhân viên CSKH."

---

### SLIDE 4 — Dữ liệu đầu vào & Bài toán dữ liệu theo kỳ
> "Về dữ liệu, nhóm sử dụng bộ dữ liệu benchmark kinh điển của IBM là **Telco Customer Churn** gồm **7.043 thuê bao** và **21 thuộc tính**, chia làm 3 nhóm: nhân khẩu học, dịch vụ đang dùng, và hợp đồng – cước phí. Trên tập này, tỷ lệ rời mạng trung bình là **26,5%** (tương ứng **1.869** khách hàng), cước trung bình **64,8 USD/tháng** và thời gian gắn bó trung bình **32,4 tháng**.
>
> *(Chủ động đón đầu câu hỏi của cô về dữ liệu cũ/mới)*: Nhóm cũng hiểu rằng ngoài doanh nghiệp thật, hành vi người dùng thay đổi liên tục nên dữ liệu cần lấy theo kỳ gần nhất (tuần hoặc tháng). Vì dữ liệu thuê bao thật của nhà mạng là dữ liệu mật, nhóm dùng tập IBM này làm bản ảnh chụp tĩnh (*Snapshot*) để kiểm chứng phương pháp; đồng thời trên Web đã thiết kế sẵn chức năng **nạp tệp CSV mới theo chu kỳ** để hệ thống chạy được với dữ liệu mới bất cứ lúc nào."

---

### SLIDE 5 — Khám phá hành vi rời mạng (EDA) — Căn cứ thiết kế dịch vụ
> "Trước khi xây dựng gói chăm sóc, nhóm phân tích dữ liệu lịch sử để trả lời câu hỏi: *Khách hàng rời mạng vì những lý do gì?* Trên màn hình là 4 quy luật rõ nhất:
> - **Loại hợp đồng:** Nhóm hợp đồng từng tháng (`Month-to-month`) có tỷ lệ rời mạng lên tới **42,7%**, cao gấp **15,1 lần** so với nhóm hợp đồng 2 năm (**2,8%**). Đây chính là cơ sở để nhóm thiết kế gói ưu đãi cam kết dài hạn **`A3`**.
> - **Phương thức thanh toán:** Nhóm trả thủ công bằng `Electronic check` rời mạng tới **45,3%**, cao hơn hẳn thanh toán tự động — là cơ sở thiết kế gói chuyển đổi hợp đồng và thanh toán tự động **`A2`**.
> - **Hỗ trợ kỹ thuật:** Nhóm thiếu `TechSupport` rời mạng **41,6%** — là cơ sở thiết kế gói hỗ trợ kỹ thuật chủ động **`A1`**.
> - **Đường truyền Internet:** Nhóm cáp quang (`Fiber optic`) có tỷ lệ rời mạng **41,9%**, gấp **2,2 lần** DSL (**19,0%**), cho thấy nhóm trả cước cao kỳ vọng rất lớn vào chất lượng ổn định."

---

### SLIDE 6 — Mô hình dự báo rủi ro (`CatBoost`) & Đánh giá hiệu năng
> "Để dự báo xác suất rời mạng ($p$) trên dữ liệu bảng có nhiều biến phân loại như ngành viễn thông, nhóm lựa chọn mô hình **CatBoost**. Ưu điểm lớn nhất là xử lý trực tiếp biến phân loại rất tốt, chấm hàng loạt **7.043 khách hàng chỉ trong khoảng 1 giây** và chạy ổn định hoàn toàn offline.
>
> Trên tập kiểm định (Validation gồm **1.056** hồ sơ), mô hình đạt **ROC-AUC = 84,8%** (`0.8476`), **PR-AUC = 0,654** và **Brier Score = 0,134** — nghĩa là xác suất dự báo có độ tin cậy tốt để đưa vào tính toán kinh tế phía sau. Biểu đồ bên phải cũng cho thấy các đặc trưng mô hình coi trọng nhất như `Contract`, `tenure`, `MonthlyCharges` hoàn toàn khớp với phân tích EDA vừa rồi."

---

### SLIDE 7 — Định giá khách hàng: Proxy CLV ($V$)
> "Sau khi có xác suất rời mạng $p$, câu hỏi tiếp theo của môn Khoa học Dịch vụ là: **Khách hàng này mang lại bao nhiêu giá trị cho công ty?**
>
> Nhóm sử dụng chỉ số **Proxy CLV ($V$)** — tức giá trị lợi nhuận gộp kỳ vọng của khách hàng trong kỳ hạn đánh giá $H$ tháng, tính theo công thức:
> $$V = \text{MonthlyCharges} \times H \times m$$
> Với kỳ hạn mặc định **$H = 12$ tháng** và biên lợi nhuận **$m = 50\%$**, một khách hàng đóng cước **$100,85/tháng** sẽ có giá trị bảo vệ **$V = \$605,10$**. Cách tính trong $H$ tháng này giúp mô hình thực tế và thận trọng hơn so với việc giả định giá trị trọn đời quá cao."

---

### SLIDE 8 — Thiết kế chi tiết 4 phương án can thiệp (`A0`, `A1`, `A2`, `A3`)
> "Từ căn cứ hành vi ở Slide 5, nhóm chuẩn hóa 4 phương án hành động cho bộ phận CSKH:
> - **`A0 — Không can thiệp`:** Chi phí $\$0$, áp dụng cho khách rủi ro thấp hoặc can thiệp bị lỗ.
> - **`A1 — Hỗ trợ kỹ thuật & bảo mật chủ động`:** Dành cho khách dùng Internet nhưng chưa có `TechSupport` hoặc `OnlineSecurity`, chi phí kỳ vọng **$\$8,00$**, mức giảm churn kỳ vọng $25\%$.
> - **`A2 — Tư vấn hợp đồng & thanh toán tự động`:** Dành cho khách hợp đồng tháng hoặc trả thủ công, chi phí kỳ vọng **$\$9,00$**, mức giảm churn $40\%$.
> - **`A3 — Ưu đãi giảm cước cam kết dài hạn`:** Dành cho khách hợp đồng từng tháng, chi phí kỳ vọng **$\$19,00$**, mức giảm churn $50\%$.
>
> Trong mỗi đợt chiến dịch, một khách hàng chỉ nhận **tối đa 1 gói tối ưu nhất** để tránh gọi điện trùng lặp và đội chi phí."

---

### SLIDE 9 — Mô hình Kinh tế học Giữ chân ($\Delta EV$ & Ngưỡng hòa vốn $p^*$)
> "Để quyết định gói nào thực sự có lời, nhóm tính 3 chỉ số cốt lõi cho từng cặp khách hàng – gói can thiệp:
> - **Chi phí kỳ vọng ($C = c + u \cdot f$):** Chi phí liên hệ $c$ mất ngay khi gọi, còn tiền ưu đãi $f$ chỉ mất theo tỷ lệ khách đồng ý nhận $u$.
> - **Lợi ích kỳ vọng ($B = V \cdot p \cdot u \cdot e$)** và **Lợi nhuận ròng ($\Delta EV = B - C$)**.
> - **Ngưỡng rủi ro hòa vốn ($p^*$):** Là mức rủi ro tối thiểu để tiền cứu được bằng tiền bỏ ra.
>
> Quy tắc rất minh bạch: chỉ khi **$p > p^*$** (tức **$\Delta EV > 0$**) thì mới can thiệp và chọn gói có $\Delta EV$ cao nhất; ngược lại sẽ tự động giữ nguyên **`A0`** để bảo vệ ngân sách."

---

### SLIDE 10 — Minh họa tính toán thực tế trên 1 khách hàng (`4132-KALRO`)
> "Để cô và các bạn dễ hình dung, đây là ví dụ thực tế trên khách hàng mã **`4132-KALRO`**: đóng cước **$100,85/tháng**, mới gắn bó 4 tháng, hợp đồng tháng, dùng cáp quang, xác suất rời mạng **$p = 84,6\%$**.
>
> Với giá trị bảo vệ $V = \$605,10$, khi chạy qua cả 4 phương án:
> - Gói `A1` lời ròng `+$43,20`; gói `A2` lời ròng `+$62,68`.
> - Riêng gói **`A3`** có chi phí kỳ vọng `$19,00`, lợi ích kỳ vọng `$102,40`, cho **lợi nhuận ròng cao nhất là `+$83,40`** (vượt xa ngưỡng hòa vốn $p^* = 15,7\%$). Vì vậy hệ thống gắn nhãn **`TỐI ƯU`** cho gói `A3`, kéo rủi ro rời mạng giảm từ **84,6% xuống 67,7%**."

---

### SLIDE 11 — Giải thích cá nhân (`TreeSHAP`) & Thẻ hành động CSKH
> "Nhưng làm sao nhân viên tổng đài biết phải nói gì khi gọi cho khách hàng `4132-KALRO` này?
>
> Ở Tab 3, nhóm tích hợp biểu đồ **TreeSHAP** cho từng cá nhân: các thanh đỏ chỉ rõ 4 nguyên nhân đang đẩy rủi ro tăng cao (*mới dùng 4 tháng, cước $100,85, hợp đồng tháng, cáp quang*), còn thanh xanh cho thấy yếu tố đang giúp giữ chân (*đã có TechSupport*).
>
> Để nhân viên CSKH không phải đọc thuật ngữ kỹ thuật, hệ thống tự tổng hợp thành **Thẻ Gợi Ý Hành Động CSKH** ngay bên dưới gồm đúng 3 phần: *Nguyên nhân chính — Gói đề xuất & Kênh liên hệ ưu tiên (Tổng đài Outbound 1-1 cho khách rủi ro cao $\ge 65\%$) — và Gợi ý chăm sóc mở rộng*."

---

### SLIDE 12 — Tổng quan kinh tế toàn mạng (Kịch bản mặc định)
> "Khi áp dụng đồng loạt mô hình kinh tế lên toàn bộ **5.163 khách hàng đang hoạt động** (ở kịch bản mặc định $m = 50\%, H = 12$ tháng):
> - Hệ thống chỉ chọn ra **1.756 khách hàng** (**34,0%**) thực sự có $\Delta EV > 0$ để can thiệp; **66% còn lại** được xếp vào `A0` để không đốt tiền vô ích.
> - Tổng chi phí bỏ ra là **23.184 USD** (trung bình `$13,2/khách`), nhưng giá trị lợi nhuận ròng kỳ vọng thu về lên tới **+31.999 USD**, tương ứng tỷ suất sinh lời **ROI = +138,0%**."

---

### SLIDE 13 — Tối ưu ngân sách hữu hạn (`Knapsack`) & Phân tích độ nhạy
> "Trong thực tế, nếu giám đốc chỉ duyệt ngân sách **15.000 USD** thay vì `23.184 USD`, hệ thống sẽ xử lý thế nào?
>
> Nhóm giải bài toán **Knapsack** bằng cách xếp hạng theo **tỷ suất lợi nhuận trên mỗi đô-la chi phí ($\Delta EV / C$)**. Điểm thú vị ở đây là: khi thiếu ngân sách, hệ thống có thể linh hoạt chuyển từ gói đắt (`A3`) sang gói tiết kiệm hiệu suất cao hơn (như `A2` với hiệu suất `6,96x`) để giữ được **1.679 khách hàng**, thu về **+27.503 USD** lợi nhuận ròng với **ROI tăng lên +183,4%**.
>
> Bên cạnh đó, Tab 4 còn có biểu đồ **Độ nhạy biên lợi nhuận (`20% – 50%`)** giúp quản lý dự trù trước kịch bản nếu biên lợi nhuận thay đổi."

---

### SLIDE 14 — Quy trình phục vụ 4 lớp (Service Blueprint)
> "Đặt toàn bộ giải pháp vào khung **Service Blueprint** của môn Khoa học Dịch vụ, hệ thống vận hành rõ ràng qua 4 lớp:
> - **Lớp Khách hàng:** Sinh hoạt tiêu dùng viễn thông hàng ngày.
> - **Lớp Tuyến đầu (Frontstage):** Nhân viên CSKH nhận file `.CSV` và Thẻ gợi ý để gọi điện Outbound 1-1 hoặc nhắn Zalo OA.
> - **Lớp Hậu trường (Backstage):** Chuyên viên vận hành sử dụng Web 4 Tab để nạp dữ liệu kỳ mới, duyệt chiến dịch và xuất danh sách.
> - **Lớp Hạ tầng hỗ trợ:** Bộ lõi CatBoost, TreeSHAP, mô-đun CLV/Knapsack và cơ sở dữ liệu lưu vết."

---

### SLIDE 15 — Đề xuất 5 hướng dịch vụ chăm sóc mở rộng (Service Design)
> "Ngoài 4 gói `A0–A3` đang khai thác 5 cột dữ liệu chính, nhóm còn tận dụng 9 cột thông tin còn lại trong hồ sơ để thiết kế **5 hướng dịch vụ mở rộng** gắn ngay vào thẻ gợi ý CSKH:
> 1. **Đồng hành thuê bao mới** (dựa trên `tenure`).
> 2. **Đường dây ưu tiên thân thiện cho người cao tuổi** (dựa trên `SeniorCitizen`).
> 3. **Tư vấn gói Gia đình & Đa thiết bị** (dựa trên `Partner`, `Dependents`, `MultipleLines`).
> 4. **Sao lưu đám mây & Bảo vệ thiết bị** (dựa trên `OnlineBackup`, `DeviceProtection`).
> 5. **Gói giải trí tích hợp trên Cáp quang** (dựa trên `StreamingTV/Movies`) nhằm đồng sáng tạo giá trị thay vì chỉ giảm giá trực tiếp."

---

### SLIDE 16 — Demo trực tiếp trên Web (`http://127.0.0.1:8505`)
> "Sau đây em xin phép demo nhanh 4 bước trên ứng dụng Web của nhóm:
> - *(Mở Tab 1)* Đây là bức tranh tổng quan tài chính với lợi nhuận ròng `+$31.999` và `ROI 138%`. Khi em kéo thử biên lợi nhuận `m` ở thanh bên trái, toàn bộ dòng tiền tự động cập nhật ngay.
> - *(Mở Tab 2)* Đây là 4 biểu đồ hành vi chứng minh cơ sở thiết kế gói `A1, A2, A3` cùng Top 10 đặc trưng CatBoost.
> - *(Mở Tab 3)* Khi chọn khách `4132-KALRO`, ta thấy ngay đồng hồ rủi ro giảm từ `84,6%` xuống `67,7%`, biểu đồ TreeSHAP lý giải từng yếu tố và Thẻ Gợi Ý Hành Động CSKH ngay bên dưới.
> - *(Mở Tab 4)* Khi em đặt ngân sách `15.000 USD`, bảng tự lọc danh sách khách hàng ưu tiên và cho phép lưu chiến dịch, xuất thẳng file `.CSV` cho đội CSKH. Ngoài ra ở thanh bên trái luôn có sẵn nút nạp file CSV mới (ví dụ 1.000 dòng) để chạy theo chu kỳ tuần/tháng."

---

### SLIDE 17 — Hạn chế & Hướng phát triển
> "Nhóm cũng nhìn nhận thẳng thắn 2 hạn chế của mô hình hiện tại: Thứ nhất, tập dữ liệu IBM là ảnh chụp tĩnh benchmark nên khi triển khai thật tại nhà mạng cần retrain định kỳ bằng dữ liệu 3–6 tháng gần nhất để tránh trôi dữ liệu. Thứ hai, các tham số tỷ lệ chấp nhận $u$ và hiệu quả giảm churn $e$ đang ở mức giả định mô phỏng — trong thực tế cần kết hợp **A/B Testing** để đo lường chính xác.
>
> Trong tương lai, nhóm có thể kết hợp thêm **LLM** ở bước hậu xử lý để tự động soạn lời thoại/email cá nhân hóa cho nhóm khách hàng VIP sau khi CatBoost và Knapsack đã lọc xong danh sách."

---

### SLIDE 18 — Kết luận & Q&A
> "Tóm lại, đồ án của nhóm đã hoàn thiện trọn vẹn từ **Dự báo chính xác (`CatBoost 84,8%`) + Giải thích minh bạch (`TreeSHAP`)**, đến **Định lượng kinh tế (`Proxy CLV`, `A0–A3`, $\Delta EV$, `Knapsack`)** và **Thiết kế dịch vụ CSKH (`Service Blueprint` & 5 dịch vụ mở rộng)**.
>
> Em xin cảm ơn cô và các bạn đã lắng nghe, rất mong nhận được góp ý và câu hỏi từ cô và cả lớp ạ!"
