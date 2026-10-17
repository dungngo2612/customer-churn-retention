# Technical Specification: Customer Retention Economics & Decision Engine

- **Target Module:** `churn_retention.modules.retention`
- **Scope:** Batch Decision Engine, Economic Optimization & Policy Simulation
- **Status:** Approved for Implementation

---

## 1. Problem Formulation & Objective

Mô-đun chịu trách nhiệm chuyển đổi xác suất rời mạng dự đoán ($p_i$) từ mô hình phân loại học máy thành quyết định can thiệp giữ chân tối ưu về mặt kinh tế cho từng khách hàng đang hoạt động.

Bài toán ra quyết định được chuẩn hóa thành bài toán tối đa hóa Giá trị ròng kỳ vọng tăng thêm ($\Delta EV$):

$$\max_{k \in \mathcal{A}_i} \Delta EV_{i,k} \quad \text{với } \Delta EV_{i,k} = B_{i,k} - C_k$$

Trong đó:
- $\mathcal{A}_i$: Tập hợp các hành động giữ chân mà khách hàng $i$ đủ điều kiện tham gia (Eligibility Set), luôn bao gồm phương án cơ sở $A0$ (Không can thiệp).
- $B_{i,k}$: Lợi ích kinh tế kỳ vọng thu hồi trước chi phí.
- $C_k$: Chi phí can thiệp kỳ vọng của hành động $k$.
- **Nguyên tắc can thiệp:** Doanh nghiệp chỉ can thiệp khi tồn tại hành động có $\Delta EV_{i,k} > 0$. Nếu tất cả các hành động đều có $\Delta EV \le 0$, hệ thống chỉ định $A0$ (Không can thiệp).

---

## 2. Target Population & Data Cleaning Scope

Bộ dữ liệu thực nghiệm `telco_churn.csv` gồm $7.043$ bản ghi và $21$ trường thông tin. Quy trình lọc dữ liệu trước khi đưa vào mô hình kinh tế được chuẩn hóa như sau:

| Phân nhóm khách hàng | Tiêu chí lọc | Số lượng (bản ghi) | Xử lý trong Pipeline Kinh tế |
| :--- | :--- | :---: | :--- |
| **Khách hàng đã rời mạng** | `Churn == "Yes"` | $1.869$ | **Loại bỏ khỏi mô phỏng can thiệp.** Chỉ dùng làm nhãn dương trong tập huấn luyện mô hình ML. |
| **Khách hàng mới gia nhập** | `tenure == 0` | $11$ | **Loại bỏ.** Chưa phát sinh chu kỳ cước hoàn chỉnh, giá trị `TotalCharges` khuyết thiếu. |
| **Khách hàng mục tiêu** | `Churn == "No"` $\wedge$ `tenure > 0` | **$5.163$** | **Tập mục tiêu duy nhất đưa vào đánh giá kinh tế và xếp hạng can thiệp.** |

---

## 3. Economic Framework & Proxy CLV Model

Do tập dữ liệu viễn thông lịch sử không ghi nhận chi phí vốn (WACC), biên đóng góp từng dịch vụ và dòng tiền chiết khấu tương lai, hệ thống thiết lập các quy chuẩn định giá thay thế:

### 3.1. Chuẩn hóa tham số định giá
* **Tiền tệ:** Đô la Mỹ (USD - $), chuẩn hóa theo đơn vị gốc của trường `MonthlyCharges`.
* **Kỳ hạn đánh giá ($H$):** $12$ tháng (chu kỳ tài khóa chuẩn).
* **Biên lợi nhuận đóng góp ($m$):** $50\%$ ($0.50$) ở kịch bản cơ sở (hỗ trợ kiểm thử độ nhạy ở dải $20\% - 50\%$).
* **Giá trị thay thế CLV ($V_i$):**
  $$V_i = \text{MonthlyCharges}_i \times H \times m$$

### 3.2. Cấu trúc hàm Chi phí và Lợi ích kỳ vọng
Với mỗi hành động $k$ có tỷ lệ khách hàng đồng ý $u_k \in [0, 1]$ và mức giảm churn tương đối ở nhóm đồng ý $e_k \in [0, 1]$:
* **Mức giảm rủi ro tương đối hiệu quả ($r_k$):**
  $$r_k = u_k \times e_k$$
* **Chi phí can thiệp kỳ vọng ($C_k$):**
  $$C_k = c_{\text{contact}, k} + u_k \times c_{\text{offer}, k}$$
* **Lợi ích kinh tế kỳ vọng ($B_{i,k}$):**
  $$B_{i,k} = p_i \times r_k \times V_i$$
* **Ngưỡng xác suất rời mạng hòa vốn ($p^*_{i,k}$):**
  $$p^*_{i,k} = \frac{C_k}{r_k \times V_i}$$

> **Nguyên tắc kế toán chống trừ trùng chi phí (Anti-Double-Counting Rule):**
> 1. $V_i$ phản ánh giá trị đóng góp tiềm năng trước can thiệp và luôn được giữ cố định trên cước gốc `MonthlyCharges`.
> 2. Toàn bộ giá trị ưu đãi (voucher, chiết khấu hóa đơn) và chi phí vận hành chỉ được hạch toán một lần duy nhất vào tham số chi phí kỳ vọng $C_k$. Tuyệt đối không khấu trừ ưu đãi vào cước phí khi tính $V_i$.
> 3. Mỗi khách hàng chỉ nhận tối đa 01 hành động tối ưu nhất (single best action), không cộng dồn lợi ích của nhiều hành động trên cùng một thuê bao.

---

## 4. Retention Action Matrix & Eligibility Rules

Hệ thống thiết lập 1 phương án cơ sở và 3 hành động giữ chân. Mọi điều kiện áp dụng đều được ánh xạ trực tiếp từ các cột hiện có trong tập dữ liệu:

| Mã | Hành động | Điều kiện áp dụng (Eligibility) | Cấu trúc Chi phí ($c_{\text{contact}} + u \cdot c_{\text{offer}}$) | Giả định $u_k, e_k$ | Mức giảm hiệu quả $r_k$ | Chi phí kỳ vọng $C_k$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **A0** | Không can thiệp | Mọi khách hàng đang hoạt động (`Churn == "No"`, `tenure > 0`) | $0 + 0$ | $u_0 = 0\%$<br>$e_0 = 0\%$ | **$0.0\%$** | **$\$0.00$** |
| **A1** | Hỗ trợ Kỹ thuật & Bảo mật chủ động | `InternetService != "No"`<br>$\wedge$ (`TechSupport == "No"` $\vee$ `OnlineSecurity == "No"`) | $2 + 0.40 \times 15$ | $u_1 = 40\%$<br>$e_1 = 25\%$ | **$10.0\%$** | **$\$8.00$** |
| **A2** | Tư vấn Hợp đồng & Auto-Pay | `Contract == "Month-to-month"`<br>$\vee$ `PaymentMethod` $\in$ `{"Electronic check", "Mailed check"}` | $2 + 0.35 \times 20$ | $u_2 = 35\%$<br>$e_2 = 40\%$ | **$14.0\%$** | **$\$9.00$** |
| **A3** | Ưu đãi Giảm cước Cam kết Dài hạn | `Contract == "Month-to-month"`<br>$\wedge$ `MonthlyCharges >= 50.0` | $3 + 0.40 \times 40$ | $u_3 = 40\%$<br>$e_3 = 50\%$ | **$20.0\%$** | **$\$19.00$** |

---

## 5. Software Architecture & Data Contracts

Mô-đun được triển khai trong `churn_retention.modules.retention.use_cases` theo kiến trúc xử lý theo lô (Vectorized Batch DataFrame Operations), đảm bảo hiệu năng cao khi tính toán kinh tế trên toàn bộ danh sách khách hàng:

```text
[Customers DataFrame] (N=7043)
        │
        ▼ (Lọc: Churn == 'No' & tenure > 0)
[Active Customers] (N=5163) ────┐
                                ├─► [use_cases.evaluate_actions()] ──► [Long-form Options DF]
[Predictions DataFrame] ────────┘    (Vectorized condition mask)                │
 (customer_id, churn_prob)                                                      ▼
                                                                     [use_cases.recommend()]
                                                                      (Argmax ΔEV > 0, tie: min Cost)
                                                                                │
                                                                                ▼
                                                                     [Ranked Strategy Output]
```

### 5.1. Input Contract Requirements
* **Customers Schema (`pd.DataFrame`):**
  - `customerID` (str, unique, non-null)
  - `MonthlyCharges` (float, $\ge 0$)
  - `Contract` (str: `Month-to-month`, `One year`, `Two year`)
  - `InternetService` (str: `DSL`, `Fiber optic`, `No`)
  - `TechSupport`, `OnlineSecurity` (str: `Yes`, `No`, `No internet service`)
  - `PaymentMethod` (str: `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`)
  - `Churn` (str: `Yes`, `No` hoặc int: `0`, `1`)
  - `tenure` (int, $> 0$)
* **Predictions Schema (`pd.DataFrame`):**
  - `customer_id` (str, ánh xạ $1:1$ với `customerID` của khách hàng đang hoạt động)
  - `churn_prob` / `churn_probability` (float, $[0.0, 1.0]$)

### 5.2. Output Contract Requirements (`recommend`)
Trả về bảng phân bổ chính sách tối ưu cho từng khách hàng, sắp xếp theo $\Delta EV$ giảm dần:
- `rank` (int, thứ tự ưu tiên can thiệp theo giá trị ròng)
- `customerID` (str)
- `action_code` (str: A0, A1, A2, A3)
- `action_name` (str)
- `churn_prob` (float, rủi ro ban đầu $p_i$)
- `churn_prob_after` (float, xác suất sau can thiệp $p'_i$)
- `breakeven_prob` (float, ngưỡng hòa vốn $p^*$)
- `proxy_value` (float, giá trị thay thế $V_i$)
- `cost` (float, chi phí kỳ vọng $C_k$)
- `benefit` (float, lợi ích kỳ vọng $B_{i,k}$)
- `net_value` (float, giá trị ròng kỳ vọng $\Delta EV_{i,k}$)

---

## 6. Verification Test Cases & Sensitivity Findings

### 6.1. Verified Calculation Baseline ($H=12, m=50\%$)
1. **Khách hàng A ($80/tháng, $p = 70\%$ - Phân khúc giá trị cao):**
   - $V = \$480.00$.
   - A1: $\text{Benefit}=\$33.60, \text{Cost}=\$8.00 \implies \Delta EV=+\$25.60$ ($p^*=16.67\%$).
   - A2: $\text{Benefit}=\$47.04, \text{Cost}=\$9.00 \implies \Delta EV=+\$38.04$ ($p^*=13.39\%$).
   - A3: $\text{Benefit}=\$67.20, \text{Cost}=\$19.00 \implies \Delta EV=+\$48.20$ ($p^*=19.79\%$).
   - **Quyết định:** Chọn **A3** ($\Delta EV$ cực đại).
2. **Khách hàng B ($25/tháng, $p = 35\%$ - Phân khúc giá trị thấp):**
   - $V = \$150.00$. A3 bị loại do điều kiện cước ($< \$50$).
   - A1: $\Delta EV = 5.25 - 8.00 = -\$2.75 < 0$ ($p^*=53.33\% > 35\%$).
   - A2: $\Delta EV = 7.35 - 9.00 = -\$1.65 < 0$ ($p^*=42.86\% > 35\%$).
   - **Quyết định:** Chọn **A0 (Không can thiệp, $\Delta EV = \$0.00$)**, ngăn ngừa rủi ro lỗ chi phí can thiệp.

### 6.2. Sensitivity Dynamics on Policy Selection
Phân tích độ nhạy chạy trên mẫu dữ liệu thực tế chỉ ra:
- **Biên lợi nhuận quyết định cơ cấu hành động:** Đối với khách hàng có cước cao ($80), ở mức $m = 20\%$ hành động tối ưu là **A2**; khi $m$ tăng lên $\ge 30\%$, hành động tối ưu chuyển dịch sang **A3**. Do đó, biên lợi nhuận không đơn thuần là hệ số khuếch đại dòng tiền mà trực tiếp định hình chiến lược giữ chân.
- **Sự lép vế của gói A1:** Ở các mức biên lợi nhuận thấp, A1 ($r=10\%, C=\$8$) bị chi phối bởi A2 ($r=14\%, C=\$9$). A1 chỉ được kích hoạt ở tập khách hàng hợp đồng dài hạn, thanh toán tự động nhưng phát sinh sự cố kỹ thuật.

---

## 7. Assumptions, Limitations & Methodological Caveats

1. **Bản chất của điểm số rủi ro ($p_i$):** $p_i$ thu được từ mô hình phân loại nhị phân trên dữ liệu cắt ngang lịch sử (Cross-sectional Churn Status), không phải mô hình phân tích thời gian sống (Survival Analysis). Do đó, $p_i$ đóng vai trò là chỉ số xếp hạng rủi ro (Risk Scoring Proxy), chưa phải xác suất chính xác theo thời gian thực trong 12 tháng kế tiếp.
2. **Giả định can thiệp phi nhân quả (Non-causal Assumption):** Các thông số hiệu quả giảm churn ($r_k$) và tỷ lệ tham gia ($u_k$) là các giả định mô phỏng kịch bản (Scenario Parameters). Do bộ dữ liệu không có nhóm đối chứng ngẫu nhiên (RCT), hệ thống không khẳng định mối quan hệ nhân quả. Việc áp dụng thực tế bắt buộc phải qua thử nghiệm A/B Testing để hiệu chỉnh tham số.
3. **Cước phí cố định:** Mô hình giả định mức cước hàng tháng không đổi trong kỳ hạn 12 tháng và bỏ qua tỷ lệ chiết khấu dòng tiền (Discount Rate).