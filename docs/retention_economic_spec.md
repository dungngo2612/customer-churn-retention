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

---

## 4. Retention Action Catalog

Hệ thống định nghĩa 4 gói chính sách chuẩn hóa:

| Mã | Tên hành động | $c_{\text{contact}}$ | $c_{\text{offer}}$ | $u_k$ | $e_k$ | $C_k$ | $r_k$ | Điều kiện hợp lệ ($\mathcal{A}_i$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0** | Không can thiệp (Baseline) | $0.00 | $0.00 | 0.00 | 0.00 | **$0.00** | **0.00%** | Luôn áp dụng cho mọi khách hàng |
| **A1** | Hỗ trợ Kỹ thuật & Bảo mật | $2.00 | $15.00 | 0.40 | 0.25 | **$8.00** | **10.00%** | Có Internet $\wedge$ (`TechSupport=No` $\vee$ `OnlineSecurity=No`) |
| **A2** | Tư vấn Hợp đồng & Auto-pay | $2.00 | $20.00 | 0.35 | 0.40 | **$9.00** | **14.00%** | `Contract=Month-to-month` $\vee$ Thanh toán thủ công |
| **A3** | Ưu đãi Cam kết Dài hạn | $3.00 | $40.00 | 0.40 | 0.50 | **$19.00** | **20.00%** | `Contract=Month-to-month` $\wedge$ `MonthlyCharges >= 50.0` |

---

## 5. System Interfaces & Input/Output Schema

### 5.1. Input Schema
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

---

## 8. Budget-Constrained Action Optimization (Task R2.1)

### 8.1. Mô hình Toán học: Multiple-Choice Knapsack Problem (MCKP)
Khi doanh nghiệp bị giới hạn bởi ngân sách cố định $B$ (USD), bài toán phân bổ can thiệp trở thành:

$$\max_{\{x_{i,k}\}} \sum_{i=1}^N \sum_{k \in \mathcal{A}_i} x_{i,k} \Delta EV_{i,k}$$

thỏa mãn 2 ràng buộc:
1. **Ràng buộc ngân sách:** $\sum_{i=1}^N \sum_{k \in \mathcal{A}_i} x_{i,k} C_k \le B$
2. **Ràng buộc tính đơn nhất:** $\sum_{k \in \mathcal{A}_i} x_{i,k} \le 1, \quad \forall i \in \{1, \dots, N\}$ (Mỗi khách hàng nhận tối đa một hành động).
3. **Biến quyết định nhị phân:** $x_{i,k} \in \{0, 1\}$.

### 8.2. Tính chất Xấp xỉ của Thuật toán Tham lam (Heuristic Approximation)
> [!IMPORTANT]
> **Tuyên bố về Tính chất Xấp xỉ (Approximation Notice):**
> Bài toán MCKP là bài toán NP-hard. Thuật toán Tham lam dựa trên tỷ số sinh lời chi phí $\frac{\Delta EV}{C}$ (Benefit-Cost Ratio Heuristic) được triển khai trong `modules/retention/budget.py` là **PHƯƠNG ÁN XẤP XỈ (heuristic approximation)** nhằm đảm bảo thời gian tính toán thực thi tức thời $O(N \log N)$ trên hàng chục nghìn khách hàng.

### 8.3. Quy tắc Thuật toán Tham lam
1. Tính toán $\Delta EV_{i,k} = B_{i,k} - C_k$ cho mọi cặp hợp lệ.
2. Lọc bỏ các cặp có $\Delta EV_{i,k} \le 0$ (chỉ giữ ứng viên có lãi).
3. Sắp xếp các cặp ứng viên:
   - Cặp có $C_k = 0$ và $\Delta EV > 0$: Ưu tiên chọn ngay lập tức (không chia cho 0).
   - Cặp có $C_k > 0$: Sắp xếp giảm dần theo tỷ số hiệu suất $\frac{\Delta EV_{i,k}}{C_k}$.
4. Duyệt tuần tự:
   - Nếu khách hàng $i$ chưa được gán và $\sum C + C_k \le B$: gán hành động $k$ cho khách $i$.
   - Ngược lại: bỏ qua.
5. Khách hàng chưa được gán: gán về phương án cơ sở $A0$ ($C=0, \Delta EV=0$).

### 8.4. Đối chiếu Nghiệm Toàn cục và Khoảng cách Tối ưu (Optimality Gap)
Trên tập dữ liệu nhỏ ($5 - 10$ khách hàng), hệ thống hỗ trợ hàm `solve_optimal_budget_brute_force` để duyệt vét cạn toàn bộ không gian tổ hợp và hàm `compare_greedy_vs_optimal` để đo khoảng cách tối ưu:

$$\text{Optimality Gap} = \frac{\Delta EV_{\text{optimal}} - \Delta EV_{\text{greedy}}}{\Delta EV_{\text{optimal}}}$$

Kết quả thực nghiệm trên tập mẫu kiểm thử cho thấy:
- Với ngân sách nhỏ hoặc vừa ($B=15$ USD): Thuật toán Tham lam đạt nghiệm tối ưu toàn cục ($\text{Optimality Gap} = 0\%$).
- Với ngân sách trung bình ($B=50$ USD): $\text{Optimality Gap} \approx 7.69\%$.
- Tổng chi phí thực tế ở mọi trường hợp luôn nghiêm ngặt thỏa mãn $\sum C \le B$.
