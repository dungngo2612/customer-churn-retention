# Hồ sơ Bộ dữ liệu (Dataset Profile) & Chiến lược Chia tập

## 1. Nguồn dữ liệu & Bản chất
- **Nguồn:** Kaggle Telco Customer Churn (`blastchar/telco-customer-churn`).
- **Quy mô:** 7.043 dòng và 21 cột.
- **Bản chất:** Dạng cắt ngang (Cross-sectional Snapshot). Cột `customerID` duy nhất 100% (không có ID trùng, không có ID rỗng), mỗi hàng là một khách hàng độc lập.
- **Khả năng dự báo theo thời gian:** Bộ dữ liệu không chứa cột mốc thời gian sự kiện (`date`/`timestamp`), do đó mô hình chỉ dự báo xác suất rời bỏ ở trạng thái tĩnh (Static Churn Risk) dựa trên thời gian đã gắn bó (`tenure`), không hỗ trợ dự báo theo cửa sổ thời gian trượt ($N$ tháng tới).

## 2. Ý nghĩa nhãn mục tiêu (Churn)
- Cột nhãn: `Churn`.
- Phân bố:
  - `No`: 5.174 dòng (73.46%) - Khách hàng tiếp tục sử dụng.
  - `Yes`: 1.869 dòng (26.54%) - Khách hàng rời bỏ dịch vụ.
- Đánh giá: Có hiện tượng mất cân bằng lớp (tỷ lệ khoảng 1:3), cần ưu tiên đánh giá mô hình bằng PR-AUC và F1-Score.

## 3. Lỗi dữ liệu & Quy tắc xử lý
- Cột `TotalCharges`: Có kiểu dữ liệu dạng chuỗi (`object`) do chứa 11 dòng là ký tự khoảng trắng `' '` (rơi vào khách hàng mới có `tenure = 0`). Quy tắc làm sạch: chuyển về kiểu `float`, thay thế các giá trị trống này bằng `0.0`.

## 4. Chiến lược chia tập (Fixed Splits)
- **Phương pháp:** Stratified Random Split (giữ nguyên tỷ lệ 26.54% Churn đồng đều ở cả 3 tập).
- **Tỷ lệ:** 70% Train (4.930 dòng) / 15% Validation (1.056 dòng) / 15% Test (1.057 dòng).
- **Seed ngẫu nhiên:** `random_state=42`.
- **Lưu trữ:** File `artifacts/splits/dataset_splits.csv` lưu toàn bộ mapping giữa `customerID` và tập split tương ứng.