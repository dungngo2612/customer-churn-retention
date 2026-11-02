from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Đường dẫn file dữ liệu và nơi lưu trữ splits
DATA_PATH = Path("data/telco_churn.csv")
SPLITS_DIR = Path("artifacts/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)
print("=== THÔNG TIN TỔNG QUAN ===")
print(f"Tổng số dòng: {df.shape[0]}")
print(f"Tổng số cột: {df.shape[1]}")

# 3. Kiểm tra cột định danh (customerID)
ID_COL = "customerID"
dup_ids = df[ID_COL].duplicated().sum()
null_ids = df[ID_COL].isna().sum()
print("\n=== KIỂM TRA ĐỊNH DANH (customerID) ===")
print(f"Số ID rỗng: {null_ids}")
print(f"Số ID bị trùng: {dup_ids}")
if dup_ids == 0:
    print("-> Kết luận: Mỗi khách hàng là 1 dòng độc lập (Cross-sectional).")

# 4. Kiểm tra nhãn mục tiêu (Churn)
TARGET_COL = "Churn"
print("\n=== PHÂN BỐ NHÃN RỜI BỎ (Churn) ===")
churn_counts = df[TARGET_COL].value_counts()
churn_rates = df[TARGET_COL].value_counts(normalize=True) * 100
for val in churn_counts.index:
    print(f"- {val}: {churn_counts[val]} dòng ({churn_rates[val]:.2f}%)")

# 5. Kiểm tra giá trị thiếu và bất thường ở TotalCharges
print("\n=== KIỂM TRA GIÁ TRỊ THIẾU ===")
missing = df.isnull().sum()
print("Các cột có giá trị NaN theo pandas:")
print(missing[missing > 0] if (missing > 0).any() else "Không có cột nào chứa NaN thuần túy.")

# Cột TotalCharges có chứa khoảng trắng ' '
space_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
print(f"Số dòng TotalCharges chứa khoảng trắng rỗng (' '): {space_total_charges}")

# 6. Chia tập Stratified 70 / 15 / 15
print("\n=== CHIA TẬP STRATIFIED 70/15/15 ===")
# Tách 70% Train, 30% còn lại (Val + Test)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df[TARGET_COL])

# Tách đôi 30% thành 15% Val và 15% Test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET_COL]
)

print(f"Train: {len(train_df)} dòng ({len(train_df) / len(df):.1%})")
print(f"Val:   {len(val_df)} dòng ({len(val_df) / len(df):.1%})")
print(f"Test:  {len(test_df)} dòng ({len(test_df) / len(df):.1%})")

# 7. Lưu cố định danh sách ID
splits_data = pd.concat(
    [
        pd.DataFrame({ID_COL: train_df[ID_COL], "split": "train"}),
        pd.DataFrame({ID_COL: val_df[ID_COL], "split": "val"}),
        pd.DataFrame({ID_COL: test_df[ID_COL], "split": "test"}),
    ],
    ignore_index=True,
)

output_file = SPLITS_DIR / "dataset_splits.csv"
splits_data.to_csv(output_file, index=False)
print(f"\n-> Đã lưu danh sách phân chia ID tại: {output_file}")
