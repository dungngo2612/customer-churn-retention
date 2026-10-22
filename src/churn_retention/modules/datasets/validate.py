from dataclasses import dataclass, field
from math import isfinite
from typing import List, Optional, Set

import pandas as pd


@dataclass
class ValidationErrorDetail:
    """Lưu chi tiết từng vị trí lỗi phát hiện được trong file dữ liệu."""

    row: Optional[int]  # Số thứ tự dòng trên file CSV (dòng 1 là tiêu đề, dữ liệu tính từ dòng 2)
    column: str  # Tên cột bị lỗi
    reason: str  # Mô tả cụ thể lỗi để trả lời cho người dùng


@dataclass
class ValidationReport:
    """Báo cáo tổng kết toàn bộ quá trình kiểm tra."""

    is_valid: bool
    total_rows: int
    errors: List[ValidationErrorDetail] = field(default_factory=list)


# 1. Danh sách các cột bắt buộc tối thiểu phải có
REQUIRED_COLUMNS: Set[str] = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

# 2. Tập hợp các giá trị nhãn rời bỏ hợp lệ
VALID_CHURN_VALUES = {"yes", "no", "0", "1", "true", "false"}


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Hàm kiểm tra toàn diện DataFrame đầu vào theo các quy tắc nghiệp vụ."""
    errors: List[ValidationErrorDetail] = []
    total_rows = len(df)

    # --- KIỂM TRA 1: THIẾU CỘT BẮT BUỘC ---
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        for col in sorted(missing_cols):
            errors.append(
                ValidationErrorDetail(
                    row=None,  # Lỗi cấu trúc bảng, không thuộc riêng dòng nào
                    column=col,
                    reason=f"Thiếu cột bắt buộc: '{col}'",
                )
            )
        # Nếu thiếu hẳn cột thì dừng kiểm tra sâu hơn vì không đủ dữ liệu
        return ValidationReport(is_valid=False, total_rows=total_rows, errors=errors)

    # --- KIỂM TRA 2: customerID BỊ RỖNG HOẶC TRÙNG LẶP ---
    for idx, val in df["customerID"].items():
        row_num = int(idx) + 2  # +2 vì pandas tính từ 0 và dòng 1 trong CSV là header
        if pd.isna(val) or str(val).strip() == "":
            errors.append(
                ValidationErrorDetail(
                    row=row_num,
                    column="customerID",
                    reason="customerID bị để trống hoặc chỉ chứa khoảng trắng",
                )
            )

    duplicated_mask = df["customerID"].duplicated(keep=False)
    for idx in df[duplicated_mask].index:
        val = df.loc[idx, "customerID"]
        errors.append(
            ValidationErrorDetail(
                row=int(idx) + 2,
                column="customerID",
                reason=f"customerID bị trùng lặp giá trị: '{val}'",
            )
        )

    # --- KIỂM TRA 3: KIỂU DỮ LIỆU VÀ MIỀN GIÁ TRỊ SỐ ---
    numeric_checks = [
        ("tenure", 0, None),  # tenure phải >= 0
        ("MonthlyCharges", 0.0, None),  # MonthlyCharges phải >= 0.0
        ("TotalCharges", 0.0, None),  # TotalCharges phải >= 0.0
    ]

    for col, min_val, max_val in numeric_checks:
        for idx, val in df[col].items():
            row_num = int(idx) + 2
            if pd.isna(val):
                errors.append(
                    ValidationErrorDetail(
                        row=row_num,
                        column=col,
                        reason=f"Giá trị tại cột '{col}' không phải số hữu hạn: '{val}'",
                    )
                )
                continue

            val_str = str(val).strip()

            if col == "TotalCharges" and val_str == "":
                continue

            if val_str == "":
                errors.append(
                    ValidationErrorDetail(
                        row=row_num,
                        column=col,
                        reason=f"Giá trị tại cột '{col}' bị để trống",
                    )
                )
                continue

            try:
                num = float(val_str)
                if not isfinite(num):
                    errors.append(
                        ValidationErrorDetail(
                            row=row_num,
                            column=col,
                            reason=f"Giá trị tại cột '{col}' không phải số hữu hạn: '{val}'",
                        )
                    )
                    continue
                if min_val is not None and num < min_val:
                    errors.append(
                        ValidationErrorDetail(
                            row=row_num,
                            column=col,
                            reason=f"Dữ liệu ngoài miền: '{col}' = {num} (yêu cầu >= {min_val})",
                        )
                    )
                if max_val is not None and num > max_val:
                    errors.append(
                        ValidationErrorDetail(
                            row=row_num,
                            column=col,
                            reason=f"Dữ liệu ngoài miền: '{col}' = {num} (yêu cầu <= {max_val})",
                        )
                    )
            except (ValueError, TypeError):
                errors.append(
                    ValidationErrorDetail(
                        row=row_num,
                        column=col,
                        reason=(
                            f"Sai kiểu dữ liệu cột '{col}': "
                            f"giá trị '{val}' không thể chuyển thành số"
                        ),
                    )
                )

    # --- KIỂM TRA 4: NHÃN CHURN HỢP LỆ ---
    for idx, val in df["Churn"].items():
        row_num = int(idx) + 2
        val_str = str(val).strip().lower() if not pd.isna(val) else ""

        if val_str == "":
            errors.append(
                ValidationErrorDetail(
                    row=row_num,
                    column="Churn",
                    reason="Cột nhãn Churn bị để trống",
                )
            )
        elif val_str not in VALID_CHURN_VALUES:
            errors.append(
                ValidationErrorDetail(
                    row=row_num,
                    column="Churn",
                    reason=f"Nhãn Churn không hợp lệ: '{val}'. "
                    "Chỉ chấp nhận: Yes, No, 0, 1, True, False",
                )
            )

    # Kiểm tra tính toàn vẹn: tenure > 0 thì TotalCharges bắt buộc phải > 0
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
        invalid_rows = df[(tenure > 0) & (total_charges.isna() | (total_charges <= 0))]
        for idx in invalid_rows.index:
            errors.append(
                ValidationErrorDetail(
                    row=int(idx) + 2,
                    column="TotalCharges",
                    reason=f"Khách hàng có tenure = {df.loc[idx, 'tenure']} "
                    "(> 0) nhưng TotalCharges bị trống hoặc <= 0.",
                )
            )

    return ValidationReport(
        is_valid=(len(errors) == 0),
        total_rows=total_rows,
        errors=errors,
    )
