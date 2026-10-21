import io
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .validate import validate_dataset

# Thư mục lưu trữ các dataset đã qua xử lý và metadata
STORAGE_DIR = Path("artifacts/datasets")
METADATA_FILE = STORAGE_DIR / "metadata.json"


def _ensure_storage() -> None:
    """Đảm bảo thư mục lưu trữ và file metadata.json luôn sẵn sàng."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load_metadata() -> Dict[str, Any]:
    """Đọc toàn bộ metadata của các dataset hiện có."""
    _ensure_storage()
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_metadata(metadata: Dict[str, Any]) -> None:
    """Ghi đè thông tin metadata mới nhất xuống đĩa cứng."""
    _ensure_storage()
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def clean_and_split_features(
    df: pd.DataFrame, drop_churn: bool = True
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Quy tắc làm sạch cố định:
    1. Chuẩn hóa khoảng trắng cho toàn bộ cột dạng text.
    2. Điền 0.0 cho các ô TotalCharges trống/khoảng trắng.
    3. Tách riêng cột 'Churn' ra khỏi features để tránh rò rỉ nhãn (label leakage).
    4. Chuyển đổi và chặn số không hữu hạn (inf, -inf) cho các cột số.
    """
    df_clean = df.copy()

    # Chuẩn hóa khoảng trắng đầu/cuối của các cột chuỗi
    for col in df_clean.select_dtypes(include=["object", "str"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Chuẩn hóa cột TotalCharges
    if "TotalCharges" in df_clean.columns:
        # Thay thế khoảng trắng rỗng hoặc chuỗi trống thành NaN
        tc_series = df_clean["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
        tc_numeric = pd.to_numeric(tc_series, errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Điều kiện nghiệp vụ: Nếu tenure == 0 và TotalCharges rỗng -> điền 0.0
        if "tenure" in df_clean.columns:
            mask_new_customer = (df_clean["tenure"] == 0) & (tc_numeric.isna())
            tc_numeric = tc_numeric.mask(mask_new_customer, 0.0)

        # Với các dòng còn lại vẫn NaN (tenure > 0 bị mất cước), điền 0.0 an toàn
        df_clean["TotalCharges"] = tc_numeric.fillna(0.0)

    # Ép kiểu chuẩn cho tenure và MonthlyCharges
    if "tenure" in df_clean.columns:
        tenure_num = pd.to_numeric(df_clean["tenure"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        df_clean["tenure"] = tenure_num.fillna(0).astype(int)
    if "MonthlyCharges" in df_clean.columns:
        mc_numeric = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        df_clean["MonthlyCharges"] = mc_numeric.fillna(0.0)

    # Tách riêng nhãn Churn (nếu có)
    labels_series = None
    if "Churn" in df_clean.columns:
        labels_series = df_clean["Churn"].map(
            lambda x: 1 if str(x).strip().lower() in {"yes", "1", "true"} else 0
        )
        # Loại bỏ hoàn toàn cột Churn khỏi tập đặc trưng (features)
        if drop_churn:
            features_df = df_clean.drop(columns=["Churn"])
        else:
            features_df = df_clean
    else:
        features_df = df_clean

    return features_df, labels_series


def process_dataset_upload(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Luồng tiếp nhận file, chạy kiểm tra và lưu trữ dataset hợp lệ."""
    _ensure_storage()

    # 1. Đọc nội dung file CSV từ bytes
    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Không thể giải mã file CSV: {str(exc)}",
            "validation": None,
        }

    # 2. Làm sạch data
    # Cắt khoảng trắng thừa,
    # tự động điền 0.0 cho TotalCharges rỗng, ép kiểu số
    cleaned_features_df, labels_series = clean_and_split_features(raw_df)

    # Ghép tạm nhãn Churn lại để tạo thành bảng dữ liệu hoàn chỉnh đưa đi kiểm định
    df_to_validate = cleaned_features_df.copy()
    if labels_series is not None:
        df_to_validate["Churn"] = labels_series

    # 3. Validate dữ liệu đã làm sạch
    report = validate_dataset(df_to_validate)

    # Nếu sau khi làm sạch mà vẫn có lỗi vi phạm (thiếu cột, ID trùng/rỗng, số âm, sai nhãn)
    if not report.is_valid:
        return {
            "status": "invalid",
            "total_rows": report.total_rows,
            "errors": [asdict(err) for err in report.errors],
        }

    # 4. Khi dữ liệu đã sạch và hợp lệ -> Cấp phát dataset_id và lưu trữ Parquet
    dataset_id = str(uuid.uuid4())
    dataset_folder = STORAGE_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)

    features_path = dataset_folder / "features.parquet"
    cleaned_features_df.to_parquet(features_path, index=False)

    labels_path = None
    if labels_series is not None:
        labels_path = dataset_folder / "labels.parquet"
        labels_df = pd.DataFrame(
            {
                "customerID": cleaned_features_df["customerID"],
                "Churn": labels_series,
            }
        )
        labels_df.to_parquet(labels_path, index=False)

    # 5. Lưu thông tin metadata
    meta = _load_metadata()
    meta[dataset_id] = {
        "dataset_id": dataset_id,
        "filename": filename,
        "total_rows": len(raw_df),
        "total_columns": len(raw_df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "has_labels": labels_series is not None,
    }
    _save_metadata(meta)

    return {
        "status": "valid",
        "dataset_id": dataset_id,
        "total_rows": len(raw_df),
        "total_columns": len(raw_df.columns),
    }


def list_datasets() -> List[Dict[str, Any]]:
    """Lấy danh sách toàn bộ các dataset đã upload thành công (sắp xếp mới nhất lên đầu)."""
    meta = _load_metadata()
    return sorted(list(meta.values()), key=lambda x: x["created_at"], reverse=True)


def get_dataset_detail(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Tra cứu chi tiết một dataset cụ thể theo dataset_id."""
    meta = _load_metadata()
    return meta.get(dataset_id)
