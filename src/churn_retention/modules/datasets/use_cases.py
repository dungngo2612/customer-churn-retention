import io
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def clean_and_split_features(df: pd.DataFrame):
    """
    Quy tắc làm sạch cố định:
    1. Chuẩn hóa khoảng trắng cho toàn bộ cột dạng text.
    2. Điền 0.0 cho các ô TotalCharges trống/khoảng trắng.
    3. Tách riêng cột 'Churn' ra khỏi features để tránh rò rỉ nhãn (label leakage).
    """
    df_clean = df.copy()

    # Chuẩn hóa khoảng trắng đầu/cuối của các cột chuỗi
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Chuẩn hóa cột TotalCharges
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].replace(r"^\s*$", "0.0", regex=True),
            errors="coerce",
        ).fillna(0.0)

    # Ép kiểu chuẩn cho tenure và MonthlyCharges
    if "tenure" in df_clean.columns:
        df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce").fillna(0).astype(int)
    if "MonthlyCharges" in df_clean.columns:
        df_clean["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce").fillna(0.0)

    # Tách riêng nhãn Churn (nếu có)
    labels_series = None
    if "Churn" in df_clean.columns:
        labels_series = df_clean["Churn"].map(
            lambda x: 1 if str(x).strip().lower() in {"yes", "1", "true"} else 0
        )
        # Loại bỏ hoàn toàn cột Churn khỏi tập đặc trưng (features)
        features_df = df_clean.drop(columns=["Churn"])
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

    # 2. Chạy module kiểm tra tính hợp lệ
    report = validate_dataset(raw_df)

    if not report.is_valid:
        return {
            "status": "invalid",
            "total_rows": report.total_rows,
            "errors": [asdict(err) for err in report.errors],
        }

    # 3. Tạo ID mới và thư mục lưu trữ
    dataset_id = str(uuid.uuid4())
    dataset_folder = STORAGE_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)

    # 4. Làm sạch và tách riêng nhãn
    features_df, labels_series = clean_and_split_features(raw_df)

    # Lưu ra định dạng Parquet để tối ưu hóa tốc độ và giữ nguyên kiểu dữ liệu
    features_df.to_parquet(dataset_folder / "features.parquet", index=False)
    if labels_series is not None:
        pd.DataFrame({
            "customerID": features_df["customerID"],
            "Churn": labels_series,
        }).to_parquet(dataset_folder / "labels.parquet", index=False)

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