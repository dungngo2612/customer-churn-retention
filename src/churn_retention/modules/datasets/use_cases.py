import hashlib
import io
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.models import DatasetMetadataModel

from .validate import validate_dataset

# Thư mục lưu trữ các dataset đã qua xử lý và metadata
STORAGE_DIR = Path("artifacts/datasets")
SCHEMA_VERSION = "1"


def _ensure_storage() -> None:
    """Ensure the local dataset object directory exists."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_dataset(dataset: DatasetMetadataModel) -> Dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "total_rows": dataset.total_rows,
        "total_columns": dataset.total_columns,
        "has_labels": dataset.has_labels,
        "features_path": dataset.features_path,
        "labels_path": dataset.labels_path,
        "schema_version": dataset.schema_version,
        "checksum_sha256": dataset.checksum_sha256,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
    }


def _checksum_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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

        # Gán lại giá trị vào dataframe
        df_clean["TotalCharges"] = tc_numeric

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


def process_dataset_upload(
    file_bytes: bytes, filename: str, dataset_repo: DatasetRepository
) -> Dict[str, Any]:
    """Luồng tiếp nhận file, chạy kiểm tra và lưu trữ dataset hợp lệ."""

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
    _ensure_storage()
    dataset_id = str(uuid.uuid4())
    dataset_folder = STORAGE_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=False)

    features_path = dataset_folder / "features.parquet"
    labels_path: Optional[Path] = None
    try:
        cleaned_features_df.to_parquet(features_path, index=False)

        if labels_series is not None:
            labels_path = dataset_folder / "labels.parquet"
            labels_df = pd.DataFrame(
                {
                    "customerID": cleaned_features_df["customerID"],
                    "Churn": labels_series,
                }
            )
            labels_df.to_parquet(labels_path, index=False)

        stored_paths = [features_path]
        if labels_path is not None:
            stored_paths.append(labels_path)

        dataset_entity = DatasetMetadataModel(
            dataset_id=dataset_id,
            filename=filename,
            total_rows=len(cleaned_features_df),
            total_columns=len(raw_df.columns),
            has_labels=labels_series is not None,
            features_path=str(features_path),
            labels_path=str(labels_path) if labels_path is not None else None,
            schema_version=SCHEMA_VERSION,
            checksum_sha256=_checksum_paths(stored_paths),
        )
        dataset_repo.create(dataset=dataset_entity)
    except Exception:
        if dataset_folder.exists():
            shutil.rmtree(dataset_folder)
        raise

    return {
        "status": "valid",
        "dataset_id": dataset_id,
        "total_rows": len(raw_df),
        "total_columns": len(raw_df.columns),
    }


def check_dataset_quality(
    dataset_id: str, dataset_repo: DatasetRepository
) -> Optional[Dict[str, Any]]:
    """Check stored objects, checksum, row counts, IDs, and labeled schema."""
    dataset = dataset_repo.get_by_id(dataset_id)
    if dataset is None:
        return None

    errors: list[dict[str, Any]] = []

    def add_error(column: str, reason: str) -> None:
        errors.append({"column": column, "reason": reason})

    if dataset.schema_version != SCHEMA_VERSION:
        add_error(
            "schema_version",
            f"Unsupported schema version: {dataset.schema_version}",
        )

    features_path = Path(dataset.features_path) if dataset.features_path else None
    labels_path = Path(dataset.labels_path) if dataset.labels_path else None

    features_df: Optional[pd.DataFrame] = None
    labels_df: Optional[pd.DataFrame] = None
    stored_paths: list[Path] = []

    if features_path is None or not features_path.is_file():
        add_error("features_path", "Features Parquet file is missing.")
    else:
        stored_paths.append(features_path)
        try:
            features_df = pd.read_parquet(features_path)
        except Exception as exc:
            add_error("features_path", f"Cannot read features Parquet: {exc}")

    if dataset.has_labels:
        if labels_path is None or not labels_path.is_file():
            add_error("labels_path", "Labels Parquet file is missing.")
        else:
            stored_paths.append(labels_path)
            try:
                labels_df = pd.read_parquet(labels_path)
            except Exception as exc:
                add_error("labels_path", f"Cannot read labels Parquet: {exc}")
    elif labels_path is not None:
        add_error("labels_path", "Dataset has no labels but a labels file is recorded.")

    if features_df is not None:
        if len(features_df) != dataset.total_rows:
            add_error(
                "total_rows",
                f"Metadata says {dataset.total_rows} rows; features contain {len(features_df)}.",
            )
        if "customerID" not in features_df.columns:
            add_error("customerID", "Features are missing customerID.")
        else:
            ids = features_df["customerID"]
            if ids.isna().any() or ids.astype("string").str.strip().eq("").any():
                add_error("customerID", "Features contain missing or blank customerID.")
            if ids.duplicated().any():
                add_error("customerID", "Features contain duplicate customerID.")

    if dataset.has_labels and labels_df is not None:
        required = {"customerID", "Churn"}
        missing = required - set(labels_df.columns)
        if missing:
            add_error("labels", f"Labels are missing columns: {sorted(missing)}")
        elif len(labels_df) != dataset.total_rows:
            add_error(
                "total_rows",
                f"Metadata says {dataset.total_rows} rows; labels contain {len(labels_df)}.",
            )
        elif labels_df["customerID"].isna().any() or labels_df["customerID"].duplicated().any():
            add_error("customerID", "Labels contain missing or duplicate customerID.")
        elif features_df is not None and "customerID" in features_df.columns:
            if set(features_df["customerID"]) != set(labels_df["customerID"]):
                add_error("customerID", "Features and labels have different customer IDs.")
            else:
                complete_df = features_df.merge(
                    labels_df, on="customerID", how="inner", validate="one_to_one"
                )
                validation = validate_dataset(complete_df)
                errors.extend(asdict(error) for error in validation.errors)

    expected_columns = dataset.total_columns - (1 if dataset.has_labels else 0)
    if features_df is not None and len(features_df.columns) != expected_columns:
        add_error(
            "total_columns",
            f"Metadata expects {expected_columns} feature columns; "
            f"file has {len(features_df.columns)}.",
        )

    if dataset.checksum_sha256 is None:
        add_error("checksum_sha256", "No SHA-256 checksum is recorded.")
    elif len(stored_paths) == (2 if dataset.has_labels else 1):
        actual_checksum = _checksum_paths(stored_paths)
        if actual_checksum != dataset.checksum_sha256:
            add_error("checksum_sha256", "Stored object checksum does not match metadata.")

    return {
        "dataset_id": dataset.dataset_id,
        "is_valid": not errors,
        "total_rows": dataset.total_rows,
        "total_columns": dataset.total_columns,
        "schema_version": dataset.schema_version,
        "checksum_sha256": dataset.checksum_sha256,
        "errors": errors,
    }


def list_datasets(dataset_repo: DatasetRepository) -> List[Dict[str, Any]]:
    """Return dataset metadata from the database, newest first."""
    return [_serialize_dataset(item) for item in dataset_repo.list_all()]


def get_dataset_detail(
    dataset_id: str, dataset_repo: DatasetRepository
) -> Optional[Dict[str, Any]]:
    """Return one dataset metadata record from the database."""
    dataset = dataset_repo.get_by_id(dataset_id)
    return _serialize_dataset(dataset) if dataset is not None else None
