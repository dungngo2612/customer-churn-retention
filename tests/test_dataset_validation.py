from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from churn_retention.modules.datasets import use_cases
from churn_retention.modules.datasets.use_cases import clean_and_split_features
from churn_retention.modules.datasets.validate import validate_dataset


def test_validate_dataset_success() -> None:
    """Kiểm tra dataset hợp lệ vượt qua khâu validation."""
    data = {
        "customerID": ["001-A", "002-B"],
        "tenure": [1, 12],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": [29.85, 1889.5],
        "Churn": ["No", "Yes"],
    }
    df = pd.DataFrame(data)
    report = validate_dataset(df)

    assert report.is_valid is True
    assert len(report.errors) == 0


def test_validate_dataset_missing_column() -> None:
    """Kiểm tra báo lỗi khi thiếu cột bắt buộc."""
    data = {
        "customerID": ["001-A"],
        "tenure": [1],
        "MonthlyCharges": [29.85],
        # Thiếu TotalCharges và Churn
    }
    df = pd.DataFrame(data)
    report = validate_dataset(df)

    assert report.is_valid is False
    assert any(err.column == "TotalCharges" for err in report.errors)


def test_clean_totalcharges_by_tenure() -> None:
    """Kiểm tra xử lý TotalCharges theo tenure: tenure=0 gán 0.0, tenure>0 giữ nguyên."""
    df_raw = pd.DataFrame(
        {
            "customerID": ["KH-001", "KH-002"],
            "tenure": [0, 10],
            "MonthlyCharges": [25.5, 70.0],
            "TotalCharges": [" ", "700.0"],
            "Churn": ["No", "Yes"],
        }
    )

    features, labels = clean_and_split_features(df_raw)

    # Khách hàng mới (tenure = 0) có TotalCharges rỗng được gán 0.0
    val_new = features.loc[features["customerID"] == "KH-001", "TotalCharges"].iloc[0]
    assert val_new == 0.0

    # Khách hàng cũ (tenure = 10) giữ nguyên giá trị số thực
    val_old = features.loc[features["customerID"] == "KH-002", "TotalCharges"].iloc[0]
    assert val_old == 700.0

    assert "Churn" not in features.columns
    assert labels is not None
    assert list(labels) == [0, 1]


def test_block_non_finite_numbers() -> None:
    """Cleaner giữ số vô cực ở MonthlyCharges để validator báo lỗi."""
    df_raw = pd.DataFrame(
        {
            "customerID": ["KH-INF"],
            "tenure": [0],
            "MonthlyCharges": [-np.inf],
            "TotalCharges": [np.inf],
            "Churn": ["No"],
        }
    )

    features, _ = clean_and_split_features(df_raw, drop_churn=False)

    assert features["MonthlyCharges"].iloc[0] == -np.inf
    report = validate_dataset(features)
    assert report.is_valid is False
    assert any(err.column == "MonthlyCharges" and err.row == 2 for err in report.errors)


@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges", "TotalCharges"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "negative-inf"])
def test_validate_rejects_non_finite_numbers(column: str, value: float) -> None:
    df = pd.DataFrame(
        {
            "customerID": ["KH-VALID", "KH-NONFINITE"],
            "tenure": [1.0, 12.0],
            "MonthlyCharges": [25.0, 50.0],
            "TotalCharges": [25.0, 600.0],
            "Churn": ["No", "Yes"],
        }
    )
    df.loc[1, column] = value

    report = validate_dataset(df)

    assert report.is_valid is False
    column_errors = [err for err in report.errors if err.column == column]
    assert column_errors
    assert all(err.row == 3 and column in err.reason for err in column_errors)
    assert any("hữu hạn" in err.reason for err in column_errors)


@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "negative-inf"])
def test_cleaner_preserves_non_finite_numbers(column: str, value: float) -> None:
    df = pd.DataFrame(
        {
            "customerID": ["KH-NONFINITE"],
            "tenure": [12.0],
            "MonthlyCharges": [50.0],
            "TotalCharges": [600.0],
            "Churn": ["No"],
        }
    )
    df.loc[0, column] = value

    cleaned_df, _ = clean_and_split_features(df, drop_churn=False)

    cleaned_value = cleaned_df[column].iloc[0]
    if np.isnan(value):
        assert pd.isna(cleaned_value)
    else:
        assert cleaned_value == value
    report = validate_dataset(cleaned_df)
    assert report.is_valid is False
    assert any(err.column == column and err.row == 2 for err in report.errors)


def test_process_dataset_upload_rejects_inf_without_writing_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(use_cases, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(use_cases, "METADATA_FILE", tmp_path / "metadata.json")
    dataset_repo = Mock(spec=use_cases.DatasetRepository)
    file_bytes = b"customerID,tenure,MonthlyCharges,TotalCharges,Churn\nKH-INF,12,inf,600,No\n"

    result = use_cases.process_dataset_upload(file_bytes, "nonfinite.csv", dataset_repo)

    assert result["status"] == "invalid"
    assert any(err["column"] == "MonthlyCharges" and err["row"] == 2 for err in result["errors"])
    assert not list(tmp_path.rglob("*.parquet"))
    dataset_repo.create.assert_not_called()


def test_upload_validate_pipeline_flow() -> None:
    """Kiểm tra toàn bộ luồng tích hợp từ nhập dữ liệu, làm sạch đến xác thực schema."""
    df_raw = pd.DataFrame(
        {
            "customerID": ["7590-VHVEG"],
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [0],
            "PhoneService": ["No"],
            "MultipleLines": ["No phone service"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [29.85],
            "TotalCharges": [" "],
            "Churn": ["No"],
        }
    )

    # Bước 1: Làm sạch dữ liệu trước nhưng giữ lại cột 'Churn' để validate (drop_churn=False)
    cleaned_df, _ = clean_and_split_features(df_raw, drop_churn=False)

    # Bước 2: Đưa DataFrame đã làm sạch vào hàm validate kiểm tra schema
    report = validate_dataset(cleaned_df)

    assert report.is_valid is True
    assert len(report.errors) == 0
    assert cleaned_df["TotalCharges"].iloc[0] == 0.0

    # Bước 3: Sau khi validate thành công, tách riêng features và nhãn (drop_churn=True mặc định)
    features, labels = clean_and_split_features(cleaned_df, drop_churn=True)

    assert "Churn" not in features.columns
    assert labels is not None
    assert labels.iloc[0] == 0


def test_validate_rejects_missing_totalcharges_when_tenure_positive() -> None:
    """Kiểm tra hệ thống báo lỗi khi tenure > 0 mà TotalCharges bị trống."""
    df_corrupted = pd.DataFrame(
        {
            "customerID": ["KH-ERR"],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["No"],
            "Dependents": ["No"],
            "tenure": [10],  # Dùng 10 tháng
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.0],
            "TotalCharges": [" "],  # Nhưng cước lại trống -> Dữ liệu hỏng
            "Churn": ["No"],
        }
    )

    cleaned_df, _ = clean_and_split_features(df_corrupted, drop_churn=False)
    report = validate_dataset(cleaned_df)

    # Bắt buộc phải phát hiện ra lỗi và từ chối file
    assert report.is_valid is False
    assert any(err.column == "TotalCharges" for err in report.errors)
