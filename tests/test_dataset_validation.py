import numpy as np
import pandas as pd

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
    """Kiểm tra chặn các giá trị vô cực (inf, -inf) trong các cột số."""
    df_raw = pd.DataFrame(
        {
            "customerID": ["KH-INF"],
            "tenure": [np.inf],
            "MonthlyCharges": [-np.inf],
            "TotalCharges": [np.inf],
            "Churn": ["No"],
        }
    )

    features, _ = clean_and_split_features(df_raw)

    # Đảm bảo toàn bộ giá trị sau khi làm sạch đều là số hữu hạn (finite)
    assert np.isfinite(features["tenure"].iloc[0])
    assert np.isfinite(features["MonthlyCharges"].iloc[0])
    assert np.isfinite(features["TotalCharges"].iloc[0])


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
