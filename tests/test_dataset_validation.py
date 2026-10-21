import pandas as pd
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