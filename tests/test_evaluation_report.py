import pandas as pd
import pytest

from churn_retention.ml.evaluation.report import generate_evaluation_report


def test_generate_evaluation_report_success():
    """Kiểm tra báo cáo tổng hợp chạy thành công với dữ liệu hợp lệ."""
    df_true = pd.DataFrame({"customerID": ["C1", "C2", "C3", "C4"], "Churn": [0, 0, 1, 1]})

    # Giả lập prediction bị đảo thứ tự dòng để kiểm tra tính năng align theo ID
    df_pred = pd.DataFrame(
        {"customerID": ["C4", "C2", "C1", "C3"], "churn_prob": [0.9, 0.2, 0.1, 0.8]}
    )

    report = generate_evaluation_report(df_true=df_true, df_pred=df_pred, threshold=0.5)

    # Kiểm tra cấu trúc báo cáo trả về
    assert report["total_samples"] == 4
    assert "average_precision" in report["ranking_metrics"]
    assert "roc_auc" in report["ranking_metrics"]
    assert "precision" in report["threshold_metrics"]
    assert "confusion_matrix" in report["threshold_metrics"]
    assert "brier_score" in report["calibration_metrics"]
    assert "log_loss" in report["calibration_metrics"]
    assert report["ranking_metrics"]["average_precision"] == 1.0
    assert report["ranking_metrics"]["roc_auc"] == 1.0

    assert report["threshold_metrics"]["precision"] == 1.0
    assert report["threshold_metrics"]["recall"] == 1.0
    assert report["threshold_metrics"]["f1"] == 1.0
    assert report["threshold_metrics"]["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }

    assert report["calibration_metrics"]["brier_score"] == 0.025
    assert report["calibration_metrics"]["log_loss"] == 0.1643

    def test_generate_evaluation_report_rejects_missing_customer_id():
        df_true = pd.DataFrame(
            {
                "customerID": ["C1", None],
                "Churn": [0, 1],
            }
        )
        df_pred = pd.DataFrame(
            {
                "customerID": ["C1", None],
                "churn_prob": [0.1, 0.9],
            }
        )

        with pytest.raises(ValueError, match="customerID bị thiếu hoặc rỗng"):
            generate_evaluation_report(df_true, df_pred)
