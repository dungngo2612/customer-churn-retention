import pandas as pd
import pytest

from churn_retention.ml.evaluation.error_analysis import analyze_errors
from churn_retention.ml.evaluation.segments import evaluate_segments


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(8)],
            "Churn": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
            "Contract": ["Month-to-month"] * 4 + ["Two year"] * 4,
            "InternetService": ["DSL", "DSL", "Fiber optic", "Fiber optic"] * 2,
            "tenure": [2, 8, 15, 22, 30, 40, 55, 60],
            "MonthlyCharges": [20, 30, 40, 50, 60, 70, 80, 90],
        }
    )


def _predictions() -> pd.DataFrame:
    # Intentionally shuffled to verify ID based alignment.
    return pd.DataFrame(
        {
            "customerID": ["C7", "C6", "C5", "C4", "C3", "C2", "C1", "C0"],
            "churn_prob": [0.1, 0.95, 0.3, 0.8, 0.7, 0.2, 0.95, 0.05],
        }
    )


def test_error_analysis_returns_high_confidence_fp_and_fn_aligned_by_id() -> None:
    result = analyze_errors(_validation(), _predictions(), threshold=0.5, limit=2)

    assert result["error_type"].value_counts().to_dict() == {
        "false_positive": 2,
        "false_negative": 2,
    }
    fp = result[result["error_type"] == "false_positive"]
    fn = result[result["error_type"] == "false_negative"]
    assert fp["customerID"].tolist() == ["C6", "C4"]
    assert fn["customerID"].tolist() == ["C7", "C5"]
    assert set(result["error_type"]) == {"false_positive", "false_negative"}


def test_error_analysis_rejects_mismatched_ids() -> None:
    predictions = _predictions().iloc[:-1]
    with pytest.raises(ValueError, match="thiếu"):
        analyze_errors(_validation(), predictions, threshold=0.5)


def test_segments_reports_required_dimensions_and_warns_on_small_groups() -> None:
    result = evaluate_segments(
        _validation(),
        _predictions(),
        threshold=0.5,
        min_samples=1,
        min_class_count=1,
    )

    assert set(result["segment_type"]) == {
        "Contract",
        "InternetService",
        "tenure",
        "MonthlyCharges",
    }
    assert {
        "sample_count",
        "churn_count",
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "status",
    }.issubset(result.columns)
    assert result["sample_count"].sum() == 8 * 4
    assert "ít mẫu" not in " ".join(result["status"])


def test_segments_flags_small_or_single_class_groups() -> None:
    val = _validation()
    predictions = _predictions()
    result = evaluate_segments(
        val,
        predictions,
        threshold=0.5,
        min_samples=30,
        min_class_count=10,
    )

    assert result["status"].eq("ít mẫu, không kết luận mạnh").all()
    assert result["roc_auc"].notna().all()
