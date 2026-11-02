from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn_retention.ml.evaluation.metrics import compute_threshold_metrics
from churn_retention.ml.evaluation.report import evaluate_predictions, generate_evaluation_report

ArrayEvaluator = Callable[[np.ndarray, np.ndarray, float], dict[str, Any]]


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


@pytest.fixture(
    params=[compute_threshold_metrics, evaluate_predictions],
    ids=["threshold_metrics", "evaluation_report"],
)
def array_evaluator(request: pytest.FixtureRequest) -> ArrayEvaluator:
    return request.param


@pytest.mark.parametrize(
    ("probability", "message"),
    [
        (np.nan, "NaN hoặc vô cực"),
        (np.inf, "NaN hoặc vô cực"),
        (-np.inf, "NaN hoặc vô cực"),
        (-0.1, "ngoài khoảng hợp lệ"),
        (1.1, "ngoài khoảng hợp lệ"),
    ],
    ids=["nan", "positive_inf", "negative_inf", "below_zero", "above_one"],
)
def test_array_evaluation_rejects_invalid_probabilities(
    array_evaluator: ArrayEvaluator, probability: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        array_evaluator(np.array([0, 1]), np.array([probability, 0.9]), 0.5)


@pytest.mark.parametrize("label", [0, 1])
def test_array_evaluation_rejects_single_class(array_evaluator: ArrayEvaluator, label: int) -> None:
    with pytest.raises(ValueError, match="cả lớp churn và không churn"):
        array_evaluator(np.array([label, label]), np.array([0.1, 0.9]), 0.5)


@pytest.mark.parametrize("label", [-1, 2, np.nan, np.inf])
def test_array_evaluation_rejects_invalid_labels(
    array_evaluator: ArrayEvaluator, label: float
) -> None:
    with pytest.raises(ValueError, match="0 và 1"):
        array_evaluator(np.array([0, label]), np.array([0.1, 0.9]), 0.5)


@pytest.mark.parametrize(
    ("y_true", "y_proba"),
    [
        (np.array([0, 1]), np.array([0.1])),
        (np.array([0]), np.array([0.1, 0.9])),
    ],
    ids=["shorter_probabilities", "shorter_labels"],
)
def test_array_evaluation_rejects_length_mismatch(
    array_evaluator: ArrayEvaluator, y_true: np.ndarray, y_proba: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="cùng độ dài"):
        array_evaluator(y_true, y_proba, 0.5)


def test_array_evaluation_rejects_empty_input(array_evaluator: ArrayEvaluator) -> None:
    with pytest.raises(ValueError, match="rỗng"):
        array_evaluator(np.array([]), np.array([]), 0.5)


def test_probability_equal_to_threshold_is_churn(array_evaluator: ArrayEvaluator) -> None:
    result = array_evaluator(np.array([0, 1]), np.array([0.5, 0.5]), 0.5)
    metrics = result.get("threshold_metrics", result)
    assert metrics["confusion_matrix"] == {
        "true_negative": 0,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 1,
    }


def test_evaluate_predictions_matches_dataframe_report() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.8, 0.5, 0.2])
    df_true = pd.DataFrame(
        {"customerID": ["C1", "C2", "C3", "C4"], "Churn": ["No", "No", "Yes", "Yes"]}
    )
    df_pred = pd.DataFrame(
        {"customerID": ["C4", "C2", "C1", "C3"], "churn_prob": [0.2, 0.8, 0.1, 0.5]}
    )

    assert evaluate_predictions(y_true, y_proba, threshold=0.5) == generate_evaluation_report(
        df_true, df_pred, threshold=0.5
    )


def test_array_evaluation_matches_hand_calculated_metrics(array_evaluator: ArrayEvaluator) -> None:
    result = array_evaluator(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.5, 0.2]), 0.5)
    metrics = result.get("threshold_metrics", result)
    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_evaluate_predictions_requires_threshold() -> None:
    with pytest.raises(TypeError, match="threshold"):
        evaluate_predictions(np.array([0, 1]), np.array([0.1, 0.9]))  # type: ignore[call-arg]
