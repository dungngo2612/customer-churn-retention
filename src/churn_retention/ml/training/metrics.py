"""
TEMPORARY - This module will be replaced by ml/evaluation module.
"""

import numpy as np

# scikit-learn ships neither stubs nor a py.typed marker for these imports.
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Compute classification metrics for binary churn prediction.

    TEMPORARY - This function will be replaced by ml/evaluation module.

    Args:
        y_true: True binary labels (0 or 1), where 1 = churn.
        y_prob: Predicted probabilities for class 1 (churn).
        threshold: Probability threshold for converting to binary predictions.

    Returns:
        Dictionary with metrics: average_precision, roc_auc, precision, recall,
        f1, brier_score, log_loss.

    Raises:
        ValueError: If y_true does not contain both classes.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "average_precision": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0, pos_label=1),
        "recall": recall_score(y_true, y_pred, zero_division=0, pos_label=1),
        "f1": f1_score(y_true, y_pred, zero_division=0, pos_label=1),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }
