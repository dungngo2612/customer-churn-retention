import numpy as np
import pandas as pd

from churn_retention.ml.evaluation.metrics import align_and_validate_data


def analyze_errors(
    df_val: pd.DataFrame,
    df_pred: pd.DataFrame,
    threshold: float,
    limit: int = 10,
    id_col: str = "customerID",
    label_col: str = "Churn",
    prob_col: str = "churn_prob",
) -> pd.DataFrame:
    """Return up to limit confident FP/FN examples with their original features.

    False positives are ordered by highest churn probability; false negatives
    are ordered by lowest churn probability. The threshold is fixed by the caller.
    """
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold phải là số hữu hạn trong khoảng [0, 1].")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit phải là số nguyên dương.")

    required_val = {id_col, label_col}
    required_pred = {id_col, prob_col}
    missing_val = required_val - set(df_val.columns)
    missing_pred = required_pred - set(df_pred.columns)
    if missing_val:
        raise KeyError(f"Thiếu cột trong df_val: {sorted(missing_val)}")
    if missing_pred:
        raise KeyError(f"Thiếu cột trong df_pred: {sorted(missing_pred)}")

    y_true, _ = align_and_validate_data(
        df_true=df_val[[id_col, label_col]],
        df_pred=df_pred[[id_col, prob_col]],
        id_col=id_col,
        label_col=label_col,
        prob_col=prob_col,
    )
    probabilities = (
        df_pred.set_index(id_col)[prob_col].reindex(df_val[id_col]).to_numpy(dtype=float)
    )

    result = df_val.copy()
    result["y_true"] = y_true
    result["churn_prob"] = probabilities
    result["predicted_label"] = (probabilities >= threshold).astype(int)
    result["threshold"] = float(threshold)

    fp_mask = (result["y_true"] == 0) & (result["predicted_label"] == 1)
    fn_mask = (result["y_true"] == 1) & (result["predicted_label"] == 0)

    # High-confidence mistakes are farthest from the decision boundary.
    false_positives = (
        result.loc[fp_mask].sort_values("churn_prob", ascending=False).head(limit).copy()
    )
    false_positives["error_type"] = "false_positive"

    false_negatives = (
        result.loc[fn_mask].sort_values("churn_prob", ascending=True).head(limit).copy()
    )
    false_negatives["error_type"] = "false_negative"

    return pd.concat([false_positives, false_negatives], ignore_index=True)
