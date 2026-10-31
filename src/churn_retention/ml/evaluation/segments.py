from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from churn_retention.ml.evaluation.metrics import align_and_validate_data

DEFAULT_GROUP_COLUMNS = ("Contract", "InternetService")


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold phải là số hữu hạn trong khoảng [0, 1].")


def _aligned_frame(
    df_val: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    prob_col: str,
) -> pd.DataFrame:
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
    result["_y_true"] = y_true
    result["_y_prob"] = probabilities
    return result


def _group_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    groups: dict[str, pd.Series] = {}

    for column in DEFAULT_GROUP_COLUMNS:
        if column not in df.columns:
            raise KeyError(f"Thiếu cột phân nhóm: {column}")
        groups[column] = df[column].astype("string").fillna("Unknown")

    if "tenure" not in df.columns:
        raise KeyError("Thiếu cột phân nhóm: tenure")
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    tenure_bins = pd.cut(
        tenure,
        bins=[-np.inf, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
        right=True,
    )
    groups["tenure"] = tenure_bins.astype("string").fillna("Unknown")

    if "MonthlyCharges" not in df.columns:
        raise KeyError("Thiếu cột phân nhóm: MonthlyCharges")
    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    charge_groups = pd.Series("Unknown", index=df.index, dtype="string")
    valid = charges.notna()
    if valid.any():
        try:
            quartiles = pd.qcut(charges.loc[valid], q=4, duplicates="drop")
            charge_groups.loc[valid] = quartiles.astype("string")
        except ValueError:
            # Degenerate case: all observed charges are equal.
            charge_groups.loc[valid] = "Q1-Q4 (giá trị bằng nhau)"
    groups["MonthlyCharges"] = charge_groups
    return groups


def evaluate_segments(
    df_val: pd.DataFrame,
    df_pred: pd.DataFrame,
    threshold: float,
    *,
    min_samples: int = 30,
    min_class_count: int = 10,
    id_col: str = "customerID",
    label_col: str = "Churn",
    prob_col: str = "churn_prob",
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Tính metric validation theo Contract, InternetService, tenure và MonthlyCharges.

    MonthlyCharges quartiles are calculated from validation feature values for
    descriptive segmentation only. This function never changes the model or
    threshold. Small/one-class groups are flagged and should not support strong
    conclusions.
    """
    _validate_threshold(threshold)
    if min_samples < 1 or min_class_count < 1:
        raise ValueError("min_samples và min_class_count phải là số nguyên dương.")

    aligned = _aligned_frame(
        df_val,
        df_pred,
        id_col=id_col,
        label_col=label_col,
        prob_col=prob_col,
    )
    available_groups = _group_labels(aligned)
    requested = list(group_columns)
    if len(set(requested)) != len(requested):
        raise ValueError("group_columns không được có giá trị trùng.")
    unknown = set(requested) - set(available_groups)
    if unknown:
        raise KeyError(f"Nhóm không được hỗ trợ: {sorted(unknown)}")
    # Task 2.2 always includes these two numeric segmentations.
    requested.extend(name for name in ("tenure", "MonthlyCharges") if name not in requested)

    rows: list[dict[str, object]] = []
    for group_name in requested:
        work = aligned.copy()
        work["_segment"] = available_groups[group_name].to_numpy()
        for segment_value, part in work.groupby("_segment", dropna=False, sort=True):
            y_true = part["_y_true"].to_numpy(dtype=int)
            y_prob = part["_y_prob"].to_numpy(dtype=float)
            y_pred = (y_prob >= threshold).astype(int)
            churn_count = int(y_true.sum())
            non_churn_count = int(len(y_true) - churn_count)
            too_small = (
                len(part) < min_samples
                or churn_count < min_class_count
                or non_churn_count < min_class_count
            )

            rows.append(
                {
                    "segment_type": group_name,
                    "segment": str(segment_value),
                    "sample_count": int(len(part)),
                    "churn_count": churn_count,
                    "churn_rate": float(churn_count / len(part)),
                    "average_precision": (
                        float(average_precision_score(y_true, y_prob)) if churn_count > 0 else None
                    ),
                    "roc_auc": (
                        float(roc_auc_score(y_true, y_prob))
                        if churn_count > 0 and non_churn_count > 0
                        else None
                    ),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "threshold": float(threshold),
                    "status": ("ít mẫu, không kết luận mạnh" if too_small else "đủ mẫu để mô tả"),
                }
            )

    return pd.DataFrame(rows)
