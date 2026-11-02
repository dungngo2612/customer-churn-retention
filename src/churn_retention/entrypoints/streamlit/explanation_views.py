"""Presentation helpers without importing or running Streamlit."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.ml.training.bundle import LoadedBundle

SCORING_CONTEXT_KEY = "active_scoring_context"


@dataclass(frozen=True)
class ScoringContext:
    bundle: LoadedBundle
    dataset: pd.DataFrame
    bundle_path: str


def scoring_context(
    bundle: LoadedBundle, dataset: pd.DataFrame, bundle_path: str
) -> ScoringContext:
    """Keep the exact successfully scored model and a snapshot of its input."""
    return ScoringContext(bundle, dataset.copy(deep=True), str(Path(bundle_path).resolve()))


def load_train_background(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    if resolved.name != "train.csv":
        raise ValueError("Background phải là train.csv; không dùng val.csv/test.csv")
    frame = pd.read_csv(resolved)
    frame.attrs["source_split"] = "train"
    return frame


def customer_ids(frame: pd.DataFrame) -> list[str]:
    column = next((c for c in ("customer_id", "customerID") if c in frame), None)
    if column is None:
        raise ValueError("Dataset cần customer_id hoặc customerID để chọn khách hàng")
    if frame[column].isna().any() or frame[column].astype(str).duplicated().any():
        raise ValueError("Customer IDs must be non-null and unique")
    return frame[column].astype(str).tolist()


def select_customer(frame: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    ids = customer_ids(frame)
    if customer_id not in ids:
        raise ValueError(f"Không tìm thấy khách hàng {customer_id}")
    return frame.iloc[[ids.index(customer_id)]].copy()


def global_importance_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["feature_importance"]).set_index("feature_name")


def local_contribution_frame(result: ModelExplanation) -> pd.DataFrame:
    rows = [
        {
            "feature": item.feature_name,
            "value": str(item.feature_value),
            "contribution_log_odds": item.contribution,
            "direction": "Tăng log-odds"
            if item.contribution > 0
            else ("Giảm log-odds" if item.contribution < 0 else "Không đổi"),
        }
        for item in result.contributions
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(
            "contribution_log_odds",
            key=lambda values: values.abs(),
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
