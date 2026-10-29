"""Train-background SHAP explanations in logistic log-odds."""

from churn_retention.ml.explainability.linear import (
    explain_global,
    explain_one,
    sample_background,
)

__all__ = ["explain_global", "explain_one", "sample_background"]
