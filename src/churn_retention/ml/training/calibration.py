"""Train-only probability calibration for the unweighted logistic pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.tune import TuneConfig, _prepare


@dataclass
class CalibrationResult:
    y_true: np.ndarray
    probabilities: dict[str, np.ndarray]
    train_rows: int
    seed: int
    n_splits: int


def fit_calibration(
    config: TuneConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    pipeline: Any | None = None,
) -> CalibrationResult:
    """Compare none/sigmoid/isotonic; every fit uses only the supplied train.

    CV uses M2.1's shuffled StratifiedKFold with the configured seed (42) and
    five folds. With ensemble=False, calibration learns from out-of-fold train
    scores, then the complete preprocessing/model pipeline is refitted on train.
    Val is used only for predict_proba and later evaluation/threshold proposals.
    A supplied pipeline is cloned, stripped of fitted state, and made unweighted.
    """
    X_train, y_train = _prepare(train, config.training)
    target = config.training["features"]["target_col"]
    X_val, y_val = _prepare(val, config.training, train[target].unique().tolist())
    counts = y_train.value_counts()
    if set(counts.index) != {0, 1} or counts.min() < config.n_splits:
        raise ValueError("Each training class must have at least n_splits rows")
    if set(y_val.unique()) != {0, 1}:
        raise ValueError("Validation must contain both classes (0 and 1)")

    if pipeline is None:
        features = config.training["features"]
        raw = train_logistic(
            X_train,
            y_train,
            {
                "numeric_features": features["numeric_cols"],
                "categorical_features": features["categorical_cols"],
                "logistic": config.training["models"]["logistic"],
            },
            config.seed,
            False,
        )
    else:
        raw = clone(pipeline)
        if not isinstance(raw.named_steps.get("model"), LogisticRegression):
            raise ValueError(
                "Calibration requires a logistic pipeline with a model step"
            )
        raw.set_params(model__class_weight=None, model__random_state=config.seed)
        raw.fit(X_train, y_train)

    probabilities = {"none": raw.predict_proba(X_val)[:, 1]}
    for method in ("sigmoid", "isotonic"):
        folds = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        calibrated = CalibratedClassifierCV(
            estimator=clone(raw), method=method, cv=folds, ensemble=False, n_jobs=1
        )
        calibrated.fit(X_train, y_train)
        probabilities[method] = calibrated.predict_proba(X_val)[:, 1]
    return CalibrationResult(
        y_true=y_val.to_numpy(dtype=int),
        probabilities=probabilities,
        train_rows=len(X_train),
        seed=config.seed,
        n_splits=config.n_splits,
    )
