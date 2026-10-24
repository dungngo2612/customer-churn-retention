"""Train-only CV experiments using the existing training pipelines."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.catboost_model import train_catboost
from churn_retention.workflows.train_model import (
    load_config,
    prepare_features_and_target,
    validate_dataframe,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    class_weight: bool
    params: dict[str, Any]


@dataclass(frozen=True)
class TuneConfig:
    train_path: Path
    val_path: Path
    seed: int
    n_splits: int
    max_trials: int
    time_budget_seconds: float
    candidates: tuple[Candidate, ...]
    training: dict[str, Any]
    source: Path


def load_tune_config(path: str | Path) -> TuneConfig:
    """Read experiment limits and reuse the sibling training configuration.

    Only train/val data paths are accepted. Relative paths use the project root
    (the parent of configs/). No data file is opened by this function.
    """
    source = Path(path).resolve()
    raw = load_config(source)
    required = {
        "train_path",
        "val_path",
        "seed",
        "n_splits",
        "max_trials",
        "time_budget_seconds",
        "candidates",
    }
    if not isinstance(raw, dict):
        raise TypeError("Experiment configuration must be a mapping")
    if missing := required - raw.keys():
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if unknown := raw.keys() - required:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    for key, minimum in (("seed", 0), ("n_splits", 2), ("max_trials", 1)):
        if type(raw[key]) is not int or raw[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if raw["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    budget = raw["time_budget_seconds"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("time_budget_seconds must be finite and positive")
    paths = {}
    for key in ("train_path", "val_path"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        paths[key] = (source.parent.parent / raw[key]).resolve()
    if paths["train_path"] == paths["val_path"]:
        raise ValueError("train_path and val_path must be different")
    groups = raw["candidates"]
    allowed = ("logistic", "catboost_default", "catboost_grid")
    if (
        not isinstance(groups, list)
        or not groups
        or any(group not in allowed for group in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError(f"candidates must be a unique, nonempty list from {allowed}")
    if "catboost_grid" in groups and "catboost_default" not in groups:
        raise ValueError(
            "catboost_grid requires catboost_default to reuse its pipeline"
        )

    training = load_config(source.with_name("train_config.yaml"))
    if not isinstance(training, dict) or not {"features", "models"} <= training.keys():
        raise ValueError("train_config.yaml must define features and models")
    candidates: list[Candidate] = []
    # Always schedule both logistic variants before the CatBoost comparison.
    for group, model in (("logistic", "logistic"), ("catboost_default", "catboost")):
        if group in groups:
            for weighted in (False, True):
                suffix = "balanced" if weighted else "unweighted"
                candidates.append(
                    Candidate(
                        f"{group}-{suffix}",
                        model,
                        weighted,
                        dict(training["models"][model]),
                    )
                )
    if "catboost_grid" in groups:
        for depth, rate, l2 in product((4, 6), (0.03, 0.1), (3, 10)):
            candidates.append(
                Candidate(
                    f"catboost-grid-d{depth}-lr{rate}-l2{l2}",
                    "catboost",
                    False,
                    {
                        **training["models"]["catboost"],
                        "depth": depth,
                        "learning_rate": rate,
                        "l2_leaf_reg": l2,
                    },
                )
            )
    return TuneConfig(
        train_path=paths["train_path"],
        val_path=paths["val_path"],
        seed=raw["seed"],
        n_splits=raw["n_splits"],
        max_trials=raw["max_trials"],
        time_budget_seconds=float(budget),
        candidates=tuple(candidates[: raw["max_trials"]]),
        training=training,
        source=source,
    )


def _score(y_true: Any, y_prob: Any, threshold: float = 0.5) -> dict[str, float]:
    # Temporary: replace with the shared evaluation module when it is available.
    prediction = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "ap": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def _prepare(
    frame: pd.DataFrame,
    training: dict[str, Any],
    allowed_labels: list[Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = training["features"]
    args = (
        features["numeric_cols"],
        features["categorical_cols"],
        features["target_col"],
        features["positive_label"],
        features.get("drop_cols", []),
    )
    validate_dataframe(frame, *args, allowed_labels=allowed_labels)
    return prepare_features_and_target(frame, *args)


def _fit_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    candidate: Candidate,
    config: TuneConfig,
    templates: dict[tuple[str, bool], Any],
) -> Any:
    """Reuse existing trainers, then clone their unfitted pipeline structure.

    Each template is cloned from a genuine fold fit; clone discards ALL fitted
    state, including preprocessing and computed class weights. Default CatBoost
    runs precede the grid, so l2_leaf_reg can be set before fitting a clone without
    modifying train_catboost or performing an extra warm-up fit.
    """
    key = (candidate.model, candidate.class_weight)
    if key in templates:
        pipeline = clone(templates[key])
        pipeline.set_params(
            **{f"model__{name}": value for name, value in candidate.params.items()}
        )
        pipeline.fit(X, y)
        return pipeline
    if "l2_leaf_reg" in candidate.params:
        raise ValueError("Run the default CatBoost candidate before its grid")
    features = config.training["features"]
    model_config = {
        "numeric_features": features["numeric_cols"],
        "categorical_features": features["categorical_cols"],
        candidate.model: candidate.params,
    }
    trainer = train_logistic if candidate.model == "logistic" else train_catboost
    pipeline = trainer(X, y, model_config, config.seed, candidate.class_weight)
    templates[key] = clone(pipeline)
    return pipeline


def evaluate_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate: Candidate,
    config: TuneConfig,
    templates: dict[tuple[str, bool], Any] | None = None,
) -> dict[str, float]:
    """Cross-validate only the supplied train rows; never open another dataset.

    All candidates use identical stratified folds. Every fit receives only that
    fold's training rows, and every prediction receives its held-out rows.
    Standard deviations use ddof=0 over the folds.
    """
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have matching indices")
    counts = y_train.value_counts()
    if set(counts.index) != {0, 1} or counts.min() < config.n_splits:
        raise ValueError("Each training class must have at least n_splits rows")
    if templates is None:
        templates = {}
    splitter = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    folds = []
    for fit_indices, score_indices in splitter.split(X_train, y_train):
        started = perf_counter()
        pipeline = _fit_candidate(
            X_train.iloc[fit_indices],
            y_train.iloc[fit_indices],
            candidate,
            config,
            templates,
        )
        fit_seconds = perf_counter() - started
        probabilities = pipeline.predict_proba(X_train.iloc[score_indices])[:, 1]
        scores = _score(y_train.iloc[score_indices], probabilities)
        folds.append({**scores, "fit_seconds": fit_seconds})
    return {
        f"cv_{metric}_{stat}": float(reducer([fold[metric] for fold in folds]))
        for metric in folds[0]
        for stat, reducer in (("mean", np.mean), ("std", np.std))
    }
