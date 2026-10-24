"""Train-only CV experiments using the existing training pipelines."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Any

from churn_retention.workflows.train_model import load_config


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
        raise ValueError("Experiment configuration must be a mapping")
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
