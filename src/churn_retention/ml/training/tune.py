"""Train-only CV experiments using the existing training pipelines."""

import json
import multiprocessing as mp
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import product
from math import isfinite
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import mlflow
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

from churn_retention.adapters.model_registry.mlflow_adapter import (
    flatten_params,
    get_tracking_uri,
)
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


def _fresh_pipeline(pipeline: Any) -> Any:
    """Clone preprocessing and rebuild the model from constructor parameters.

    CatBoost copies cat_features in its constructor, which fails sklearn's strict
    whole-pipeline clone check in some versions. Reconstructing that final step
    copies only constructor parameters, never fitted trees or class weights.
    The original pipeline structure and preprocessing are retained unchanged.
    """
    parameters = pipeline.get_params(deep=False)
    parameters["steps"] = [
        (
            name,
            type(step)(**deepcopy(step.get_params(deep=False)))
            if name == "model"
            else clone(step),
        )
        for name, step in pipeline.steps
    ]
    return type(pipeline)(**parameters)


def _fit_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    candidate: Candidate,
    config: TuneConfig,
    templates: dict[tuple[str, bool], Any],
) -> Any:
    """Reuse existing trainers, then clone their unfitted pipeline structure.

    Each template is derived from a genuine fold fit; _fresh_pipeline discards ALL
    fitted state, including preprocessing and computed class weights. Default CatBoost
    runs precede the grid, so l2_leaf_reg can be set before fitting a clone without
    modifying train_catboost or performing an extra warm-up fit.
    """
    key = (candidate.model, candidate.class_weight)
    if key in templates:
        pipeline = _fresh_pipeline(templates[key])
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
    templates[key] = _fresh_pipeline(pipeline)
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


@dataclass
class TrialResult:
    candidate: Candidate
    run_id: str
    cv: dict[str, float] = field(default_factory=dict)
    val: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    status: str = "completed"


@dataclass
class ExperimentReport:
    config: TuneConfig
    trials: list[TrialResult]
    smoke: bool
    seconds: float
    stop_reason: str


def _worker(
    connection: Connection, function: Callable[..., Any], args: tuple[Any, ...]
) -> None:
    try:
        connection.send((True, function(*args)))
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        connection.close()


def _within_budget(
    function: Callable[..., Any], args: tuple[Any, ...], deadline: float
) -> Any:
    """Terminate even a native model fit when the shared deadline expires.

    Spawn avoids inheriting fitted models or the parent's MLflow run. Tracking
    and report finalization may finish after the computation budget expires.
    """
    if monotonic() >= deadline:
        raise TimeoutError("Experiment time budget exhausted")
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, function, args))
    process.start()
    sender.close()
    try:
        if not receiver.poll(max(0.0, deadline - monotonic())):
            raise TimeoutError("Experiment time budget exhausted")
        try:
            success, value = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("Training worker exited without a result") from exc
        if not success:
            raise RuntimeError(value)
        return value
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def _cv_job(
    X: pd.DataFrame,
    y: pd.Series,
    candidate: Candidate,
    config: TuneConfig,
    templates: dict[tuple[str, bool], Any],
) -> tuple[dict[str, float], dict[tuple[str, bool], Any]]:
    return evaluate_cv(X, y, candidate, config, templates), templates


def _validation_job(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: Candidate,
    config: TuneConfig,
    templates: dict[tuple[str, bool], Any],
) -> dict[str, float]:
    started = perf_counter()
    pipeline = _fit_candidate(X_train, y_train, candidate, config, templates)
    fit_seconds = perf_counter() - started
    # One prediction/evaluation per shortlisted candidate, always at threshold 0.5.
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    return {
        **{f"val_{key}": value for key, value in _score(y_val, probabilities).items()},
        "val_fit_seconds": fit_seconds,
    }


def run_experiments(config: TuneConfig, *, smoke: bool = False) -> ExperimentReport:
    """Rank by train CV AP; evaluate up to three candidates on val exactly once.

    Ranking never uses validation metrics and does not change the team's primary
    model (logistic). Partial CV trials are logged but excluded from ranking.
    """
    if smoke:
        config = replace(
            config,
            n_splits=2,
            max_trials=min(config.max_trials, 2),
            candidates=config.candidates[:2],
        )
    started = monotonic()
    deadline = started + config.time_budget_seconds
    train_frame = pd.read_csv(config.train_path, nrows=500 if smoke else None)
    X_train, y_train = _prepare(train_frame, config.training)
    mlflow.set_tracking_uri(get_tracking_uri())
    experiment = config.training.get("mlflow", {}).get(
        "experiment_name", "churn-retention"
    )
    mlflow.set_experiment(f"{experiment}-experiments{'-smoke' if smoke else ''}")
    templates: dict[tuple[str, bool], Any] = {}
    trials: list[TrialResult] = []
    stop_reason = "configured_trials_completed"
    for candidate in config.candidates[: config.max_trials]:
        if monotonic() >= deadline:
            stop_reason = "time_budget_exhausted"
            break
        with mlflow.start_run(run_name=candidate.name) as run:
            trial = TrialResult(candidate=candidate, run_id=run.info.run_id)
            trials.append(trial)
            mlflow.log_params(
                {
                    **flatten_params({candidate.model: candidate.params}),
                    "model": candidate.model,
                    "class_weight": candidate.class_weight,
                    "seed": config.seed,
                    "n_splits": config.n_splits,
                    "max_trials": config.max_trials,
                    "time_budget_seconds": config.time_budget_seconds,
                    "train_rows": len(X_train),
                    "metric_threshold": 0.5,
                    "threshold_name": "fixed_0.5",
                    "smoke": smoke,
                }
            )
            mlflow.set_tags(
                {"team_primary_model": "logistic", "ranking_metric": "cv_ap_mean"}
            )
            mlflow.log_artifact(str(config.source), artifact_path="config")
            mlflow.log_artifact(
                str(config.source.with_name("train_config.yaml")),
                artifact_path="config",
            )
            trial_started = monotonic()
            try:
                trial.cv, templates = _within_budget(
                    _cv_job, (X_train, y_train, candidate, config, templates), deadline
                )
            except TimeoutError:
                trial.status = stop_reason = "time_budget_exhausted"
            trial.seconds = monotonic() - trial_started
            mlflow.log_metrics({**trial.cv, "trial_seconds": trial.seconds})
            mlflow.set_tag("trial_status", trial.status)
        if trial.status != "completed":
            break

    ranked = sorted(
        (trial for trial in trials if trial.cv),
        key=lambda trial: -trial.cv["cv_ap_mean"],
    )
    shortlist = ranked[:3]
    if shortlist and monotonic() < deadline:
        # Val is loaded only after CV ranking, and is never passed to evaluate_cv.
        val_frame = pd.read_csv(config.val_path)
        target = config.training["features"]["target_col"]
        X_val, y_val = _prepare(
            val_frame, config.training, train_frame[target].unique().tolist()
        )
        for trial in shortlist:
            if monotonic() >= deadline:
                stop_reason = "time_budget_exhausted"
                break
            with mlflow.start_run(run_id=trial.run_id):
                mlflow.set_tag("selected_for_validation", "true")
                val_started = monotonic()
                try:
                    trial.val = _within_budget(
                        _validation_job,
                        (
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            trial.candidate,
                            config,
                            templates,
                        ),
                        deadline,
                    )
                except TimeoutError:
                    trial.status = stop_reason = "time_budget_exhausted"
                trial.seconds += monotonic() - val_started
                mlflow.log_metrics({**trial.val, "trial_seconds": trial.seconds})
                mlflow.set_tag("trial_status", trial.status)
            if trial.status != "completed":
                break
    elif shortlist:
        stop_reason = "time_budget_exhausted"
    return ExperimentReport(
        config=config,
        trials=trials,
        smoke=smoke,
        seconds=monotonic() - started,
        stop_reason=stop_reason,
    )


def write_report(report: ExperimentReport, path: Path) -> None:
    """Write measured results, including explicit smoke and budget limitations."""
    config = report.config
    mode = (
        "Smoke only: 2 folds, first 500 train rows, at most 2 candidates. "
        "The full model comparison has not been run."
        if report.smoke
        else "Full experiment with the configured trial and time limits."
    )
    lines = [
        "# ML experiments",
        "",
        mode,
        "",
        (
            "The team's primary model is logistic regression. Candidates are ranked "
            "by mean train CV AP; these results do not automatically change that decision."
        ),
        "",
        (
            f"Limits: max_trials={config.max_trials}, "
            f"time_budget_seconds={config.time_budget_seconds:g}, seed={config.seed}, "
            f"threshold=0.5 (churn when probability >= 0.5), folds={config.n_splits}."
        ),
        "",
        (
            "All candidates use identical shuffled stratified folds. Each fold uses "
            "a fresh pipeline with preprocessing and class weights fitted only on its "
            "training rows. The grid clones the existing CatBoost pipeline and sets "
            "l2_leaf_reg before fitting. Defaults come from configs/train_config.yaml."
        ),
        "",
        (
            "The top three completed CV candidates are refitted on all available train "
            "rows and evaluated once on val, if budget remains. In smoke mode, train "
            "means the first 500 rows. No test dataset is read. Std uses ddof=0."
        ),
        "",
        (
            "| Candidate | Configuration | CV AP mean±std | CV ROC-AUC mean±std | "
            "Brier mean±std | Log loss mean±std | Val AP | Time |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial in sorted(
        report.trials, key=lambda row: -row.cv.get("cv_ap_mean", float("-inf"))
    ):

        def summary(metric: str, values: dict[str, float] = trial.cv) -> str:
            if not values:
                return "—"
            return f"{values[f'cv_{metric}_mean']:.4f}±{values[f'cv_{metric}_std']:.4f}"

        settings = json.dumps(
            {**trial.candidate.params, "class_weight": trial.candidate.class_weight},
            sort_keys=True,
        )
        val_ap = f"{trial.val['val_ap']:.4f}" if trial.val else "—"
        timing = f"{trial.seconds:.2f}s; fit/fold {summary('fit_seconds')}s"
        lines.append(
            f"| {trial.candidate.name} | `{settings}` | {summary('ap')} | "
            f"{summary('roc_auc')} | {summary('brier')} | {summary('log_loss')} | "
            f"{val_ap} | {timing} |"
        )
    lines.extend(
        [
            "",
            (
                f"Stop reason: {report.stop_reason}; elapsed: {report.seconds:.2f}s. "
                "The shared computation deadline terminates active workers. Tracking "
                "and report finalization may finish after that deadline. Time includes "
                "worker startup; fit/fold measures training only."
            ),
            "",
            (
                "MLflow records CV mean/std for AP, ROC-AUC, Brier, log loss, precision, "
                "recall, F1@0.5 and fit time; shortlisted runs also record val metrics. "
                "A dash means validation was not performed or CV did not finish."
            ),
            "",
        ]
    )
    for trial in report.trials:
        lines.append(
            f"- `{trial.candidate.name}`: run `{trial.run_id}`, {trial.status}."
        )
    lines.extend(
        [
            "",
            "Run the full experiment from the repository root:",
            "",
            "```sh",
            (
                "PYTHONPATH=src python3 -m churn_retention.entrypoints.cli tune "
                "--config configs/training.yaml"
            ),
            "```",
            "",
            (
                "Append `--smoke` for a quick check. The command replaces this report; "
                "MLflow retains previous runs. Both CSV paths must already point to "
                "prepared train/val files; this command does not create dataset splits."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
