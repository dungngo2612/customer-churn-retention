"""Train-only probability calibration for the unweighted logistic pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.metrics import compute_metrics
from churn_retention.ml.training.tune import TuneConfig, _prepare

MIN_BRIER_IMPROVEMENT = 0.005
MAX_ROC_AUC_DROP = 0.005


@dataclass
class CalibrationResult:
    y_true: np.ndarray
    probabilities: dict[str, np.ndarray]
    train_rows: int
    seed: int
    n_splits: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    reliability: dict[str, pd.DataFrame] = field(default_factory=dict)
    selected: str = "none"
    thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def best_threshold(self) -> pd.Series:
        """Temporary max-F1 proposal; ties choose the lower threshold."""
        return self.thresholds.sort_values(
            ["f1", "threshold"], ascending=[False, True], kind="stable"
        ).iloc[0]


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
            raise ValueError("Calibration requires a logistic pipeline with a model step")
        raw.set_params(model__class_weight=None, model__random_state=config.seed)
        raw.fit(X_train, y_train)

    probabilities = {"none": raw.predict_proba(X_val)[:, 1]}
    for method in ("sigmoid", "isotonic"):
        folds = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        calibrated = CalibratedClassifierCV(
            estimator=clone(raw), method=method, cv=folds, ensemble=False, n_jobs=1
        )
        calibrated.fit(X_train, y_train)
        probabilities[method] = calibrated.predict_proba(X_val)[:, 1]
    result = CalibrationResult(
        y_true=y_val.to_numpy(dtype=int),
        probabilities=probabilities,
        train_rows=len(X_train),
        seed=config.seed,
        n_splits=config.n_splits,
    )
    for name, values in probabilities.items():
        scores = compute_metrics(result.y_true, values)
        result.reliability[name] = reliability_table(result.y_true, values)
        result.metrics[name] = {
            key: float(scores[key])
            for key in ("brier_score", "log_loss", "roc_auc", "average_precision")
        }
        result.metrics[name]["ece"] = _table_ece(result.reliability[name])
    result.selected = select_calibration(result.metrics)
    result.thresholds = scan_thresholds(result.y_true, probabilities[result.selected])
    return result


def reliability_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Aggregate equal-width probability bins [i/B, (i+1)/B), last includes 1.

    Empty bins have count=0 and undefined (NaN) means/rates, written as empty CSV
    cells. Only bin aggregates are exported, never IDs or individual predictions.
    """
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_proba, dtype=float)
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("Labels and probabilities must be equal-length 1D arrays")
    if not labels.size or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be nonempty and contain only 0/1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": bin_id / n_bins,
                "upper": (bin_id + 1) / n_bins,
                "count": count,
                "mean_probability": float(probabilities[mask].mean()) if count else np.nan,
                "churn_rate": float(labels[mask].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _table_ece(table: pd.DataFrame) -> float:
    populated = table.loc[table["count"] > 0]
    gaps = (populated["mean_probability"] - populated["churn_rate"]).abs()
    return float((populated["count"] * gaps).sum() / populated["count"].sum())


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * abs(mean_probability_b - churn_rate_b).

    Default: 10 equal-width bins [i/10, (i+1)/10), with 1 in the final bin.
    Empty bins contribute zero. This is binary positive-class probability ECE,
    not the top-label-confidence variant. No fitting or threshold selection occurs.
    """
    return _table_ece(reliability_table(y_true, y_proba, n_bins))


def write_reliability_tables(result: CalibrationResult, directory: Path) -> None:
    """Persist only ten-bin validation aggregates, in deterministic CSV format."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in result.reliability.items():
        table.to_csv(directory / f"reliability_{name}.csv", index=False, float_format="%.10f")


def select_calibration(metrics: dict[str, dict[str, float]]) -> str:
    """Propose calibration only for Brier gain >= .005 AND ROC-AUC drop <= .005.

    Compare with the uncalibrated validation baseline, using a numerical tolerance
    of 1e-12 at the inclusive boundaries. Among eligible methods choose lowest
    Brier, then highest ROC-AUC, then method name. Otherwise retain none.
    This proposal never changes a saved model or configuration.
    """
    baseline = metrics["none"]
    eligible = [
        name
        for name in ("sigmoid", "isotonic")
        if baseline["brier_score"] - metrics[name]["brier_score"] >= MIN_BRIER_IMPROVEMENT - 1e-12
        and baseline["roc_auc"] - metrics[name]["roc_auc"] <= MAX_ROC_AUC_DROP + 1e-12
    ]
    return min(
        eligible,
        key=lambda name: (
            metrics[name]["brier_score"],
            -metrics[name]["roc_auc"],
            name,
        ),
        default="none",
    )


def selection_message(result: CalibrationResult) -> str:
    if result.selected == "none":
        return "Đề xuất: không áp dụng hiệu chỉnh; giữ logistic không class_weight."
    return f"Đề xuất: áp dụng hiệu chỉnh {result.selected} cho logistic không class_weight."


def scan_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Evaluate all 91 thresholds .05 through .95 at .01 steps on val only.

    Churn means probability >= threshold. Reuse training.metrics because the
    shared evaluation module is not present on main or this branch's base.
    Counts supplement that existing helper; no threshold is locked or configured.
    """
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_proba)
    reliability_table(labels, probabilities)
    positive = labels == 1
    rows = []
    for integer in range(5, 96):
        threshold = integer / 100
        metrics = compute_metrics(labels, probabilities, threshold=threshold)
        predicted = probabilities >= threshold
        rows.append(
            {
                "threshold": threshold,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "TP": int((predicted & positive).sum()),
                "FP": int((predicted & ~positive).sum()),
                "FN": int((~predicted & positive).sum()),
                "TN": int((~predicted & ~positive).sum()),
            }
        )
    return pd.DataFrame(rows)


def significant_thresholds(result: CalibrationResult) -> pd.DataFrame:
    """Display every .05 increment plus max F1; the CSV retains all .01 steps."""
    table = result.thresholds
    return table.loc[
        (np.arange(len(table)) % 5 == 0)
        | (table["threshold"] == result.best_threshold["threshold"])
    ]


def write_calibration_artifacts(result: CalibrationResult, directory: Path) -> None:
    write_reliability_tables(result, directory)
    result.thresholds.to_csv(directory / "thresholds.csv", index=False, float_format="%.10f")


def run_calibration(config: TuneConfig) -> CalibrationResult:
    """Read exactly the existing prepared train/val files; perform no MLflow writes."""
    root = config.source.parent.parent
    for key, actual in (("train", config.train_path), ("val", config.val_path)):
        expected = (root / "data" / "processed" / f"{key}.csv").resolve()
        if actual.resolve() != expected:
            raise ValueError(f"Calibration only accepts data/processed/{key}.csv")
    train = pd.read_csv(config.train_path)
    val = pd.read_csv(config.val_path)
    return fit_calibration(config, train, val)


def render_calibration_report(result: CalibrationResult, config: TuneConfig) -> str:
    """Render measured results without timestamps, run IDs or fit-time noise."""
    settings = config.training["models"]["logistic"]
    best = result.best_threshold
    lines = [
        "# M2.2: Logistic calibration",
        "",
        "Base: `origin/feat/ml-experiments@1a3b674` (main did not contain M2.1).",
        (
            f"Logistic, class_weight=None, C={settings['C']}, max_iter={settings['max_iter']}, "
            f"seed={result.seed}, folds={result.n_splits}."
        ),
        (
            f"Existing M2.1 split: train={result.train_rows}, val={len(result.y_true)}; "
            "only data/processed/train.csv and val.csv are opened. Test data is never read."
        ),
        "",
        "## Train-only calibration",
        "",
        (
            "The baseline reuses the existing logistic preprocessing pipeline. "
            "Sigmoid (Platt) and isotonic wrap the complete pipeline in "
            "CalibratedClassifierCV with shuffled StratifiedKFold and ensemble=False. "
            "Each preprocessing fit sees only its training fold. Calibration learns "
            "from out-of-fold train scores; the final base pipeline is refitted on all "
            "train rows. Validation is never supplied to fit, including calibrator fit."
        ),
        "",
        "## Validation results",
        "",
        "| State | Brier | Log loss | ROC-AUC | PR-AUC (average precision) | ECE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in result.metrics.items():
        lines.append(
            f"| {name} | {metrics['brier_score']:.10f} | {metrics['log_loss']:.10f} | "
            f"{metrics['roc_auc']:.10f} | {metrics['average_precision']:.10f} | "
            f"{metrics['ece']:.10f} |"
        )
    lines.extend(
        [
            "",
            (
                "ECE = sum_b (n_b/N) * abs(mean_probability_b - churn_rate_b), with ten "
                "equal-width bins [i/10, (i+1)/10); the final bin includes probability 1. "
                "This is positive-class probability ECE. Empty bins contribute zero; "
                "their undefined means/rates are empty CSV cells."
            ),
            "",
            (
                "Reliability data (bin, bounds, count, mean probability, observed churn): "
                "[none](calibration/reliability_none.csv), "
                "[sigmoid](calibration/reliability_sigmoid.csv), "
                "[isotonic](calibration/reliability_isotonic.csv). "
                "These files contain aggregates only, without IDs or individual predictions."
            ),
            "",
            "## Calibration proposal",
            "",
            (
                "Apply calibration only if validation Brier decreases by at least 0.005 "
                "AND ROC-AUC decreases by no more than 0.005 versus none. "
                "Inclusive comparisons allow 1e-12 numerical tolerance. Among eligible "
                "methods choose lowest Brier, then highest ROC-AUC, then method name; "
                "otherwise retain none. This is a proposal; no saved model is replaced."
            ),
            "",
            selection_message(result),
            "",
            "## Validation threshold proposal",
            "",
            (
                "Scan: 0.05 through 0.95 inclusive, step 0.01; churn when proba >= threshold. "
                f"Selected state: {result.selected}. Max-F1 ties choose the lower threshold. "
                "Display every 0.05 increment plus max F1; "
                "[all 91 thresholds](calibration/thresholds.csv) are retained in CSV."
            ),
            "",
            (
                f"Temporary default only: threshold={best['threshold']:.2f}, "
                f"precision={best['precision']:.10f}, recall={best['recall']:.10f}, "
                f"F1={best['f1']:.10f}. This threshold is NOT locked and is NOT written to config."
            ),
            "",
            "| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in significant_thresholds(result).iterrows():
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {int(row['TP'])} | {int(row['FP'])} | "
            f"{int(row['FN'])} | {int(row['TN'])} |"
        )
    lines.extend(
        [
            "",
            (
                "The shared evaluation module is absent from both main and the selected "
                "base. Metrics reuse training.metrics.compute_metrics; confusion counts "
                "and reliability/ECE supplement it locally."
            ),
            "",
            "Reproduce from the repository root:",
            "",
            "```sh",
            "python3 -m churn_retention.entrypoints.cli calibrate --config configs/training.yaml",
            "```",
            "",
            (
                "The command replaces only this report and aggregate CSVs. It does not "
                "write MLflow or modify docs/experiments.md. Repeated runs with the "
                "same inputs and library versions produce identical docs and CSV bytes."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_calibration_report(result: CalibrationResult, config: TuneConfig, path: Path) -> str:
    text = render_calibration_report(result, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_calibration_artifacts(result, path.parent / "calibration")
    path.write_text(text, encoding="utf-8")
    return text
