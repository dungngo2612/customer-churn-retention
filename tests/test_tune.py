"""Small synthetic checks for configuration and fold isolation."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_retention.ml.training import baseline, tune


def _synthetic(tmp_path: Path) -> tuple[pd.DataFrame, pd.Series, tune.TuneConfig]:
    config = tune.load_tune_config(
        Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    )
    config = replace(
        config,
        train_path=tmp_path / "train.csv",
        val_path=tmp_path / "val.csv",
        n_splits=3,
        training={
            **config.training,
            "features": {
                "numeric_cols": ["value"],
                "categorical_cols": ["segment"],
                "target_col": "Churn",
                "positive_label": "Yes",
                "drop_cols": [],
            },
        },
    )
    index = pd.Index(range(1000, 1060))
    X = pd.DataFrame(
        {
            "value": np.random.default_rng(42).normal(size=60),
            "segment": ["a", "b", "c"] * 20,
        },
        index=index,
    )
    y = pd.Series([0, 1] * 30, index=index)
    return X, y, config


def test_config_rejects_missing_fields(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    complete = yaml.safe_load(source.read_text())
    path = tmp_path / "training.yaml"
    for missing in complete:
        path.write_text(yaml.safe_dump({k: v for k, v in complete.items() if k != missing}))
        with pytest.raises(ValueError, match="Missing configuration fields"):
            tune.load_tune_config(path)


def test_cv_uses_only_train_indices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    X, y, config = _synthetic(tmp_path)
    train_ids = set(X.index)
    val_ids = set(range(2000, 2010))
    held_out_ids: list[int] = []

    def reject_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("CV must not open val or any other data file")

    class CheckedSplitter(StratifiedKFold):
        def split(self, X: pd.DataFrame, y: pd.Series, groups: Any = None) -> Any:
            assert set(X.index) == train_ids
            for fit_rows, score_rows in super().split(X, y, groups):
                assert set(X.iloc[fit_rows].index) <= train_ids
                scored = X.iloc[score_rows].index.tolist()
                assert set(scored) <= train_ids
                assert not set(scored) & val_ids
                held_out_ids.extend(scored)
                yield fit_rows, score_rows

    monkeypatch.setattr(pd, "read_csv", reject_read)
    monkeypatch.setattr(tune, "StratifiedKFold", CheckedSplitter)
    result = tune.evaluate_cv(X, y, config.candidates[0], config)
    assert sorted(held_out_ids) == sorted(train_ids)
    assert len(result) == 16  # Seven scores plus fit time, each with mean/std.
    assert all(np.isfinite(value) for value in result.values())


def test_fold_fit_never_sees_held_out_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    X, y, config = _synthetic(tmp_path)
    fits: list[set[int]] = []
    predictions: list[set[int]] = []

    class RecordingModel(BaseEstimator):
        def __init__(
            self,
            C: float = 1.0,
            max_iter: int = 1000,
            cat_features: list[str] | None = None,
        ) -> None:
            self.C = C
            self.max_iter = max_iter
            # Like CatBoost, copy constructor input: strict sklearn clone rejects it.
            self.cat_features = list(cat_features or [])

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecordingModel":
            assert not hasattr(self, "seen_indices_")
            assert X.index.equals(y.index)
            self.seen_indices_ = set(X.index)
            fits.append(self.seen_indices_)
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            assert not self.seen_indices_ & set(X.index)
            predictions.append(set(X.index))
            return np.tile([0.5, 0.5], (len(X), 1))

    def trainer(X: pd.DataFrame, y: pd.Series, *args: Any) -> Pipeline:
        return Pipeline([("model", RecordingModel())]).fit(X, y)

    monkeypatch.setattr(tune, "train_logistic", trainer)
    tune.evaluate_cv(X, y, config.candidates[0], config)
    assert len(fits) == len(predictions) == config.n_splits
    expected = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    ).split(X, y)
    for fit_ids, score_ids, (fit_rows, score_rows) in zip(fits, predictions, expected, strict=True):
        assert fit_ids == set(X.iloc[fit_rows].index)
        assert score_ids == set(X.iloc[score_rows].index)
        assert not fit_ids & score_ids
    assert tune._score([0, 1], [0.49, 0.5])["recall"] == 1.0


def test_real_preprocessing_fit_never_sees_held_out_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y, config = _synthetic(tmp_path)
    X.loc[[1000, 1007, 1014, 1021], "value"] = np.nan
    X.loc[[1002, 1011, 1035], "segment"] = np.nan
    fits: list[tuple[str, set[int]]] = []
    pipelines: list[Pipeline] = []

    class RecordingImputer(SimpleImputer):
        def fit(self, X: pd.DataFrame, y: Any = None) -> "RecordingImputer":
            assert not hasattr(self, "statistics_"), "Imputer must be fresh per fold"
            fits.append((self.strategy, set(X.index)))
            return super().fit(X, y)

    original_fit = tune._fit_candidate

    def capture_pipeline(*args: Any, **kwargs: Any) -> Pipeline:
        pipeline = original_fit(*args, **kwargs)
        pipelines.append(pipeline)
        return pipeline

    # Keep the real trainer and all transformations. Only observe imputer.fit;
    # inherited transform/fit_transform still execute the real sklearn behavior.
    monkeypatch.setattr(baseline, "SimpleImputer", RecordingImputer)
    monkeypatch.setattr(tune, "_fit_candidate", capture_pipeline)
    tune.evaluate_cv(X, y, config.candidates[0], config)
    expected = list(
        StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed).split(
            X, y
        )
    )
    expected_fits = [
        (strategy, set(X.iloc[fit_rows].index), set(X.iloc[val_rows].index))
        for fit_rows, val_rows in expected
        for strategy in ("median", "constant")
    ]
    for (strategy, seen), (expected_strategy, train_ids, val_ids) in zip(
        fits, expected_fits, strict=True
    ):
        assert seen.isdisjoint(val_ids), "Preprocessing saw held-out rows"
        assert seen == train_ids
        assert strategy == expected_strategy
    assert len(fits) == 2 * config.n_splits
    for pipeline, (fit_rows, _) in zip(pipelines, expected, strict=True):
        preprocessor = pipeline.named_steps["preprocessor"]
        numeric = preprocessor.named_transformers_["num"]
        categorical = preprocessor.named_transformers_["cat"]
        assert isinstance(numeric.named_steps["imputer"], RecordingImputer)
        assert isinstance(numeric.named_steps["scaler"], StandardScaler)
        assert isinstance(categorical.named_steps["onehot"], OneHotEncoder)
        assert numeric.named_steps["imputer"].statistics_[0] == pytest.approx(
            X.iloc[fit_rows]["value"].median()
        )


@pytest.mark.parametrize(
    ("budget", "completed", "attempted", "validated"),
    [(20.0, 4, 4, 3), (4.0, 4, 4, 0), (2.5, 2, 3, 0), (4.5, 4, 4, 0), (5.0, 4, 4, 1)],
)
def test_budget_counts_and_validation_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: float,
    completed: int,
    attempted: int,
    validated: int,
) -> None:
    X, y, config = _synthetic(tmp_path)
    config = replace(
        config,
        max_trials=4,
        candidates=config.candidates[:4],
        time_budget_seconds=budget,
        training={**config.training, "mlflow": {"experiment_name": "budget-check"}},
    )
    frame = X.assign(Churn=y.map({0: "No", 1: "Yes"}))
    frame.to_csv(config.train_path, index=False)
    frame.iloc[:10].to_csv(config.val_path, index=False)
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "budget-check-experiments", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    monkeypatch.setattr(tune, "get_tracking_uri", lambda: tracking_uri)
    clock = [0.0]
    monkeypatch.setattr(tune, "monotonic", lambda: clock[0])
    original_read = pd.read_csv
    original_log = tune.mlflow.log_params

    def slow_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        clock[0] += 100.0
        return original_read(*args, **kwargs)

    def slow_mlflow(*args: Any, **kwargs: Any) -> Any:
        clock[0] += 100.0
        return original_log(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", slow_csv)
    monkeypatch.setattr(tune.mlflow, "log_params", slow_mlflow)
    validation_calls: list[str] = []
    ap_by_name = dict(zip((c.name for c in config.candidates), [0.1, 0.9, 0.8, 0.7], strict=True))

    def timed_worker(function: Any, args: tuple[Any, ...], deadline: float) -> Any:
        remaining = deadline - clock[0]
        clock[0] += min(1.0, remaining)
        if remaining < 1.0:
            raise TimeoutError("Synthetic worker exhausted the budget")
        if function is tune._cv_job:
            scores = {
                f"cv_{metric}_{stat}": 0.1
                for metric in ("ap", "roc_auc", "brier", "log_loss", "fit_seconds")
                for stat in ("mean", "std")
            }
            scores["cv_ap_mean"] = ap_by_name[args[2].name]
            return scores, args[-1]
        assert function is tune._validation_job
        validation_calls.append(args[4].name)
        return {"val_ap": 0.8}

    monkeypatch.setattr(tune, "_within_budget", timed_worker)
    report = tune.run_experiments(config)
    assert report.trials_completed == completed
    assert report.trials_planned == 4
    assert len(report.trials) == attempted
    assert sum(bool(trial.val) for trial in report.trials) == validated
    assert report.seconds > budget  # CSV and MLflow do not consume worker time.
    ranked = sorted(
        (trial for trial in report.trials if trial.cv),
        key=lambda trial: -trial.cv["cv_ap_mean"],
    )[:3]
    assert validation_calls == [trial.candidate.name for trial in ranked[:validated]]
    skip_reason = "val: bỏ qua do hết ngân sách thời gian"
    for trial in report.trials:
        run = client.get_run(trial.run_id)
        assert run.data.metrics["trials_completed"] == completed
        assert run.data.metrics["trials_planned"] == 4
        assert run.data.tags["val_status"] == trial.val_status
        if trial.val:
            assert trial.val_status == "val: đã đánh giá"
        elif trial in ranked or not trial.cv:
            assert trial.val_status == skip_reason
        else:
            assert trial.val_status == "val: không thuộc top 3"
    assert len(client.search_runs([experiment_id])) == attempted
    report_path = tmp_path / "experiments.md"
    tune.write_report(report, report_path)
    markdown = report_path.read_text()
    assert f"trials_completed={completed}, trials_planned=4" in markdown
    assert (skip_reason in markdown) == (validated < len(ranked))
    assert "Thời gian đọc CSV và MLflow không nằm trong ngân sách" in markdown
