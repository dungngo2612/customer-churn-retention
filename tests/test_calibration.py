"""Calibration leakage, reproducibility, metric and proposal checks."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from churn_retention.entrypoints.cli import __main__ as cli
from churn_retention.ml.training import calibration
from churn_retention.ml.training.tune import TuneConfig, load_tune_config


def _synthetic(tmp_path: Path) -> tuple[TuneConfig, pd.DataFrame, pd.DataFrame]:
    config = load_tune_config(
        Path(__file__).resolve().parents[1] / "configs" / "training.yaml"
    )
    config = replace(
        config,
        source=tmp_path / "configs" / "training.yaml",
        train_path=tmp_path / "data" / "processed" / "train.csv",
        val_path=tmp_path / "data" / "processed" / "val.csv",
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
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "value": rng.normal(size=60),
            "segment": ["a", "b", "c"] * 20,
            "Churn": ["No", "Yes"] * 30,
        },
        index=range(1000, 1060),
    )
    train.loc[1001, "value"] = np.nan
    val = pd.DataFrame(
        {
            "value": rng.normal(size=20),
            "segment": ["a", "b"] * 10,
            "Churn": ["No", "Yes"] * 10,
        },
        index=range(2000, 2020),
    )
    return config, train, val


def test_calibrators_and_pipeline_fit_only_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, train, val = _synthetic(tmp_path)
    train_ids = frozenset(train.index)
    val_ids = set(val.index)
    fits: list[frozenset[int]] = []
    calibrator_fits: list[frozenset[int]] = []
    original_fit = Pipeline.fit
    original_calibrator_fit = CalibratedClassifierCV.fit

    def observe_pipeline(self: Any, X: Any, y: Any = None, **kwargs: Any) -> Any:
        if "model" in self.named_steps:
            seen = frozenset(X.index)
            assert seen <= train_ids and not seen & val_ids
            np.testing.assert_array_equal(
                np.asarray(y), train.loc[X.index, "Churn"].eq("Yes").to_numpy(dtype=int)
            )
            assert self.named_steps["model"].class_weight is None
            assert self.named_steps["model"].random_state == 42
            assert not hasattr(self.named_steps["preprocessor"], "transformers_")
            fits.append(seen)
        return original_fit(self, X, y, **kwargs)

    def observe_calibrator(self: Any, X: Any, y: Any, **kwargs: Any) -> Any:
        assert frozenset(X.index) == train_ids
        assert not set(X.index) & val_ids
        assert self.ensemble is False
        calibrator_fits.append(frozenset(X.index))
        return original_calibrator_fit(self, X, y, **kwargs)

    def reject_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("In-memory calibration must not open any dataset")

    monkeypatch.setattr(Pipeline, "fit", observe_pipeline)
    monkeypatch.setattr(CalibratedClassifierCV, "fit", observe_calibrator)
    monkeypatch.setattr(pd, "read_csv", reject_read)
    result = calibration.fit_calibration(config, train, val)
    assert calibrator_fits == [train_ids, train_ids]
    expected = Counter({train_ids: 3})  # Raw fit plus two final train refits.
    for fit_rows, _ in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        train, train["Churn"]
    ):
        expected[frozenset(train.iloc[fit_rows].index)] += 2
    assert Counter(fits) == expected
    assert set(result.probabilities) == {"none", "sigmoid", "isotonic"}


def test_fixed_seed_and_report_bytes_are_reproducible(tmp_path: Path) -> None:
    config, train, val = _synthetic(tmp_path)
    first = calibration.fit_calibration(config, train, val)
    second = calibration.fit_calibration(config, train, val)
    for method in first.probabilities:
        np.testing.assert_array_equal(
            first.probabilities[method], second.probabilities[method]
        )
    assert first.metrics == second.metrics
    assert first.selected == second.selected
    pd.testing.assert_frame_equal(first.thresholds, second.thresholds)
    path = tmp_path / "docs" / "calibration.md"
    calibration.write_calibration_report(first, config, path)
    before = {p.name: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    calibration.write_calibration_report(second, config, path)
    after = {p.name: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    assert before == after
    assert len(before) == 5  # Report, three reliability CSVs, full threshold CSV.


def test_ece_is_zero_for_perfect_predictions() -> None:
    labels = np.array([0, 0, 1, 1])
    assert calibration.expected_calibration_error(labels, labels.astype(float)) == 0


def test_ece_matches_weighted_hand_calculation() -> None:
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.2, 0.8, 1.0])
    assert calibration.expected_calibration_error(
        labels, probabilities, 2
    ) == pytest.approx(0.5 * abs(0.15 - 0.5) + 0.5 * abs(0.9 - 0.5))
    table = calibration.reliability_table(labels, probabilities)
    assert len(table) == 10
    assert table["count"].sum() == 4
    assert table.iloc[-1]["count"] == 1  # Probability 1 belongs to the last bin.
    assert table.loc[table["count"] == 0, "mean_probability"].isna().all()


@pytest.mark.parametrize(
    ("sigmoid", "isotonic", "expected"),
    [
        ((0.144, 0.847), (0.146, 0.851), "sigmoid"),
        ((0.146, 0.850), (0.140, 0.844), "none"),
        ((0.145, 0.845), (0.149, 0.850), "sigmoid"),
        ((0.140, 0.850), (0.139, 0.849), "isotonic"),
    ],
    ids=["apply", "reject_both_guards", "inclusive_boundary", "lowest_eligible_brier"],
)
def test_calibration_selection_rule(
    sigmoid: tuple[float, float], isotonic: tuple[float, float], expected: str
) -> None:
    metrics = {"none": {"brier_score": 0.15, "roc_auc": 0.85}}
    for name, (brier, auc) in (("sigmoid", sigmoid), ("isotonic", isotonic)):
        metrics[name] = {"brier_score": brier, "roc_auc": auc}
    assert calibration.select_calibration(metrics) == expected


def test_threshold_scan_and_equality_boundary() -> None:
    table = calibration.scan_thresholds(np.array([0, 1]), np.array([0.1, 0.5]))
    np.testing.assert_array_equal(table["threshold"], np.array(range(5, 96)) / 100)
    row = table.loc[table["threshold"] == 0.5].iloc[0]
    assert (row["TP"], row["FP"], row["FN"], row["TN"]) == (1, 0, 0, 1)
    assert row["f1"] == 1.0


def test_runner_opens_only_prepared_train_and_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, train, val = _synthetic(tmp_path)
    frames = {config.train_path: train, config.val_path: val}
    opened: list[Path] = []

    def read_only_allowed(path: Path) -> pd.DataFrame:
        assert path in frames
        opened.append(path)
        return frames[path]

    monkeypatch.setattr(pd, "read_csv", read_only_allowed)
    calibration.run_calibration(config)
    assert opened == [config.train_path, config.val_path]
    with pytest.raises(ValueError, match="only accepts"):
        calibration.run_calibration(replace(config, val_path=tmp_path / "other.csv"))
    assert opened == [config.train_path, config.val_path]


def test_calibrate_cli_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _synthetic(tmp_path)
    sentinel = object()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(cli, "load_tune_config", lambda path: config)
    monkeypatch.setattr(cli, "run_calibration", lambda loaded: sentinel)

    def write(result: Any, loaded: TuneConfig, path: Path) -> str:
        calls.append((result, loaded, path))
        return "không áp dụng\n"

    monkeypatch.setattr(cli, "write_calibration_report", write)
    assert cli.main(["calibrate", "--config", str(config.source)]) == 0
    assert calls == [(sentinel, config, tmp_path / "docs" / "calibration.md")]
