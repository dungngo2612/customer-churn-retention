from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from churn_retention.workflows import train_model as workflow
from scripts.make_splits import make_splits


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "value": [0, 1, 2, 3],
            "segment": ["a", "b", "a", "b"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )
    val = pd.DataFrame({"value": [4.5, 5.5], "segment": ["a", "b"], "Churn": ["No", "Yes"]})
    return train, val


def _write_inputs(
    tmp_path: Path, train: pd.DataFrame, val: pd.DataFrame
) -> tuple[Path, Path, Path]:
    config = {
        "features": {
            "numeric_cols": ["value"],
            "categorical_cols": ["segment"],
            "drop_cols": [
                col for col in ("customerID", "customer_id") if col in train or col in val
            ],
            "target_col": "Churn",
            "positive_label": "Yes",
        },
        "models": {
            "logistic": {"C": 1.0, "max_iter": 100},
            "catboost": {"iterations": 5, "depth": 2, "learning_rate": 0.1},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    return config_path, train_path, val_path


@pytest.fixture
def reject_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_training(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Overlapping splits must be rejected before training or saving")

    for name in ("train_logistic", "train_catboost", "save_bundle", "log_training_run"):
        monkeypatch.setattr(workflow, name, unexpected_training)


@pytest.mark.usefixtures("reject_training")
@pytest.mark.parametrize("model_type", ["logistic", "catboost"])
@pytest.mark.parametrize("path_kind", ["same", "relative", "symlink"])
def test_rejects_paths_resolving_to_same_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    path_kind: str,
) -> None:
    config, train, val = _write_inputs(tmp_path, *_frames())
    if path_kind == "same":
        val = train
    elif path_kind == "relative":
        monkeypatch.chdir(tmp_path)
        val = Path("train.csv")
    else:
        val = tmp_path / "train-link.csv"
        val.symlink_to(train)

    with pytest.raises(ValueError, match="resolve to the same file"):
        workflow.train_model(config, model_type, False, 42, train, val, tmp_path / "bundles")
    assert not (tmp_path / "bundles").exists()


@pytest.mark.usefixtures("reject_training")
@pytest.mark.parametrize(
    ("train_id", "val_id"),
    [
        ("customerID", "customerID"),
        ("customer_id", "customer_id"),
        ("customerID", "customer_id"),
        ("customer_id", "customerID"),
    ],
)
def test_rejects_shared_customer_ids(tmp_path: Path, train_id: str, val_id: str) -> None:
    train_df, val_df = _frames()
    train_df[train_id] = ["shared", "train-1", "train-2", "train-3"]
    val_df[val_id] = [" shared ", "val-1"]
    config, train, val = _write_inputs(tmp_path, train_df, val_df)

    with pytest.raises(ValueError, match="share 1 customer ID"):
        workflow.train_model(config, "logistic", False, 42, train, val, tmp_path / "bundles")
    assert not (tmp_path / "bundles").exists()


@pytest.mark.usefixtures("reject_training")
@pytest.mark.parametrize("remaining_id", [None, "train", "val"])
def test_rejects_identical_rows_when_ids_are_missing(
    tmp_path: Path, remaining_id: str | None
) -> None:
    train_df, val_df = _frames()
    val_df = train_df.iloc[: len(val_df)].copy()
    val_df = val_df.loc[:, ["Churn", "segment", "value"]]
    if remaining_id == "train":
        train_df["customerID"] = ["train-0", "train-1", "train-2", "train-3"]
    elif remaining_id == "val":
        val_df["customer_id"] = ["val-0", "val-1"]
    config, train, val = _write_inputs(tmp_path, train_df, val_df)

    with pytest.raises(ValueError, match="identical rows.*customer IDs are unavailable"):
        workflow.train_model(config, "logistic", False, 42, train, val, tmp_path / "bundles")
    assert not (tmp_path / "bundles").exists()


@pytest.mark.parametrize("id_column", [None, "customerID", "customer_id"])
def test_disjoint_splits_still_train_and_save_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, id_column: str | None
) -> None:
    train_df, val_df = _frames()
    if id_column:
        # Different customers may legitimately have identical features and labels.
        val_df = train_df.iloc[:2].copy()
        train_df[id_column] = ["train-0", "train-1", "train-2", "train-3"]
        val_df[id_column] = ["val-0", "val-1"]
    config, train, val = _write_inputs(tmp_path, train_df, val_df)
    monkeypatch.setattr(workflow, "log_training_run", lambda **kwargs: "test-run")

    result = workflow.train_model(config, "logistic", False, 42, train, val, tmp_path / "bundles")

    assert result["run_id"] == "test-run"
    assert (result["bundle_dir"] / "model.pkl").is_file()
    assert (result["bundle_dir"] / "metadata.json").is_file()
    assert "roc_auc" in result["metrics"]


def test_row_fallback_compares_labels_as_well_as_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_df, _ = _frames()
    val_df = train_df.iloc[:2].copy()
    val_df["Churn"] = ["Yes", "No"]
    config, train, val = _write_inputs(tmp_path, train_df, val_df)
    monkeypatch.setattr(workflow, "log_training_run", lambda **kwargs: "test-run")

    result = workflow.train_model(config, "logistic", False, 42, train, val, tmp_path / "bundles")

    assert (result["bundle_dir"] / "model.pkl").is_file()


def test_real_splits_without_ids_allow_three_matching_validation_rows(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    make_splits(
        root / "data/telco_churn.csv",
        root / "artifacts/splits/dataset_splits.csv",
        train_path,
        val_path,
    )
    train = pd.read_csv(train_path).drop(columns="customerID")
    val = pd.read_csv(val_path).drop(columns="customerID")
    assert len(train) == 4930
    assert len(val) == 1056
    assert len(val.merge(train.drop_duplicates(), on=list(val.columns))) == 3

    workflow._validate_split_separation(train, val)


@pytest.mark.usefixtures("reject_training")
def test_rejects_validation_copy_of_train_without_ids(tmp_path: Path) -> None:
    train_df, _ = _frames()
    config, train, val = _write_inputs(tmp_path, train_df, train_df.copy())

    with pytest.raises(ValueError, match=r"4/4 validation rows \(100\.0%\)"):
        workflow.train_model(config, "logistic", False, 42, train, val, tmp_path / "bundles")
    assert not (tmp_path / "bundles").exists()


@pytest.mark.parametrize(("matching_rows", "should_raise"), [(17, False), (18, True), (19, True)])
def test_missing_id_overlap_threshold(matching_rows: int, should_raise: bool) -> None:
    train = pd.DataFrame({"value": range(20), "Churn": ["No", "Yes"] * 10})
    val = train.copy()
    val.loc[matching_rows:, "value"] += 1000

    if should_raise:
        with pytest.raises(ValueError, match=rf"{matching_rows}/20 validation rows"):
            workflow._validate_split_separation(train, val)
    else:
        workflow._validate_split_separation(train, val)
