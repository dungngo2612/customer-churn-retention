from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split


def test_training_config_accounts_for_source_columns_without_id_or_target_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "data" / "telco_churn.csv", nrows=0)
    config = yaml.safe_load((root / "configs" / "train_config.yaml").read_text(encoding="utf-8"))

    features = config["features"]
    configured_features = set(features["numeric_cols"]) | set(features["categorical_cols"])
    excluded = set(features["drop_cols"])
    target = features["target_col"]

    assert target == "Churn"
    assert target not in configured_features
    assert "customerID" in excluded
    assert "customerID" not in configured_features
    assert configured_features | excluded | {target} == set(source.columns)
    assert not (configured_features & excluded)


def test_split_manifest_partitions_source_customers_once_and_stratified() -> None:
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "data" / "telco_churn.csv", usecols=["customerID", "Churn"])
    splits = pd.read_csv(root / "artifacts" / "splits" / "dataset_splits.csv")

    assert list(splits.columns) == ["customerID", "split"]
    assert splits["customerID"].notna().all()
    assert splits["customerID"].is_unique
    assert set(splits["split"]) == {"train", "val", "test"}
    assert set(splits["customerID"]) == set(source["customerID"])

    labeled = splits.merge(source, on="customerID", validate="one_to_one")
    rates = labeled.groupby("split")["Churn"].apply(lambda labels: (labels == "Yes").mean())
    assert (rates.max() - rates.min()) < 0.01


def test_split_manifest_locks_row_and_churn_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "data" / "telco_churn.csv", usecols=["customerID", "Churn"])
    splits = pd.read_csv(root / "artifacts" / "splits" / "dataset_splits.csv")
    labeled = splits.merge(source, on="customerID", validate="one_to_one")

    assert labeled["split"].value_counts().to_dict() == {
        "train": 4930,
        "val": 1056,
        "test": 1057,
    }
    assert labeled.loc[labeled["Churn"] == "Yes", "split"].value_counts().to_dict() == {
        "train": 1308,
        "val": 280,
        "test": 281,
    }


def test_seed_42_reproduces_stratified_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    source = pd.read_csv(root / "data" / "telco_churn.csv", usecols=["customerID", "Churn"])
    manifest = pd.read_csv(root / "artifacts" / "splits" / "dataset_splits.csv")
    config = yaml.safe_load((root / "configs" / "train_config.yaml").read_text(encoding="utf-8"))
    seed = config["seed"]
    assert seed == 42

    train, remaining = train_test_split(
        source, test_size=0.30, random_state=seed, stratify=source["Churn"]
    )
    val, test = train_test_split(
        remaining, test_size=0.50, random_state=seed, stratify=remaining["Churn"]
    )
    recreated = pd.concat(
        [
            frame[["customerID"]].assign(split=name)
            for name, frame in (("train", train), ("val", val), ("test", test))
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        recreated.sort_values("customerID").reset_index(drop=True),
        manifest.sort_values("customerID").reset_index(drop=True),
    )


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize(
    ("relative_path", "expected_hash"),
    [
        (
            "data/telco_churn.csv",
            "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        ),
        (
            "artifacts/splits/dataset_splits.csv",
            "bced80937bf69798aa314d03bd35d59d6c14134a593997d3e3dd8c7fe8b6512c",
        ),
    ],
)
def test_audited_checksums_normalize_line_endings_to_lf(
    relative_path: str, expected_hash: str, newline: bytes
) -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / relative_path).read_bytes()
    normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert sha256(normalized).hexdigest() == expected_hash

    alternate_endings = normalized.replace(b"\n", newline)
    renormalized = alternate_endings.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert sha256(renormalized).hexdigest() == expected_hash
    audit = (root / "docs" / "leakage_audit.md").read_text(encoding="utf-8")
    assert expected_hash in audit
