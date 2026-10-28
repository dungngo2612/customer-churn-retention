from pathlib import Path

import pandas as pd
import yaml


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
