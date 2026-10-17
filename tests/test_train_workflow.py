import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification

from churn_retention.workflows.train_model import (
    create_feature_schema,
    get_git_info,
    get_library_versions,
    prepare_features_and_target,
    train_model,
    validate_dataframe,
)


@pytest.fixture
def synthetic_train_val_data(tmp_path):
    """Create synthetic train/val CSV files."""
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        weights=[0.9, 0.1],
        flip_y=0,
        class_sep=0.8,
        random_state=42,
    )

    # Create DataFrame
    df = pd.DataFrame(X, columns=["num_1", "num_2", "num_3", "num_4"])

    # Add categorical feature
    np.random.seed(42)
    df["contract_type"] = np.random.choice(
        ["month_to_month", "one_year", "two_year"], size=len(df)
    )

    # Add target
    df["churn"] = ["Yes" if label == 1 else "No" for label in y]

    # Split train/val
    train_df = df.iloc[:800].copy()
    val_df = df.iloc[800:].copy()

    # Save CSVs
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, val_path


@pytest.fixture
def config_file(tmp_path):
    """Create a test config YAML."""
    config = {
        "seed": 42,
        "features": {
            "numeric_cols": ["num_1", "num_2", "num_3", "num_4"],
            "categorical_cols": ["contract_type"],
            "drop_cols": [],
            "target_col": "churn",
            "positive_label": "Yes",
        },
        "models": {
            "logistic": {"C": 1.0, "max_iter": 1000},
            "catboost": {"iterations": 50, "depth": 3, "learning_rate": 0.1},
        },
        "mlflow": {"experiment_name": "test-experiment"},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def test_validate_dataframe_valid(synthetic_train_val_data):
    """Test that valid DataFrame passes validation."""
    train_path, _ = synthetic_train_val_data
    df = pd.read_csv(train_path)

    # Should not raise
    validate_dataframe(
        df,
        numeric_cols=["num_1", "num_2", "num_3", "num_4"],
        categorical_cols=["contract_type"],
        target_col="churn",
        positive_label="Yes",
        drop_cols=[],
    )


def test_validate_dataframe_missing_column():
    """Test that missing required column raises error."""
    df = pd.DataFrame({"num_1": [1, 2, 3], "churn": ["Yes", "No", "Yes"]})

    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(
            df,
            numeric_cols=["num_1", "num_2"],  # num_2 missing
            categorical_cols=[],
            target_col="churn",
            positive_label="Yes",
            drop_cols=[],
        )

    assert "Missing required columns" in str(exc_info.value)


def test_validate_dataframe_non_numeric_values():
    """Test that non-numeric values in numeric column raises error."""
    df = pd.DataFrame({
        "num_1": ["1", "2", "invalid"],
        "churn": ["Yes", "No", "Yes"],
    })

    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(
            df,
            numeric_cols=["num_1"],
            categorical_cols=[],
            target_col="churn",
            positive_label="Yes",
            drop_cols=[],
        )

    assert "non-numeric values" in str(exc_info.value)


def test_validate_dataframe_invalid_target_label():
    """Test that invalid positive_label raises error."""
    df = pd.DataFrame({"num_1": [1, 2, 3], "churn": ["Yes", "No", "Yes"]})

    with pytest.raises(ValueError) as exc_info:
        validate_dataframe(
            df,
            numeric_cols=["num_1"],
            categorical_cols=[],
            target_col="churn",
            positive_label="Maybe",  # Invalid
            drop_cols=[],
        )

    assert "positive_label" in str(exc_info.value)


def test_prepare_features_and_target(synthetic_train_val_data):
    """Test feature and target preparation."""
    train_path, _ = synthetic_train_val_data
    df = pd.read_csv(train_path)

    X, y = prepare_features_and_target(
        df,
        numeric_cols=["num_1", "num_2", "num_3", "num_4"],
        categorical_cols=["contract_type"],
        target_col="churn",
        positive_label="Yes",
        drop_cols=[],
    )

    # Check X shape and column order
    assert X.shape[0] == len(df)
    assert list(X.columns) == ["num_1", "num_2", "num_3", "num_4", "contract_type"]

    # Check y encoding
    assert set(y.unique()) <= {0, 1}
    assert y.dtype == int


def test_create_feature_schema():
    """Test feature schema creation."""
    df = pd.DataFrame({
        "num_1": [1.0, 2.0, 3.0],
        "num_2": [4.0, 5.0, 6.0],
        "cat_1": ["a", "b", "c"],
    })

    schema = create_feature_schema(
        df, numeric_cols=["num_1", "num_2"], categorical_cols=["cat_1"]
    )

    assert len(schema) == 3
    assert schema[0] == {"name": "num_1", "kind": "numeric", "dtype": "float64"}
    assert schema[1] == {"name": "num_2", "kind": "numeric", "dtype": "float64"}
    assert schema[2] == {"name": "cat_1", "kind": "categorical", "dtype": "str"}


def test_get_git_info():
    """Test git info retrieval."""
    git_info = get_git_info()

    assert "git_commit" in git_info
    assert "git_dirty" in git_info

    # Should not fail even if not in git repo
    assert isinstance(git_info["git_commit"], (str, type(None)))
    assert isinstance(git_info["git_dirty"], (bool, type(None)))


def test_get_library_versions():
    """Test library version retrieval."""
    versions = get_library_versions()

    assert "python" in versions
    assert "scikit-learn" in versions
    assert "catboost" in versions
    assert "pandas" in versions
    assert "numpy" in versions

    # All should be strings
    assert all(isinstance(v, str) for v in versions.values())


def test_train_model_logistic_creates_bundle(
    synthetic_train_val_data, config_file, tmp_path
):
    """Test that train_model creates valid bundle for logistic model."""
    train_path, val_path = synthetic_train_val_data
    out_dir = tmp_path / "bundles"

    result = train_model(
        config_path=config_file,
        model_type="logistic",
        class_weight=False,
        seed=42,
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
    )

    # Check result structure
    assert "bundle_dir" in result
    assert "run_id" in result
    assert "model_version" in result
    assert "metrics" in result

    bundle_dir = result["bundle_dir"]

    # Check bundle files exist
    assert (bundle_dir / "model.pkl").exists()
    assert (bundle_dir / "feature_schema.json").exists()
    assert (bundle_dir / "metadata.json").exists()

    # Load and check model
    with open(bundle_dir / "model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    # Model should be able to predict
    df = pd.read_csv(val_path)
    X, _ = prepare_features_and_target(
        df,
        numeric_cols=["num_1", "num_2", "num_3", "num_4"],
        categorical_cols=["contract_type"],
        target_col="churn",
        positive_label="Yes",
        drop_cols=[],
    )
    y_prob = pipeline.predict_proba(X)[:, 1]
    assert len(y_prob) == len(X)
    assert all(0 <= p <= 1 for p in y_prob)


def test_train_model_catboost_creates_bundle(
    synthetic_train_val_data, config_file, tmp_path
):
    """Test that train_model creates valid bundle for catboost model."""
    train_path, val_path = synthetic_train_val_data
    out_dir = tmp_path / "bundles"

    result = train_model(
        config_path=config_file,
        model_type="catboost",
        class_weight=True,
        seed=100,
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
    )

    bundle_dir = result["bundle_dir"]

    # Check bundle exists
    assert (bundle_dir / "model.pkl").exists()
    assert (bundle_dir / "metadata.json").exists()

    # Check metadata
    with open(bundle_dir / "metadata.json") as f:
        metadata = json.load(f)

    assert metadata["model_type"] == "catboost"
    assert metadata["class_weight"] is True
    assert metadata["seed"] == 100
    assert "model_version" in metadata
    assert "catboost-" in metadata["model_version"]
    assert "-seed100" in metadata["model_version"]


def test_train_model_bundle_has_valid_metadata(
    synthetic_train_val_data, config_file, tmp_path
):
    """Test that bundle metadata contains all required fields."""
    train_path, val_path = synthetic_train_val_data
    out_dir = tmp_path / "bundles"

    result = train_model(
        config_path=config_file,
        model_type="logistic",
        class_weight=False,
        seed=42,
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
    )

    with open(result["bundle_dir"] / "metadata.json") as f:
        metadata = json.load(f)

    # Check required top-level keys
    assert "model_version" in metadata
    assert "model_type" in metadata
    assert "class_weight" in metadata
    assert "seed" in metadata
    assert "config" in metadata
    assert "label" in metadata
    assert "training" in metadata
    assert "git" in metadata
    assert "libraries" in metadata
    assert "metrics" in metadata

    # Check label info
    assert metadata["label"]["target_col"] == "churn"
    assert metadata["label"]["positive_label"] == "Yes"

    # Check git info
    assert "git_commit" in metadata["git"]
    assert "git_dirty" in metadata["git"]

    # Check libraries
    assert "python" in metadata["libraries"]
    assert "scikit-learn" in metadata["libraries"]


def test_train_model_feature_schema_correct_order(
    synthetic_train_val_data, config_file, tmp_path
):
    """Test that feature schema preserves correct column order."""
    train_path, val_path = synthetic_train_val_data
    out_dir = tmp_path / "bundles"

    result = train_model(
        config_path=config_file,
        model_type="logistic",
        class_weight=False,
        seed=42,
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
    )

    with open(result["bundle_dir"] / "feature_schema.json") as f:
        schema = json.load(f)

    # Check order: numeric_cols + categorical_cols
    expected_order = ["num_1", "num_2", "num_3", "num_4", "contract_type"]
    actual_order = [col["name"] for col in schema]

    assert actual_order == expected_order

    # Check kinds
    for i in range(4):
        assert schema[i]["kind"] == "numeric"
    assert schema[4]["kind"] == "categorical"


def test_mlflow_logging(synthetic_train_val_data, config_file, tmp_path, monkeypatch):
    """Test that MLflow logging works correctly."""
    import os
    
    train_path, val_path = synthetic_train_val_data
    out_dir = tmp_path / "bundles"
    
    # Set MLflow tracking URI to tmp_path sqlite
    mlflow_uri = f"sqlite:///{tmp_path}/test_mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    
    result = train_model(
        config_path=config_file,
        model_type="logistic",
        class_weight=True,
        seed=99,
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
    )
    
    # Verify run_id returned
    assert "run_id" in result
    assert result["run_id"] is not None
    
    # Load MLflow run to verify logging
    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)
    
    run = mlflow.get_run(result["run_id"])
    
    # Check params
    assert run.data.params["model_type"] == "logistic"
    assert run.data.params["seed"] == "99"
    assert run.data.params["class_weight"] == "True"
    assert "logistic.C" in run.data.params
    
    # Check metrics have val_ prefix
    assert "val_average_precision" in run.data.metrics
    assert "val_roc_auc" in run.data.metrics
    assert "val_precision" in run.data.metrics
    assert "val_recall" in run.data.metrics
    assert "val_f1" in run.data.metrics
    
    # Check tag
    assert "model_version" in run.data.tags
    assert "logistic-" in run.data.tags["model_version"]
    assert "-seed99" in run.data.tags["model_version"]
    
    # Check artifacts
    client = mlflow.tracking.MlflowClient(mlflow_uri)
    artifacts = client.list_artifacts(result["run_id"])
    artifact_paths = [a.path for a in artifacts]

    assert "bundle" in artifact_paths
    assert "config" in artifact_paths
