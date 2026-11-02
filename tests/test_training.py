import json
import time

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.catboost_model import train_catboost
from churn_retention.ml.training.compare import compare_runs
from churn_retention.ml.training.metrics import compute_metrics
from churn_retention.ml.training.run_log import log_training_run


@pytest.fixture
def synthetic_data():
    """Generate synthetic imbalanced dataset for testing."""
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
    df["contract_type"] = np.random.choice(["month_to_month", "one_year", "two_year"], size=len(df))

    # Add some missing values
    df.loc[0:10, "num_1"] = np.nan
    df.loc[20:25, "contract_type"] = np.nan

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Add unknown category in validation only
    X_val = X_val.copy()
    X_val.loc[X_val.index[0], "contract_type"] = "unknown_category"

    return X_train, X_val, y_train, y_val


@pytest.fixture
def config():
    """Training configuration."""
    return {
        "numeric_features": ["num_1", "num_2", "num_3", "num_4"],
        "categorical_features": ["contract_type"],
        "logistic": {
            "C": 1.0,
            "max_iter": 1000,
        },
        "catboost": {
            "iterations": 50,
            "depth": 3,
            "learning_rate": 0.1,
        },
        "evaluation": {
            "threshold": 0.5,
        },
    }


def test_compute_metrics_hand_calculated():
    """Test compute_metrics with hand-calculated example."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    # y_pred = [0, 0, 1, 1]
    # precision = 2/2 = 1.0
    # recall = 2/2 = 1.0
    # f1 = 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_compute_metrics_missing_class_raises_error():
    """Test that compute_metrics raises error when y_true has only one class."""
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6])

    with pytest.raises(ValueError) as exc_info:
        compute_metrics(y_true, y_prob)

    assert "must contain both classes" in str(exc_info.value)


def test_logistic_same_seed_same_results(synthetic_data, config):
    """Test that same seed produces same predictions."""
    X_train, X_val, y_train, _ = synthetic_data

    pipeline1 = train_logistic(X_train, y_train, config, seed=42, class_weight=False)
    y_prob1 = pipeline1.predict_proba(X_val)[:, 1]

    pipeline2 = train_logistic(X_train, y_train, config, seed=42, class_weight=False)
    y_prob2 = pipeline2.predict_proba(X_val)[:, 1]

    assert_allclose(y_prob1, y_prob2, rtol=1e-5)


def test_catboost_same_seed_same_results(synthetic_data, config):
    """Test that same seed produces same predictions."""
    X_train, X_val, y_train, _ = synthetic_data

    pipeline1 = train_catboost(X_train, y_train, config, seed=42, class_weight=False)
    y_prob1 = pipeline1.predict_proba(X_val)[:, 1]

    pipeline2 = train_catboost(X_train, y_train, config, seed=42, class_weight=False)
    y_prob2 = pipeline2.predict_proba(X_val)[:, 1]

    assert_allclose(y_prob1, y_prob2, rtol=1e-5)


def test_logistic_handles_missing_and_unknown(synthetic_data, config):
    """Test that logistic pipeline handles missing values and unknown categories."""
    X_train, X_val, y_train, y_val = synthetic_data

    pipeline = train_logistic(X_train, y_train, config, seed=42, class_weight=False)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    assert len(y_prob) == len(y_val)
    assert all(0 <= p <= 1 for p in y_prob)
    assert not np.any(np.isnan(y_prob))


def test_catboost_handles_missing_and_unknown(synthetic_data, config):
    """Test that catboost pipeline handles missing values and unknown categories."""
    X_train, X_val, y_train, y_val = synthetic_data

    pipeline = train_catboost(X_train, y_train, config, seed=42, class_weight=False)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    assert len(y_prob) == len(y_val)
    assert all(0 <= p <= 1 for p in y_prob)
    assert not np.any(np.isnan(y_prob))


def test_logistic_class_weight_parameter(synthetic_data, config):
    """Test that class_weight parameter is applied correctly."""
    X_train, X_val, y_train, _ = synthetic_data

    pipeline_unweighted = train_logistic(X_train, y_train, config, seed=42, class_weight=False)
    pipeline_weighted = train_logistic(X_train, y_train, config, seed=42, class_weight=True)

    y_prob_unweighted = pipeline_unweighted.predict_proba(X_val)[:, 1]
    y_prob_weighted = pipeline_weighted.predict_proba(X_val)[:, 1]

    # Probabilities should be different
    assert not np.allclose(y_prob_unweighted, y_prob_weighted)


def test_catboost_class_weight_parameter(synthetic_data, config):
    """Test that class_weight parameter is applied correctly."""
    X_train, X_val, y_train, _ = synthetic_data

    pipeline_unweighted = train_catboost(X_train, y_train, config, seed=42, class_weight=False)
    pipeline_weighted = train_catboost(X_train, y_train, config, seed=42, class_weight=True)

    y_prob_unweighted = pipeline_unweighted.predict_proba(X_val)[:, 1]
    y_prob_weighted = pipeline_weighted.predict_proba(X_val)[:, 1]

    # Probabilities should be different
    assert not np.allclose(y_prob_unweighted, y_prob_weighted)


def test_log_training_run_creates_file(tmp_path):
    """Test that log_training_run creates a JSON file."""
    run_info = {
        "model_name": "logistic",
        "config": {"C": 1.0},
        "metrics": {"roc_auc": 0.85},
        "dataset_id": "synthetic-demo-v1",
        "split_id": "split-001",
        "evaluation_split": "validation",
        "duration_seconds": 1.5,
    }

    output_path = log_training_run(run_info, tmp_path)

    assert output_path.exists()
    assert output_path.suffix == ".json"

    # Read and verify
    with open(output_path) as f:
        logged_data = json.load(f)

    assert "run_id" in logged_data
    assert "created_at" in logged_data
    assert logged_data["model_name"] == "logistic"
    assert logged_data["metrics"]["roc_auc"] == 0.85


def test_log_training_run_two_calls_create_two_files(tmp_path):
    """Test that two log calls create two separate files."""
    run_info_1 = {
        "model_name": "logistic",
        "config": {},
        "metrics": {},
        "dataset_id": "synthetic-demo-v1",
        "split_id": "split-001",
        "evaluation_split": "validation",
        "duration_seconds": 1.0,
    }

    run_info_2 = {
        "model_name": "catboost",
        "config": {},
        "metrics": {},
        "dataset_id": "synthetic-demo-v1",
        "split_id": "split-001",
        "evaluation_split": "validation",
        "duration_seconds": 2.0,
    }

    # Ensure different UUIDs by adding small delay
    path1 = log_training_run(run_info_1, tmp_path)
    time.sleep(0.01)
    path2 = log_training_run(run_info_2, tmp_path)

    assert path1 != path2
    assert path1.exists()
    assert path2.exists()

    # Verify both can be read
    with open(path1) as f:
        data1 = json.load(f)
    with open(path2) as f:
        data2 = json.load(f)

    assert data1["model_name"] == "logistic"
    assert data2["model_name"] == "catboost"


def test_compare_runs(tmp_path):
    """Test that compare_runs aggregates and sorts by average_precision."""
    run_info_1 = {
        "model_name": "logistic",
        "config": {},
        "metrics": {"average_precision": 0.75, "roc_auc": 0.70},
        "dataset_id": "synthetic-demo-v1",
        "split_id": "split-001",
        "evaluation_split": "validation",
        "duration_seconds": 1.0,
    }

    run_info_2 = {
        "model_name": "catboost",
        "config": {},
        "metrics": {"average_precision": 0.85, "roc_auc": 0.80},
        "dataset_id": "synthetic-demo-v1",
        "split_id": "split-001",
        "evaluation_split": "validation",
        "duration_seconds": 2.0,
    }

    path1 = log_training_run(run_info_1, tmp_path)
    time.sleep(0.01)
    path2 = log_training_run(run_info_2, tmp_path)

    df = compare_runs([path1, path2])

    assert len(df) == 2
    assert df.iloc[0]["model_name"] == "catboost"  # Higher AP
    assert df.iloc[1]["model_name"] == "logistic"  # Lower AP
    assert df.iloc[0]["metrics.average_precision"] == 0.85


def test_missing_columns_raises_error(config):
    """Test that missing columns in X_train raises ValueError."""
    X_train = pd.DataFrame({"num_1": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])

    with pytest.raises(ValueError) as exc_info:
        train_logistic(X_train, y_train, config, seed=42, class_weight=False)

    assert "Missing columns" in str(exc_info.value)


def test_catboost_column_order_invariance():
    """R08: CatBoost must select and order columns per config, making predictions
    invariant to input column order. Extra columns are ignored.
    """
    # Train with columns in config order
    X_train = pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0],
            "num_b": [5.0, 6.0, 7.0, 8.0],
            "cat_x": ["p", "q", "p", "q"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1])
    cfg = {
        "numeric_features": ["num_a", "num_b"],
        "categorical_features": ["cat_x"],
        "catboost": {"iterations": 10, "depth": 2, "learning_rate": 0.1},
    }
    pipeline = train_catboost(X_train, y_train, cfg, seed=42, class_weight=False)

    # Test with same data, different column order
    X_test_ordered = pd.DataFrame(
        {
            "num_a": [1.5, 3.5],
            "num_b": [5.5, 7.5],
            "cat_x": ["p", "q"],
        }
    )
    X_test_shuffled = pd.DataFrame(
        {
            "cat_x": ["p", "q"],
            "num_b": [5.5, 7.5],
            "num_a": [1.5, 3.5],
        }
    )

    y_prob_ordered = pipeline.predict_proba(X_test_ordered)[:, 1]
    y_prob_shuffled = pipeline.predict_proba(X_test_shuffled)[:, 1]

    assert_allclose(y_prob_ordered, y_prob_shuffled, rtol=1e-5)


def test_catboost_ignores_extra_columns():
    """R08: CatBoost must ignore extra columns not in config."""
    X_train = pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0],
            "cat_x": ["p", "q", "p", "q"],
            "extra_col": ["ignored", "ignored", "ignored", "ignored"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1])
    cfg = {
        "numeric_features": ["num_a"],
        "categorical_features": ["cat_x"],
        "catboost": {"iterations": 10, "depth": 2, "learning_rate": 0.1},
    }
    pipeline = train_catboost(X_train, y_train, cfg, seed=42, class_weight=False)

    X_test_with_extra = pd.DataFrame(
        {
            "num_a": [1.5, 3.5],
            "cat_x": ["p", "q"],
            "extra_col": ["also_ignored", "also_ignored"],
        }
    )
    X_test_without_extra = pd.DataFrame(
        {
            "num_a": [1.5, 3.5],
            "cat_x": ["p", "q"],
        }
    )

    y_prob_with = pipeline.predict_proba(X_test_with_extra)[:, 1]
    y_prob_without = pipeline.predict_proba(X_test_without_extra)[:, 1]

    assert_allclose(y_prob_with, y_prob_without, rtol=1e-5)


def test_catboost_probabilities_unchanged_after_refactor():
    """R15: After fixing column selection and removing duplicate fit_transform,
    probabilities on fixed seed data must not change from before the fix.

    This test records the expected probabilities from the CURRENT (pre-fix) code
    and ensures the fix doesn't change the actual predictions.
    """
    X_train = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    cfg = {
        "numeric_features": ["x1", "x2"],
        "categorical_features": ["cat"],
        "catboost": {"iterations": 20, "depth": 2, "learning_rate": 0.1},
    }

    pipeline = train_catboost(X_train, y_train, cfg, seed=99, class_weight=False)

    X_test = pd.DataFrame(
        {
            "x1": [1.5, 4.5],
            "x2": [15.0, 45.0],
            "cat": ["a", "b"],
        }
    )

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Expected probabilities from current implementation (before fix)
    # Run this test now to capture baseline, then ensure it doesn't change after fix
    # These values were obtained by running the current code
    expected = np.array([0.35785969911029286, 0.6498951514485928])

    assert_allclose(y_prob, expected, rtol=1e-5)
