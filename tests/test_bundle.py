from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.bundle import (
    create_feature_schema,
    load_bundle,
    save_bundle,
)
from churn_retention.ml.training.catboost_model import train_catboost


@pytest.fixture(params=["logistic", "catboost"])
def demo_bundle(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]]:
    """Fit and save a small demo bundle; these data are not project metrics."""
    features = pd.DataFrame(
        {
            "demo_usage": [1.0, 3.0, 2.0, 8.0, 9.0, 7.0, 4.0, 10.0],
            "demo_months": [10.0, 8.0, 9.0, 2.0, 1.0, 3.0, 7.0, 4.0],
            "demo_plan": ["a", "b", "a", "b", "a", "b", "a", "b"],
        },
        index=["C08", "C02", "C07", "C01", "C06", "C03", "C05", "C04"],
    )
    target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1], index=features.index)
    numeric_cols = ["demo_usage", "demo_months"]
    categorical_cols = ["demo_plan"]
    config = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "logistic": {"C": 1.0, "max_iter": 100},
        "catboost": {"iterations": 8, "depth": 2, "learning_rate": 0.1},
    }
    trainer = train_logistic if request.param == "logistic" else train_catboost
    pipeline = trainer(features, target, config, seed=42, class_weight=False)
    version = f"{request.param}-20261017T000000-seed42"
    metadata = {
        "model_version": version,
        "label": {"target_col": "demo_outcome", "positive_label": "demo_left"},
    }
    bundle_dir = tmp_path / version
    save_bundle(
        pipeline,
        bundle_dir,
        create_feature_schema(features, numeric_cols, categorical_cols),
        metadata,
    )
    return bundle_dir, pipeline, features, metadata


def test_load_bundle_predictions_and_metadata(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, pipeline, features, metadata = demo_bundle
    bundle = load_bundle(bundle_dir)

    assert bundle.model_version == bundle_dir.name
    assert bundle.metadata == metadata
    assert bundle.feature_schema == create_feature_schema(
        features, ["demo_usage", "demo_months"], ["demo_plan"]
    )
    probabilities = bundle.predict_proba(features)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(features),)
    assert np.allclose(probabilities, pipeline.predict_proba(features)[:, 1])


def test_load_bundle_reorders_columns_and_ignores_extras(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, pipeline, features, _ = demo_bundle
    shuffled = features[["demo_plan", "demo_months", "demo_usage"]].copy()
    shuffled["demo_extra"] = "ignored"
    original = shuffled.copy(deep=True)

    with pytest.warns(UserWarning, match="Ignoring extra columns.*demo_extra"):
        probabilities = load_bundle(bundle_dir).predict_proba(shuffled)

    assert np.allclose(probabilities, pipeline.predict_proba(features)[:, 1])
    pd.testing.assert_frame_equal(shuffled, original)


def test_load_bundle_missing_column_names_error(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, _, features, _ = demo_bundle
    with pytest.raises(ValueError, match="Missing required columns.*demo_months"):
        load_bundle(bundle_dir).predict_proba(features.drop(columns="demo_months"))


def test_load_bundle_rejects_non_numeric_values(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, _, features, _ = demo_bundle
    invalid = features.copy()
    invalid["demo_usage"] = invalid["demo_usage"].astype(object)
    invalid.iloc[0, invalid.columns.get_loc("demo_usage")] = "invalid-demo-value"

    with pytest.raises(ValueError, match="demo_usage.*non-numeric.*invalid-demo-value"):
        load_bundle(bundle_dir).predict_proba(invalid)


def test_load_bundle_leaves_numeric_nan_to_preprocessing(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, pipeline, features, _ = demo_bundle
    features = features.copy()
    features.iloc[0, features.columns.get_loc("demo_usage")] = np.nan

    probabilities = load_bundle(bundle_dir).predict_proba(features)

    assert np.allclose(probabilities, pipeline.predict_proba(features)[:, 1])
