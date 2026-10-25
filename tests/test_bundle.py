import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.bundle import (
    create_feature_schema,
    load_bundle,
    save_bundle,
)
from churn_retention.ml.training.catboost_model import train_catboost
from churn_retention.modules.scoring.ports import Predictor


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
        "model_type": request.param,
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


@pytest.mark.parametrize("operation", ["model_version", "predict_batch"])
def test_bundle_predictor_requires_load(operation: str) -> None:
    predictor: Predictor = BundlePredictor()

    with pytest.raises(RuntimeError, match="^chưa load bundle$"):
        if operation == "model_version":
            _ = predictor.model_version
        else:
            predictor.predict_batch(pd.DataFrame(), [])


def test_bundle_predictor_model_version(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, _, _, metadata = demo_bundle
    predictor: Predictor = BundlePredictor()
    predictor.load(str(bundle_dir))

    assert predictor.model_version == metadata["model_version"]


def test_bundle_predictor_preserves_customer_order(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, pipeline, features, _ = demo_bundle
    predictor: Predictor = BundlePredictor()
    predictor.load(str(bundle_dir))
    customer_ids = [f"demo-customer-{i}" for i in range(len(features))]
    expected_by_id = dict(
        zip(customer_ids, pipeline.predict_proba(features)[:, 1], strict=True)
    )
    order = [3, 0, 5, 1, 7, 2, 6, 4]
    ordered_ids = tuple(customer_ids[i] for i in order)
    ordered_features = features.iloc[order, ::-1].copy()
    original = ordered_features.copy(deep=True)

    probabilities = predictor.predict_batch(ordered_features, ordered_ids)

    assert isinstance(probabilities, list)
    assert len(probabilities) == len(ordered_ids)
    assert all(isinstance(probability, float) for probability in probabilities)
    assert np.allclose(
        probabilities, [expected_by_id[customer_id] for customer_id in ordered_ids]
    )
    pd.testing.assert_frame_equal(ordered_features, original)


def test_bundle_predictor_rejects_length_mismatch(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, _, features, _ = demo_bundle
    predictor: Predictor = BundlePredictor()
    predictor.load(str(bundle_dir))

    with pytest.raises(
        ValueError, match="Length mismatch: 1 customer_ids vs 8 features"
    ):
        predictor.predict_batch(features, ["demo-customer"])


def test_bundle_predictor_empty_batch(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]],
) -> None:
    bundle_dir, _, _, _ = demo_bundle
    predictor: Predictor = BundlePredictor()
    predictor.load(str(bundle_dir))

    assert predictor.predict_batch(pd.DataFrame(), []) == []


def test_save_bundle_refuses_overwrite_existing(tmp_path: Path) -> None:
    """R03: saving a bundle twice to the same path must fail; first bundle unchanged."""

    from churn_retention.ml.training.baseline import train_logistic

    features = pd.DataFrame(
        {"demo_x": [1.0, 2.0, 3.0, 4.0], "demo_cat": ["a", "b", "a", "b"]}
    )
    target = pd.Series([0, 1, 0, 1])
    cfg = {
        "numeric_features": ["demo_x"],
        "categorical_features": ["demo_cat"],
        "logistic": {"C": 1.0, "max_iter": 100},
    }
    pipeline = train_logistic(features, target, cfg, seed=42, class_weight=False)
    schema = create_feature_schema(features, ["demo_x"], ["demo_cat"])
    metadata = {"model_version": "logistic-20261017T120000-seed42"}

    bundle_dir = tmp_path / "logistic-20261017T120000-seed42"
    save_bundle(pipeline, bundle_dir, schema, metadata)

    # Record the original contents
    original_metadata = (bundle_dir / "metadata.json").read_bytes()
    original_model = (bundle_dir / "model.pkl").read_bytes()

    # Second save to same path must raise
    with pytest.raises(FileExistsError, match=str(bundle_dir)):
        save_bundle(pipeline, bundle_dir, schema, metadata)

    # First bundle is unmodified
    assert (bundle_dir / "metadata.json").read_bytes() == original_metadata
    assert (bundle_dir / "model.pkl").read_bytes() == original_model


def test_bundle_reload_in_fresh_python_process(
    demo_bundle: tuple[Path, Pipeline, pd.DataFrame, dict[str, Any]], tmp_path: Path
) -> None:
    bundle_dir, pipeline, features, metadata = demo_bundle
    ordered_features = features.iloc[[3, 0, 5, 1, 7, 2, 6, 4]]
    input_path = tmp_path / "demo-features.csv"
    ordered_features.iloc[:, ::-1].to_csv(input_path, index=False)
    expected = pipeline.predict_proba(ordered_features)[:, 1]
    script = """
import json
import os
import sys

import pandas as pd

from churn_retention.ml.training.bundle import load_bundle

bundle = load_bundle(sys.argv[1])
features = pd.read_csv(sys.argv[2])
print(json.dumps({
    "pid": os.getpid(),
    "model_version": bundle.model_version,
    "probabilities": bundle.predict_proba(features).tolist(),
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle_dir), str(input_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    reloaded = json.loads(result.stdout)

    assert reloaded["pid"] != os.getpid()
    assert reloaded["model_version"] == metadata["model_version"]
    assert len(reloaded["probabilities"]) == len(ordered_features)
    assert np.allclose(reloaded["probabilities"], expected)


def test_load_bundle_validates_metadata_and_schema(tmp_path: Path) -> None:
    """R05: load_bundle must validate metadata has required keys and correct types,
    and feature_schema has correct structure (name/kind/dtype, kind in numeric|categorical).
    """
    from churn_retention.ml.training.baseline import train_logistic

    features = pd.DataFrame({"x": [1.0, 2.0], "cat": ["a", "b"]})
    target = pd.Series([0, 1])
    cfg = {
        "numeric_features": ["x"],
        "categorical_features": ["cat"],
        "logistic": {"C": 1.0, "max_iter": 100},
    }
    pipeline = train_logistic(features, target, cfg, seed=42, class_weight=False)
    schema = create_feature_schema(features, ["x"], ["cat"])

    # Valid bundle
    metadata = {
        "model_version": "valid-bundle",
        "model_type": "logistic",
        "training": {"created_at": "2026-10-17T00:00:00Z"},
    }
    bundle_dir = tmp_path / "valid"
    save_bundle(pipeline, bundle_dir, schema, metadata)
    # Should not raise
    loaded = load_bundle(bundle_dir)
    assert loaded.model_version == "valid-bundle"

    # Missing model_version
    bundle_dir2 = tmp_path / "missing_version"
    bad_meta = {"model_type": "logistic"}
    save_bundle(pipeline, bundle_dir2, schema, bad_meta)
    with pytest.raises(ValueError, match="model_version.*" + str(bundle_dir2)):
        load_bundle(bundle_dir2)

    # Missing model_type
    bundle_dir3 = tmp_path / "missing_type"
    bad_meta2 = {"model_version": "v1"}
    save_bundle(pipeline, bundle_dir3, schema, bad_meta2)
    with pytest.raises(ValueError, match="model_type.*" + str(bundle_dir3)):
        load_bundle(bundle_dir3)

    # Wrong type for model_version
    bundle_dir4 = tmp_path / "wrong_type_version"
    bad_meta3 = {"model_version": 123, "model_type": "logistic"}
    save_bundle(pipeline, bundle_dir4, schema, bad_meta3)
    with pytest.raises(ValueError, match="model_version.*str.*" + str(bundle_dir4)):
        load_bundle(bundle_dir4)

    # Invalid feature_schema: missing 'name'
    bundle_dir5 = tmp_path / "invalid_schema_name"
    bad_schema = [{"kind": "numeric", "dtype": "float64"}]
    save_bundle(pipeline, bundle_dir5, bad_schema, metadata)
    with pytest.raises(ValueError, match="name.*" + str(bundle_dir5)):
        load_bundle(bundle_dir5)

    # Invalid feature_schema: missing 'kind'
    bundle_dir6 = tmp_path / "invalid_schema_kind"
    bad_schema2 = [{"name": "x", "dtype": "float64"}]
    save_bundle(pipeline, bundle_dir6, bad_schema2, metadata)
    with pytest.raises(ValueError, match="kind.*" + str(bundle_dir6)):
        load_bundle(bundle_dir6)

    # Invalid feature_schema: wrong kind value
    bundle_dir7 = tmp_path / "invalid_kind_value"
    bad_schema3 = [{"name": "x", "kind": "invalid_kind", "dtype": "float64"}]
    save_bundle(pipeline, bundle_dir7, bad_schema3, metadata)
    with pytest.raises(
        ValueError, match="kind.*numeric.*categorical.*" + str(bundle_dir7)
    ):
        load_bundle(bundle_dir7)


def test_load_bundle_warns_on_version_mismatch(tmp_path: Path) -> None:
    """R05: when model_version in metadata differs from directory name, warn but don't error."""
    from churn_retention.ml.training.baseline import train_logistic

    features = pd.DataFrame({"x": [1.0, 2.0]})
    target = pd.Series([0, 1])
    cfg = {
        "numeric_features": ["x"],
        "categorical_features": [],
        "logistic": {"C": 1.0, "max_iter": 100},
    }
    pipeline = train_logistic(features, target, cfg, seed=42, class_weight=False)
    schema = create_feature_schema(features, ["x"], [])
    metadata = {"model_version": "v1-mismatch", "model_type": "logistic"}

    bundle_dir = tmp_path / "v2-dirname"
    save_bundle(pipeline, bundle_dir, schema, metadata)

    with pytest.warns(UserWarning, match="model_version.*v1-mismatch.*v2-dirname"):
        loaded = load_bundle(bundle_dir)
        assert loaded.model_version == "v1-mismatch"


def test_predict_proba_empty_dataframe(tmp_path: Path) -> None:
    """R11: LoadedBundle.predict_proba must return empty array for empty DataFrame (with correct columns),
    for both logistic and catboost models.
    """
    from churn_retention.ml.training.baseline import train_logistic
    from churn_retention.ml.training.catboost_model import train_catboost

    features = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    target = pd.Series([0, 1, 0, 1])
    cfg = {
        "numeric_features": ["x"],
        "categorical_features": ["cat"],
        "logistic": {"C": 1.0, "max_iter": 100},
        "catboost": {"iterations": 10, "depth": 2, "learning_rate": 0.1},
    }

    for model_type, trainer in [
        ("logistic", train_logistic),
        ("catboost", train_catboost),
    ]:
        pipeline = trainer(features, target, cfg, seed=42, class_weight=False)
        schema = create_feature_schema(features, ["x"], ["cat"])
        metadata = {"model_version": f"{model_type}-empty", "model_type": model_type}

        bundle_dir = tmp_path / f"empty-{model_type}"
        save_bundle(pipeline, bundle_dir, schema, metadata)
        loaded = load_bundle(bundle_dir)

        # Empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=["x", "cat"])
        result = loaded.predict_proba(empty_df)

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == np.float64
