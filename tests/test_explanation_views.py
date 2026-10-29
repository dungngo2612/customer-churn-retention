from pathlib import Path

import pandas as pd
import pytest

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.entrypoints.streamlit.explanation_views import (
    customer_ids,
    global_importance_frame,
    load_train_background,
    local_contribution_frame,
    scoring_context,
    select_customer,
)
from churn_retention.ml.training.bundle import LoadedBundle


def test_scoring_context_keeps_same_bundle_and_snapshots_data():
    bundle = LoadedBundle(None, {"model_version": "v1"}, [])
    frame = pd.DataFrame({"customer_id": ["a", "b"], "value": [1, 2]})
    context = scoring_context(bundle, frame, "artifacts/v1")
    assert context.bundle is bundle
    assert Path(context.bundle_path).is_absolute()
    frame.loc[0, "value"] = 99
    assert select_customer(context.dataset, "a").value.tolist() == [1]
    assert customer_ids(context.dataset) == ["a", "b"]


def test_background_loader_rejects_val_test_before_reading(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("Must not read validation/test background")

    monkeypatch.setattr(pd, "read_csv", forbid)
    for name in ["val.csv", "test.csv"]:
        with pytest.raises(ValueError, match="train.csv"):
            load_train_background(tmp_path / name)
    (tmp_path / "train.csv").symlink_to(tmp_path / "val.csv")
    with pytest.raises(ValueError, match="train.csv"):
        load_train_background(tmp_path / "train.csv")


def test_background_loader_marks_train_provenance(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame({"customerID": ["train-a"]}).to_csv(path, index=False)
    frame = load_train_background(path)
    assert frame.attrs["source_split"] == "train"
    assert frame.customerID.tolist() == ["train-a"]


def test_select_customer_requires_unique_ids():
    with pytest.raises(ValueError, match="unique"):
        customer_ids(pd.DataFrame({"customerID": ["a", "a"]}))
    with pytest.raises(ValueError, match="Không tìm"):
        select_customer(pd.DataFrame({"customerID": ["a"]}), "b")


@pytest.fixture
def explanation_result():
    return ModelExplanation.model_validate(
        {
            "customer_id": "a",
            "model_version": "v1",
            "output_type": "log_odds",
            "calibration_applied": "original",
            "base_value": -1.0,
            "log_odds": 0.0,
            "probability": 0.5,
            "background": {"seed": 42, "rows": 1, "customer_ids": ["train-a"]},
            "feature_schema": [
                {"name": name, "kind": "numeric", "dtype": "float64"}
                for name in ["x", "y"]
            ],
            "contributions": [
                {
                    "customer_id": "a",
                    "model_version": "v1",
                    "feature_name": name,
                    "feature_value": 1.0,
                    "contribution": value,
                    "base_value": -1.0,
                    "explained_output": "log_odds",
                    "calibration_applied": "original",
                }
                for name, value in [("x", 2.0), ("y", -1.0)]
            ],
            "detailed_contributions": [
                {
                    "feature_name": "num__" + name,
                    "source_feature": name,
                    "transformed_value": 1.0,
                    "contribution": value,
                }
                for name, value in [("x", 2.0), ("y", -1.0)]
            ],
        }
    )


def test_pure_chart_functions(explanation_result):
    frame = local_contribution_frame(explanation_result)
    assert frame.feature.tolist() == ["x", "y"]
    assert frame.direction.tolist() == ["Tăng log-odds", "Giảm log-odds"]
    assert frame.contribution_log_odds.tolist() == [2.0, -1.0]
    global_frame = global_importance_frame(
        {"feature_importance": [{"feature_name": "x", "mean_abs_shap": 2.0}]}
    )
    assert global_frame.loc["x", "mean_abs_shap"] == 2.0


def test_local_contribution_values_are_strings_for_mixed_types(explanation_result):
    explanation_result.contributions[0].feature_value = 18.0
    explanation_result.contributions[1].feature_value = "Fiber optic"

    frame = local_contribution_frame(explanation_result)

    assert frame["value"].tolist() == ["18.0", "Fiber optic"]
    assert frame["value"].map(lambda value: isinstance(value, str)).all()
    assert pd.api.types.is_string_dtype(frame["value"])


def test_predictor_bundle_property_requires_loaded_model():
    with pytest.raises(RuntimeError, match="chưa load"):
        _ = BundlePredictor().loaded_bundle
