"""Small fitted models only: no artifacts and no real validation/test files."""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.ml.explainability import (
    explain_global,
    explain_one,
    sample_background,
)
from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.bundle import LoadedBundle, create_feature_schema


@pytest.fixture
def trained():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "customerID": [f"train-{i}" for i in range(260)],
            "strong_signal": rng.normal(size=260),
            "weak_signal": rng.normal(size=260),
            "contract_type": rng.choice(
                ["month_to_month", "one_year", "two_year"], 260
            ),
        }
    )
    labels = (train.strong_signal + rng.normal(scale=0.2, size=260) > 0).astype(int)
    config = {
        "numeric_features": ["strong_signal", "weak_signal"],
        "categorical_features": ["contract_type"],
        "logistic": {"C": 1.0, "max_iter": 500},
    }
    pipeline = train_logistic(
        train.drop(columns="customerID"), labels, config, 42, False
    )
    schema = create_feature_schema(
        train, config["numeric_features"], config["categorical_features"]
    )
    bundle = LoadedBundle(
        pipeline,
        {
            "model_version": "logistic-test",
            "model_type": "logistic",
            "class_weight": False,
        },
        schema,
    )
    return bundle, train


def test_additivity_probability_and_onehot_grouping(trained):
    bundle, train = trained
    row = train.iloc[[4]].copy()
    row["customerID"] = "val-1"
    result = ModelExplanation.model_validate(explain_one(bundle, row, train))
    expected_logit = float(
        bundle._pipeline.decision_function(row.drop(columns="customerID"))[0]
    )
    assert result.base_value + sum(
        c.contribution for c in result.contributions
    ) == pytest.approx(expected_logit, abs=1e-6, rel=0)
    assert result.log_odds == pytest.approx(expected_logit, abs=1e-6, rel=0)
    assert 1 / (1 + np.exp(-result.log_odds)) == pytest.approx(
        bundle.predict_proba(row.drop(columns="customerID"))[0], abs=1e-6, rel=0
    )
    for contribution in result.contributions:
        details = [
            d
            for d in result.detailed_contributions
            if d.source_feature == contribution.feature_name
        ]
        assert sum(d.contribution for d in details) == pytest.approx(
            contribution.contribution, abs=1e-6, rel=0
        )
    category_details = [
        d for d in result.detailed_contributions if d.source_feature == "contract_type"
    ]
    assert (
        len(category_details) == 3
    )  # Both feature and category names contain underscores.
    assert result.customer_id == "val-1"


def test_global_is_stable_and_groups_before_absolute_value(trained):
    bundle, train = trained
    first = explain_global(bundle, train)
    assert first == explain_global(bundle, train)
    assert first["feature_importance"][0]["feature_name"] == "strong_signal"
    sample = sample_background(train)
    local = [explain_one(bundle, sample.iloc[[i]], train) for i in range(8)]
    # Check the aggregation rule independently over the SAME eight-row background.
    subset = sample.iloc[:8]
    result = explain_global(bundle, subset)
    contributions = [explain_one(bundle, subset.iloc[[i]], subset) for i in range(8)]
    for importance in result["feature_importance"]:
        values = [
            next(
                c["contribution"]
                for c in r["contributions"]
                if c["feature_name"] == importance["feature_name"]
            )
            for r in contributions
        ]
        assert importance["mean_abs_shap"] == pytest.approx(
            np.mean(np.abs(values)), abs=1e-9
        )
    assert all(r["background"]["rows"] == 200 for r in local)


def test_background_seed_cap_and_train_only_ids(trained):
    bundle, train = trained
    val_ids = {f"val-{i}" for i in range(20)}
    test_ids = {f"test-{i}" for i in range(20)}  # Synthetic IDs; no test set is read.
    result = explain_global(bundle, train)
    selected = result["background"]["customer_ids"]
    assert selected == train.sample(n=200, random_state=42).customerID.tolist()
    assert len(selected) == 200
    assert set(selected) <= set(train.customerID)
    assert not set(selected) & (val_ids | test_ids)
    expected_base = bundle._pipeline.decision_function(
        sample_background(train).drop(columns="customerID")
    ).mean()
    assert result["base_value"] == pytest.approx(expected_base, abs=1e-9)


@pytest.mark.parametrize("split", ["val", "test"])
def test_rejects_explicit_non_train_background(trained, split):
    bundle, train = trained
    train.attrs["source_split"] = split
    with pytest.raises(ValueError, match="only from train"):
        explain_global(bundle, train)
    train.attrs.clear()
    train["split"] = split
    with pytest.raises(ValueError, match="outside the train"):
        explain_one(bundle, train.iloc[[0]], train)


def test_explaining_never_fits_preprocessing_or_model(trained, monkeypatch):
    bundle, train = trained

    def forbid(*args, **kwargs):
        pytest.fail("Explanation attempted to fit")

    pipeline = bundle._pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    for estimator in [
        pipeline,
        preprocessor,
        pipeline.named_steps["model"],
        *[
            step
            for _, transformer, _ in preprocessor.transformers_
            for _, step in transformer.steps
        ],
    ]:
        monkeypatch.setattr(estimator, "fit", forbid)
        if hasattr(estimator, "fit_transform"):
            monkeypatch.setattr(estimator, "fit_transform", forbid)
    explain_global(bundle, train)
    explain_one(bundle, train.iloc[[0]], train)


def test_missing_numeric_value_and_unknown_category(trained):
    bundle, train = trained
    row = train.iloc[[0]].copy()
    row["strong_signal"] = np.nan
    row["contract_type"] = "previously_unseen"
    result = ModelExplanation.model_validate(explain_one(bundle, row, train))
    assert result.contributions[0].feature_value == "(missing; imputed)"


def test_rejects_unsupported_models(trained):
    bundle, train = trained
    weighted = copy.deepcopy(bundle)
    weighted._pipeline.named_steps["model"].class_weight = "balanced"
    with pytest.raises(ValueError, match="no-class-weight"):
        explain_global(weighted, train)
    calibrated = copy.deepcopy(bundle)
    calibrated._pipeline.steps[-1] = ("model", CalibratedClassifierCV())
    with pytest.raises(TypeError, match="uncalibrated"):
        explain_global(calibrated, train)
    bundle.metadata["model_type"] = "catboost"
    with pytest.raises(NotImplementedError, match="chưa hỗ trợ"):
        explain_global(bundle, train)


def test_rejects_empty_background_or_multiple_customers(trained):
    bundle, train = trained
    with pytest.raises(ValueError, match="train rows"):
        explain_global(bundle, train.iloc[:0])
    with pytest.raises(ValueError, match="exactly one"):
        explain_one(bundle, train.iloc[:2], train)
