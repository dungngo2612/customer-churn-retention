from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from churn_retention.contracts.http.scoring import (
    Explanation,
    ExplanationList,
    Prediction,
    PredictionBatch,
)


def test_prediction_valid():
    p = Prediction(
        customer_id="C001",
        dataset_id="ds1",
        scoring_run_id="run1",
        churn_probability=0.5,
        model_version="v1",
        scored_at=datetime.now(timezone.utc),
    )
    assert p.churn_probability == 0.5


def test_prediction_probability_at_bounds():
    p0 = Prediction(
        customer_id="C001",
        dataset_id="ds1",
        scoring_run_id="run1",
        churn_probability=0.0,
        model_version="v1",
        scored_at=datetime.now(timezone.utc),
    )
    assert p0.churn_probability == 0.0

    p1 = Prediction(
        customer_id="C001",
        dataset_id="ds1",
        scoring_run_id="run1",
        churn_probability=1.0,
        model_version="v1",
        scored_at=datetime.now(timezone.utc),
    )
    assert p1.churn_probability == 1.0


def test_prediction_probability_below_zero():
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            customer_id="C001",
            dataset_id="ds1",
            scoring_run_id="run1",
            churn_probability=-0.1,
            model_version="v1",
            scored_at=datetime.now(timezone.utc),
        )
    assert "churn_probability must be between 0 and 1" in str(exc_info.value)


def test_prediction_probability_above_one():
    with pytest.raises(ValidationError) as exc_info:
        Prediction(
            customer_id="C001",
            dataset_id="ds1",
            scoring_run_id="run1",
            churn_probability=1.01,
            model_version="v1",
            scored_at=datetime.now(timezone.utc),
        )
    assert "churn_probability must be between 0 and 1" in str(exc_info.value)


def test_prediction_missing_field():
    with pytest.raises(ValidationError):
        Prediction(
            customer_id="C001",
            dataset_id="ds1",
            scoring_run_id="run1",
            churn_probability=0.5,
            # missing model_version
            scored_at=datetime.now(timezone.utc),
        )


def test_prediction_batch_valid():
    batch = PredictionBatch(
        predictions=[
            Prediction(
                customer_id="C001",
                dataset_id="ds1",
                scoring_run_id="run1",
                churn_probability=0.3,
                model_version="v1",
                scored_at=datetime.now(timezone.utc),
            ),
            Prediction(
                customer_id="C002",
                dataset_id="ds1",
                scoring_run_id="run1",
                churn_probability=0.7,
                model_version="v1",
                scored_at=datetime.now(timezone.utc),
            ),
        ]
    )
    assert len(batch.predictions) == 2


def test_explanation_valid_with_float_feature():
    e = Explanation(
        customer_id="C001",
        model_version="v1",
        feature_name="monthly_charges",
        feature_value=89.5,
        contribution=0.42,
        base_value=-0.85,
        explained_output="log_odds",
        calibration_applied="original",
    )
    assert e.feature_value == 89.5


def test_explanation_valid_with_string_feature():
    e = Explanation(
        customer_id="C001",
        model_version="v1",
        feature_name="contract_type",
        feature_value="month_to_month",
        contribution=0.38,
        base_value=-0.85,
        explained_output="probability",
        calibration_applied="calibrated",
    )
    assert e.feature_value == "month_to_month"


def test_explanation_missing_field():
    with pytest.raises(ValidationError):
        Explanation(
            customer_id="C001",
            model_version="v1",
            feature_name="tenure",
            feature_value=12,
            contribution=0.2,
            # missing base_value
            explained_output="log_odds",
            calibration_applied="original",
        )


def test_explanation_list_valid():
    el = ExplanationList(
        explanations=[
            Explanation(
                customer_id="C001",
                model_version="v1",
                feature_name="f1",
                feature_value=10,
                contribution=0.1,
                base_value=0.0,
                explained_output="log_odds",
                calibration_applied="original",
            ),
            Explanation(
                customer_id="C001",
                model_version="v1",
                feature_name="f2",
                feature_value="cat",
                contribution=-0.2,
                base_value=0.0,
                explained_output="log_odds",
                calibration_applied="original",
            ),
        ]
    )
    assert len(el.explanations) == 2
