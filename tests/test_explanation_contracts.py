import copy
import math

import pytest
from pydantic import ValidationError

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.contracts.http.scoring import Explanation, ExplanationList


@pytest.fixture
def payload():
    return {
        "customer_id": "val-1",
        "model_version": "logistic-v1",
        "output_type": "log_odds",
        "calibration_applied": "original",
        "base_value": -1.0,
        "log_odds": -0.5,
        "probability": 1 / (1 + math.exp(0.5)),
        "feature_schema": [{"name": "tenure", "kind": "numeric", "dtype": "float64"}],
        "background": {"seed": 42, "rows": 200, "customer_ids": []},
        "contributions": [
            {
                "customer_id": "val-1",
                "model_version": "logistic-v1",
                "feature_name": "tenure",
                "feature_value": 12.0,
                "contribution": 0.5,
                "base_value": -1.0,
                "explained_output": "log_odds",
                "calibration_applied": "original",
            }
        ],
        "detailed_contributions": [
            {
                "feature_name": "num__tenure",
                "source_feature": "tenure",
                "transformed_value": 1.0,
                "contribution": 0.5,
            }
        ],
    }


def test_reuses_feature_contract_and_serializes_contributions(payload):
    result = ModelExplanation.model_validate(payload)
    assert isinstance(result, ExplanationList)
    assert isinstance(result.contributions[0], Explanation)
    assert result.model_dump(by_alias=True) == payload


@pytest.mark.parametrize(
    "value", ["probability", "percentage_points", "calibrated_log_odds", "unknown"]
)
def test_output_type_only_accepts_uncalibrated_log_odds(payload, value):
    payload["output_type"] = value
    with pytest.raises(ValidationError, match="output_type"):
        ModelExplanation.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "customer_id",
        "model_version",
        "output_type",
        "base_value",
        "contributions",
        "feature_schema",
        "log_odds",
        "probability",
        "detailed_contributions",
        "background",
        "calibration_applied",
    ],
)
def test_required_fields(payload, field):
    del payload[field]
    with pytest.raises(ValidationError):
        ModelExplanation.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"calibration_applied": "calibrated"},
        {"probability": 0.9},
        {"base_value": float("nan")},
        {"log_odds": 2.0},
    ],
)
def test_rejects_invalid_units_or_additivity(payload, change):
    payload.update(change)
    with pytest.raises(ValidationError):
        ModelExplanation.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("explained_output", "probability"),
        ("calibration_applied", "calibrated"),
        ("customer_id", "another-customer"),
        ("model_version", "another-model"),
        ("contribution", 0.6),
        ("base_value", 0.0),
    ],
)
def test_rejects_inconsistent_feature_records(payload, field, value):
    changed = copy.deepcopy(payload)
    changed["contributions"][0][field] = value
    with pytest.raises(ValidationError):
        ModelExplanation.model_validate(changed)
