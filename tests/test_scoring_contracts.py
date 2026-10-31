import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

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
        scored_at=datetime.now(UTC),
    )
    assert p.churn_probability == 0.5


def test_prediction_probability_at_bounds():
    p0 = Prediction(
        customer_id="C001",
        dataset_id="ds1",
        scoring_run_id="run1",
        churn_probability=0.0,
        model_version="v1",
        scored_at=datetime.now(UTC),
    )
    assert p0.churn_probability == 0.0

    p1 = Prediction(
        customer_id="C001",
        dataset_id="ds1",
        scoring_run_id="run1",
        churn_probability=1.0,
        model_version="v1",
        scored_at=datetime.now(UTC),
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
            scored_at=datetime.now(UTC),
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
            scored_at=datetime.now(UTC),
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
            scored_at=datetime.now(UTC),
        )


def test_prediction_batch_valid():
    batch = PredictionBatch(
        scoring_run_id="run1",
        model_version="v1",
        predictions=[
            Prediction(
                customer_id="C001",
                dataset_id="ds1",
                scoring_run_id="run1",
                churn_probability=0.3,
                model_version="v1",
                scored_at=datetime.now(UTC),
            ),
            Prediction(
                customer_id="C002",
                dataset_id="ds1",
                scoring_run_id="run1",
                churn_probability=0.7,
                model_version="v1",
                scored_at=datetime.now(UTC),
            ),
        ],
    )
    assert len(batch.predictions) == 2
    assert batch.scoring_run_id == "run1"
    assert batch.model_version == "v1"


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
        customer_id="C001",
        model_version="v1",
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
        ],
    )
    assert len(el.explanations) == 2
    assert el.customer_id == "C001"
    assert el.model_version == "v1"


@pytest.fixture
def demo_prediction_payload() -> dict[str, Any]:
    return {
        "customer_id": "demo-customer-001",
        "dataset_id": "demo-dataset",
        "scoring_run_id": "demo-run",
        "churn_probability": 0.5,
        "model_version": "demo-model",
        "scored_at": "2026-10-17T00:00:00+00:00",
    }


@pytest.fixture
def demo_explanation_payload() -> dict[str, Any]:
    return {
        "customer_id": "demo-customer-001",
        "model_version": "demo-model",
        "feature_name": "demo-usage",
        "feature_value": 10.0,
        "contribution": 0.2,
        "base_value": -0.5,
        "explained_output": "log_odds",
        "calibration_applied": "original",
    }


@pytest.mark.parametrize("scored_at", [datetime(2026, 10, 17), "2026-10-17T00:00:00"])
def test_prediction_rejects_missing_timezone(
    scored_at: datetime | str, demo_prediction_payload: dict[str, Any]
) -> None:
    demo_prediction_payload["scored_at"] = scored_at

    with pytest.raises(ValidationError) as exc_info:
        Prediction.model_validate(demo_prediction_payload)

    assert exc_info.value.errors()[0]["loc"] == ("scored_at",)
    assert exc_info.value.errors()[0]["type"] == "timezone_aware"


@pytest.mark.parametrize(
    "missing_fields",
    [
        ("explained_output",),
        ("calibration_applied",),
        ("explained_output", "calibration_applied"),
    ],
)
def test_explanation_requires_explicit_literals(
    missing_fields: tuple[str, ...], demo_explanation_payload: dict[str, Any]
) -> None:
    for field in missing_fields:
        del demo_explanation_payload[field]

    with pytest.raises(ValidationError) as exc_info:
        Explanation.model_validate(demo_explanation_payload)

    assert {error["loc"] for error in exc_info.value.errors()} == {
        (field,) for field in missing_fields
    }
    assert all(error["type"] == "missing" for error in exc_info.value.errors())


@pytest.mark.parametrize(
    "field, value", [("explained_output", "raw"), ("calibration_applied", "unknown")]
)
def test_explanation_rejects_invalid_literals(
    field: str, value: str, demo_explanation_payload: dict[str, Any]
) -> None:
    demo_explanation_payload[field] = value

    with pytest.raises(ValidationError) as exc_info:
        Explanation.model_validate(demo_explanation_payload)

    assert exc_info.value.errors()[0]["loc"] == (field,)
    assert exc_info.value.errors()[0]["type"] == "literal_error"


@pytest.mark.parametrize("field", ["scoring_run_id", "model_version", "predictions"])
def test_prediction_batch_requires_fields(
    field: str, demo_prediction_payload: dict[str, Any]
) -> None:
    payload = {
        "scoring_run_id": "demo-run",
        "model_version": "demo-model",
        "predictions": [demo_prediction_payload],
    }
    del payload[field]

    with pytest.raises(ValidationError) as exc_info:
        PredictionBatch.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == (field,)
    assert exc_info.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("field", ["customer_id", "model_version", "explanations"])
def test_explanation_list_requires_fields(
    field: str, demo_explanation_payload: dict[str, Any]
) -> None:
    payload = {
        "customer_id": "demo-customer-001",
        "model_version": "demo-model",
        "explanations": [demo_explanation_payload],
    }
    del payload[field]

    with pytest.raises(ValidationError) as exc_info:
        ExplanationList.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == (field,)
    assert exc_info.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    "filename, model",
    [
        ("prediction_example.json", PredictionBatch),
        ("explanation_example.json", ExplanationList),
    ],
)
def test_demo_json_examples_parse_with_contracts(filename: str, model: type[BaseModel]) -> None:
    example_path = Path(__file__).resolve().parents[1] / "docs" / "contracts" / filename
    text = example_path.read_text(encoding="utf-8")
    comment = json.loads(text)["_comment"].lower()

    assert "demo" in comment
    assert "fabricated" in comment
    assert isinstance(model.model_validate_json(text), model)


def test_prediction_batch_validates_scoring_run_id_consistency():
    """R04: PredictionBatch must validate that wrapper scoring_run_id matches all items."""
    with pytest.raises(ValidationError) as exc_info:
        PredictionBatch(
            scoring_run_id="run1",
            model_version="v1",
            predictions=[
                Prediction(
                    customer_id="C001",
                    dataset_id="ds1",
                    scoring_run_id="run1",  # matches wrapper
                    churn_probability=0.3,
                    model_version="v1",
                    scored_at=datetime.now(UTC),
                ),
                Prediction(
                    customer_id="C002",
                    dataset_id="ds1",
                    scoring_run_id="run2",  # MISMATCH
                    churn_probability=0.7,
                    model_version="v1",
                    scored_at=datetime.now(UTC),
                ),
            ],
        )

    assert "scoring_run_id" in str(exc_info.value)
    assert "run2" in str(exc_info.value)


def test_prediction_batch_validates_model_version_consistency():
    """R04: PredictionBatch must validate that wrapper model_version matches all items."""
    with pytest.raises(ValidationError) as exc_info:
        PredictionBatch(
            scoring_run_id="run1",
            model_version="v1",
            predictions=[
                Prediction(
                    customer_id="C001",
                    dataset_id="ds1",
                    scoring_run_id="run1",
                    churn_probability=0.3,
                    model_version="v1",  # matches wrapper
                    scored_at=datetime.now(UTC),
                ),
                Prediction(
                    customer_id="C002",
                    dataset_id="ds1",
                    scoring_run_id="run1",
                    churn_probability=0.7,
                    model_version="v2",  # MISMATCH
                    scored_at=datetime.now(UTC),
                ),
            ],
        )

    assert "model_version" in str(exc_info.value)
    assert "v2" in str(exc_info.value)


def test_explanation_list_validates_customer_id_consistency():
    """R04: ExplanationList must validate that wrapper customer_id matches all items."""
    with pytest.raises(ValidationError) as exc_info:
        ExplanationList(
            customer_id="C001",
            model_version="v1",
            explanations=[
                Explanation(
                    customer_id="C001",  # matches wrapper
                    model_version="v1",
                    feature_name="f1",
                    feature_value=10,
                    contribution=0.1,
                    base_value=0.0,
                    explained_output="log_odds",
                    calibration_applied="original",
                ),
                Explanation(
                    customer_id="C002",  # MISMATCH
                    model_version="v1",
                    feature_name="f2",
                    feature_value="cat",
                    contribution=-0.2,
                    base_value=0.0,
                    explained_output="log_odds",
                    calibration_applied="original",
                ),
            ],
        )

    assert "customer_id" in str(exc_info.value)
    assert "C002" in str(exc_info.value)


def test_explanation_list_validates_model_version_consistency():
    """R04: ExplanationList must validate that wrapper model_version matches all items."""
    with pytest.raises(ValidationError) as exc_info:
        ExplanationList(
            customer_id="C001",
            model_version="v1",
            explanations=[
                Explanation(
                    customer_id="C001",
                    model_version="v1",  # matches wrapper
                    feature_name="f1",
                    feature_value=10,
                    contribution=0.1,
                    base_value=0.0,
                    explained_output="log_odds",
                    calibration_applied="original",
                ),
                Explanation(
                    customer_id="C001",
                    model_version="v2",  # MISMATCH
                    feature_name="f2",
                    feature_value="cat",
                    contribution=-0.2,
                    base_value=0.0,
                    explained_output="log_odds",
                    calibration_applied="original",
                ),
            ],
        )

    assert "model_version" in str(exc_info.value)
    assert "v2" in str(exc_info.value)
