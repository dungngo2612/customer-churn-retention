"""Synthetic boundary and economic regression tests; no dataset, artifacts or API."""

from uuid import UUID

import pandas as pd
import pytest
from pydantic import ValidationError

from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_prediction_record import (
    FEATURE_NAMES,
    PredictionRecord,
)
from churn_retention.entrypoints.streamlit.llm_prediction_record import (
    RECORD_KEY,
    capture_submission,
    clear_record,
)
from churn_retention.modules.retention.demo import demo_inputs
from churn_retention.modules.retention.llm_adapter import (
    CustomerContext,
    RetentionScenario,
    project_inputs,
    simulate,
)


def accepted() -> PredictionRecord:
    sample = demo_inputs()[0][0]
    prediction = LLMChurnPrediction.model_validate(
        {
            **sample.prediction.model_dump(),
            "provider": "Groq",
            "reference_count": 5,
            "reference_version": "rag-v2-synthetic-test",
        }
    )
    return PredictionRecord.capture(sample.features(), prediction, reference_sha256="a" * 64)


def test_record_is_deeply_immutable_and_round_trips() -> None:
    record = accepted()
    assert isinstance(record.record_id, UUID) and record.created_at.utcoffset() is not None
    with pytest.raises(ValidationError):
        record.origin = "synthetic"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        record.prediction.churn_probability = 0.1  # type: ignore[misc]
    features = record.features()
    features["MonthlyCharges"] = 999.0
    assert record.features()["MonthlyCharges"] == 70.0
    assert isinstance(record.input_snapshot, tuple)
    assert all(isinstance(item, tuple) for item in record.input_snapshot)
    assert PredictionRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("extra", [{"customerID": "fake"}, {"Churn": "Yes"}, {"api_key": "fake"}])
def test_snapshot_does_not_accept_identity_labels_or_credentials(extra) -> None:
    record = accepted()
    with pytest.raises(ValueError):
        PredictionRecord.capture(
            {**record.features(), **extra}, record.prediction, reference_sha256="a" * 64
        )


def test_capture_freezes_source_and_does_not_deduplicate_distinct_submissions() -> None:
    record = accepted()
    values = record.features()
    state: dict[str, object] = {}
    capture_submission(state, values, record.prediction, "a" * 64, "context")
    first = state[RECORD_KEY]
    assert isinstance(first, PredictionRecord)
    values["MonthlyCharges"] = 1.0
    assert first.features()["MonthlyCharges"] == 70.0
    capture_submission(state, record.features(), record.prediction, "a" * 64, "context")
    second = state[RECORD_KEY]
    assert isinstance(second, PredictionRecord) and second.record_id != first.record_id
    clear_record(state)
    assert state == {}


def test_record_requires_v2_provenance_and_synthetic_records_cannot_claim_it() -> None:
    record = accepted()
    with pytest.raises(ValueError):
        PredictionRecord.capture(record.features(), record.prediction)
    with pytest.raises(ValueError):
        PredictionRecord.capture(record.features(), record.prediction, origin="synthetic")
    assert all(r.origin == "synthetic" and r.prediction.provider is None for r, _ in demo_inputs())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "70"])
def test_numeric_snapshot_validation(value) -> None:
    record = accepted()
    with pytest.raises(ValueError):
        PredictionRecord.capture(
            {**record.features(), "MonthlyCharges": value},
            record.prediction,
            reference_sha256="a" * 64,
        )


def test_status_is_observed_metadata_not_predicted_label() -> None:
    record = accepted()
    assert record.prediction.churn_prediction == 1
    active = CustomerContext(customer_ref="fictional-active", observed_active=True)
    customers, predictions = project_inputs([(record, active)])
    assert customers.iloc[0]["Churn"] == "No"
    assert predictions.iloc[0]["churn_prob"] == record.prediction.churn_probability
    result = simulate([(record, active)], RetentionScenario(budget=100.0))
    assert len(result.allocations) == 1
    assert result.allocations.iloc[0]["action_code"] != "A0"
    inactive = CustomerContext(customer_ref="fictional-inactive", observed_active=False)
    assert simulate([(record, inactive)], RetentionScenario()).allocations.empty
    with pytest.raises(ValueError, match="trạng thái"):
        project_inputs([(record, CustomerContext(customer_ref="unknown"))])


def test_duplicate_customer_or_run_and_mixed_demo_are_rejected() -> None:
    inputs = demo_inputs()
    with pytest.raises(ValueError, match="trùng ID"):
        project_inputs([inputs[0], inputs[0]])
    with pytest.raises(ValueError, match="trùng ID"):
        project_inputs([inputs[0], (inputs[1][0], inputs[0][1])])
    with pytest.raises(ValueError, match="trộn"):
        simulate(
            [
                inputs[0],
                (
                    accepted(),
                    CustomerContext(customer_ref="live", observed_active=True),
                ),
            ],
            RetentionScenario(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget", -1.0),
        ("budget", float("nan")),
        ("budget", float("inf")),
        ("budget", True),
        ("margin", 0.0),
        ("margin", 1.1),
        ("horizon_months", 1.5),
        ("horizon_months", True),
        ("horizon_months", 0),
        ("contact_cost_multiplier", float("inf")),
        ("contact_cost_multiplier", -1.0),
    ],
)
def test_invalid_economic_config_rejected(field, value) -> None:
    with pytest.raises(ValueError):
        RetentionScenario.model_validate({field: value})


@pytest.mark.parametrize("budget", [0.0, 8.0, 20.0, 100.0])
def test_budget_and_ordering_are_deterministic_and_records_unchanged(budget) -> None:
    inputs = demo_inputs()
    before = [r.model_dump_json() for r, _ in inputs]
    config = RetentionScenario(budget=budget)
    result = simulate(inputs, config)
    assert len(result.allocations) == 2  # Inactive and zero-tenure cases are excluded.
    assert result.allocations["cost"].sum() <= budget + 1e-9
    assert result.allocations["customerID"].is_unique
    pd.testing.assert_frame_equal(result.allocations, simulate(inputs, config).allocations)
    assert before == [r.model_dump_json() for r, _ in inputs]
    assert result.synthetic and all(r.prediction.probability_calibrated is False for r, _ in inputs)


def test_all_a0_and_empty_inputs_are_valid_scenarios() -> None:
    result = simulate(demo_inputs(), RetentionScenario(contact_cost_multiplier=1000.0))
    assert set(result.allocations["action_code"]) == {"A0"}
    assert result.allocations["cost"].sum() == result.allocations["net_value"].sum() == 0
    assert simulate([], RetentionScenario()).allocations.empty


def test_feature_interface_stays_the_same_as_the_existing_form() -> None:
    from churn_retention.entrypoints.streamlit.llm_input import FEATURE_GROUPS

    assert FEATURE_NAMES == tuple(name for _, group in FEATURE_GROUPS for name in group)


def test_finite_inputs_cannot_overflow_economic_outputs() -> None:
    record = accepted()
    huge = PredictionRecord.capture(
        {**record.features(), "MonthlyCharges": 1e308},
        record.prediction,
        reference_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="hữu hạn"):
        simulate(
            [
                (
                    huge,
                    CustomerContext(customer_ref="synthetic-overflow", observed_active=True),
                )
            ],
            RetentionScenario(),
        )
    with pytest.raises(ValueError, match="hữu hạn"):
        simulate(demo_inputs(), RetentionScenario(contact_cost_multiplier=1e308))
