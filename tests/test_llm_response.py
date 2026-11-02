import json
import traceback

import pytest

from churn_retention.adapters.llm.base import LLMResponseValidationError
from churn_retention.modules.scoring.llm_response import (
    MAX_RESPONSE_CHARS,
    parse_churn_response,
)

VALID = {
    "churn_prediction": 1,
    "churn_probability": 0.75,
    "reason": "Khách hàng có hợp đồng ngắn hạn; bằng chứng còn hạn chế.",
}


def test_valid_response_is_preserved_without_recalculation() -> None:
    data = {
        **VALID,
        "churn_probability": 0.13,
        "reason": "  Chưa đủ bằng chứng để khẳng định chắc chắn.  ",
    }
    assert parse_churn_response(json.dumps(data)).model_dump() == data
    # Do not impose an unapproved probability threshold or change the label.
    assert parse_churn_response(json.dumps(data)).churn_prediction == 1


@pytest.mark.parametrize("probability", [0, 1, 0.0, 1.0, 0.125])
def test_probability_accepts_json_numbers_without_type_coercion(
    probability: float,
) -> None:
    parsed = parse_churn_response(json.dumps({**VALID, "churn_probability": probability}))
    assert parsed.churn_probability == probability
    assert type(parsed.churn_probability) is type(probability)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text",
        '```json\n{"churn_prediction":1,"churn_probability":0.75,"reason":"x"}\n```',
        'Result: {"churn_prediction":1,"churn_probability":0.75,"reason":"x"}',
        '{"churn_prediction":1,"churn_probability":0.75,"reason":"x",}',
        json.dumps(VALID) + "\nextra",
        json.dumps(VALID) + json.dumps(VALID),
        '{"churn_prediction":1 /* comment */,"churn_probability":0.75,"reason":"x"}',
    ],
)
def test_invalid_json_is_never_repaired_or_extracted(text: str) -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(text)
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize("value", [[], None, True, False, 1, "JSON as a string"])
def test_only_json_object_root_is_accepted(value: object) -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps(value))
    assert caught.value.code == "invalid_object"


@pytest.mark.parametrize("change", ["missing", "extra", "case"])
def test_exactly_three_keys_are_required(change: str) -> None:
    data = dict(VALID)
    if change == "missing":
        data.pop("reason")
    elif change == "extra":
        data["provider"] = "Gemini"
    else:
        data["Reason"] = data.pop("reason")
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps(data))
    assert caught.value.code == "invalid_keys"


def test_duplicate_keys_are_rejected() -> None:
    text = '{"churn_prediction":0,"churn_prediction":1,"churn_probability":0.75,"reason":"x"}'
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(text)
    assert caught.value.code == "invalid_keys"


@pytest.mark.parametrize("value", [True, False, 1.0, 2, -1, "1", None, [], {}])
def test_prediction_is_strict_integer_binary(value: object) -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps({**VALID, "churn_prediction": value}))
    assert caught.value.code == "invalid_prediction"


@pytest.mark.parametrize("value", [True, False, "0.75", -0.1, 1.1, None, [], {}])
def test_probability_is_strict_finite_number_in_range(value: object) -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps({**VALID, "churn_probability": value}))
    assert caught.value.code == "invalid_probability"


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_nonfinite_json_numbers_are_rejected(number: str) -> None:
    text = '{"churn_prediction":1,"churn_probability":' + number + ',"reason":"x"}'
    with pytest.raises(LLMResponseValidationError):
        parse_churn_response(text)


@pytest.mark.parametrize("value", ["", " \n\t", "x" * 301, None, 10, True, [], {}, "\ud800"])
def test_reason_is_bounded_nonempty_unicode_string(value: object) -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps({**VALID, "reason": value}))
    assert caught.value.code == "invalid_reason"


def test_whitespace_around_json_and_reason_boundary_are_valid() -> None:
    parsed = parse_churn_response(" \n" + json.dumps({**VALID, "reason": "a" * 300}) + "\t")
    assert len(parsed.reason) == 300


def test_response_size_and_malformed_type_are_rejected() -> None:
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(" " * (MAX_RESPONSE_CHARS + 1))
    assert caught.value.code == "response_too_large"
    with pytest.raises(LLMResponseValidationError):
        parse_churn_response(None)  # type: ignore[arg-type]


def test_validation_errors_do_not_expose_response_or_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PRIVATE_CUSTOMER_AND_API_KEY_PAYLOAD"
    with pytest.raises(LLMResponseValidationError) as caught:
        parse_churn_response(json.dumps({**VALID, "churn_probability": secret}))
    assert secret not in str(caught.value)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert secret not in caplog.text
    error = LLMResponseValidationError(secret, code=secret)
    assert error.code == "validator_error"
