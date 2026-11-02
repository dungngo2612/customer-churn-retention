"""Strict JSON parsing; no repair, Markdown extraction, or inferred probabilities."""

import json

from pydantic import ValidationError

from churn_retention.adapters.llm.base import LLMResponseValidationError
from churn_retention.contracts.llm import LLMChurnResponse

MAX_RESPONSE_CHARS = 16_000


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LLMResponseValidationError(
                "Duplicate JSON keys are forbidden.", code="invalid_keys"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LLMResponseValidationError(
        "Non-standard JSON numeric constants are forbidden.", code="invalid_json"
    )


def parse_churn_response(text: str) -> LLMChurnResponse:
    if not isinstance(text, str):
        raise LLMResponseValidationError(
            "Response must be JSON text.", code="invalid_json"
        )
    if len(text) > MAX_RESPONSE_CHARS:
        raise LLMResponseValidationError(
            "Response exceeds the size limit.", code="response_too_large"
        )
    try:
        data = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except LLMResponseValidationError:
        raise
    except (ValueError, RecursionError):
        raise LLMResponseValidationError(
            "Response must be a single valid JSON object without Markdown.",
            code="invalid_json",
        ) from None
    if not isinstance(data, dict):
        raise LLMResponseValidationError(
            "Response root must be an object.", code="invalid_object"
        )
    if set(data) != {"churn_prediction", "churn_probability", "reason"}:
        raise LLMResponseValidationError(
            "Response must contain exactly the three required keys.",
            code="invalid_keys",
        )
    try:
        return LLMChurnResponse.model_validate(data)
    except ValidationError as error:
        field = error.errors(include_input=False, include_context=False)[0]["loc"][0]
        codes = {
            "churn_prediction": "invalid_prediction",
            "churn_probability": "invalid_probability",
            "reason": "invalid_reason",
        }
        raise LLMResponseValidationError(
            "Response fields do not satisfy the churn contract.",
            code=codes.get(str(field), "validator_error"),
        ) from None
