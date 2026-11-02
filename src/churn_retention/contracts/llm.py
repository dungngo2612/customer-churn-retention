"""Separate, uncalibrated LLM churn outputs; existing ML Prediction is unchanged."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
)

ProviderName = Literal["Groq", "Gemini"]
MAX_REASON_CHARS = 300


class LLMChurnResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)

    churn_prediction: StrictInt = Field(ge=0, le=1)
    churn_probability: StrictInt | StrictFloat = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=MAX_REASON_CHARS)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must contain non-whitespace text.")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("reason must contain valid Unicode text.") from None
        return value  # Preserve the supplied explanation, including whitespace.


class LLMChurnPrediction(LLMChurnResponse):
    provider: ProviderName | None
    reference_count: StrictInt = Field(ge=0)
    reference_version: str = Field(min_length=1)
    probability_calibrated: Literal[False] = False
