"""Immutable accepted predictions; business identity never enters the LLM DTO."""

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Self, TypeAlias
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from churn_retention.contracts.llm import LLMChurnPrediction

FEATURE_NAMES = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)
FeatureValue: TypeAlias = str | StrictInt | StrictFloat | None  # noqa: UP040 -- Keep the runtime Pydantic union unchanged.


class PredictionRecord(BaseModel):
    """Deeply immutable scalar snapshot, with one UUID per accepted submission.

    config_id identifies the application contract, NOT an actual provider model.
    It deliberately contains no credentials, customer ID or observed churn label.
    Synthetic records are explicitly distinguished from LLM service outputs.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    record_id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    origin: Literal["llm", "synthetic"]
    input_snapshot: tuple[tuple[str, FeatureValue], ...]
    prediction: LLMChurnPrediction
    reference_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    config_id: Literal["llm-rag-v2-defaults@7341854", "synthetic-demo-v1"]

    @field_validator("input_snapshot")
    @classmethod
    def validate_snapshot(
        cls, value: tuple[tuple[str, FeatureValue], ...]
    ) -> tuple[tuple[str, FeatureValue], ...]:
        if tuple(name for name, _ in value) != FEATURE_NAMES:
            raise ValueError("PredictionRecord requires the canonical 19 features only.")
        numeric = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
        for name, item in value:
            if name in numeric:
                if item is None and name == "TotalCharges":
                    continue
                if not isinstance(item, (int, float)) or isinstance(item, bool):
                    raise ValueError("Invalid numeric snapshot feature.")
                try:
                    valid = math.isfinite(item) and item >= 0
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError("Invalid numeric snapshot feature.")
                if name in {"tenure", "SeniorCitizen"} and int(item) != item:
                    raise ValueError("Snapshot integer feature required.")
                if name == "SeniorCitizen" and item not in (0, 1):
                    raise ValueError("SeniorCitizen must be 0 or 1.")
            elif not isinstance(item, str) or not item.strip():
                raise ValueError("Snapshot categorical feature required.")
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.origin == "llm":
            if (
                self.reference_sha256 is None
                or not self.prediction.reference_version.startswith("rag-v2-")
                or self.config_id != "llm-rag-v2-defaults@7341854"
            ):
                raise ValueError("Accepted LLM record requires verified v2 provenance.")
        elif (
            self.reference_sha256 is not None
            or self.prediction.reference_version != "synthetic-demo-v1"
            or self.config_id != "synthetic-demo-v1"
            or self.prediction.provider is not None
            or self.prediction.reference_count != 0
        ):
            raise ValueError("Synthetic records must not claim real RAG/provider provenance.")
        return self

    @classmethod
    def capture(
        cls,
        snapshot: Mapping[str, object],
        prediction: LLMChurnPrediction,
        *,
        reference_sha256: str | None = None,
        origin: Literal["llm", "synthetic"] = "llm",
    ) -> Self:
        if set(snapshot) != set(FEATURE_NAMES):
            raise ValueError("Only 19 prediction features may be captured.")
        return cls.model_validate(
            {
                "origin": origin,
                "input_snapshot": tuple((name, snapshot[name]) for name in FEATURE_NAMES),
                "prediction": prediction,
                "reference_sha256": reference_sha256,
                "config_id": "synthetic-demo-v1"
                if origin == "synthetic"
                else "llm-rag-v2-defaults@7341854",
            }
        )

    def features(self) -> dict[str, FeatureValue]:
        """Return a disposable projection; callers cannot mutate the record."""
        return dict(self.input_snapshot)
