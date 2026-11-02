"""Aggregate logistic SHAP output, reusing the existing per-feature contract."""

import math
from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, Field, FiniteFloat, model_validator

from churn_retention.contracts.http.scoring import Explanation, ExplanationList


class FeatureColumn(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    dtype: str


class BackgroundSample(BaseModel):
    seed: Literal[42]
    rows: int = Field(ge=1, le=200)
    customer_ids: list[str]


class DetailedContribution(BaseModel):
    feature_name: str
    source_feature: str
    transformed_value: FiniteFloat
    contribution: FiniteFloat


class ModelExplanation(ExplanationList):
    """The old feature records are exposed as `contributions` in the new output.

    Use model_dump(by_alias=True) for the wire representation. Existing callers
    of Explanation and ExplanationList keep their original field names.
    Calibrated probability explanations need a different future contract.
    """

    explanations: list[Explanation] = Field(
        validation_alias=AliasChoices("contributions", "explanations"),
        serialization_alias="contributions",
        min_length=1,
    )
    output_type: Literal["log_odds"]
    calibration_applied: Literal["original"]
    base_value: FiniteFloat
    log_odds: FiniteFloat
    probability: FiniteFloat = Field(ge=0, le=1)
    feature_schema: list[FeatureColumn] = Field(min_length=1)
    detailed_contributions: list[DetailedContribution] = Field(min_length=1)
    background: BackgroundSample

    @property
    def contributions(self) -> list[Explanation]:
        return self.explanations

    @model_validator(mode="after")
    def validate_shap_units(self) -> Self:
        names = [column.name for column in self.feature_schema]
        actual = [item.feature_name for item in self.contributions]
        if len(set(names)) != len(names) or sorted(actual) != sorted(names):
            raise ValueError("Contributions must match the unique original feature_schema")
        for item in self.contributions:
            if (
                item.explained_output != self.output_type
                or item.calibration_applied != self.calibration_applied
                or item.base_value != self.base_value
                or not math.isfinite(item.contribution)
            ):
                raise ValueError("Contribution units, calibration and base must match wrapper")
            details = [
                d for d in self.detailed_contributions if d.source_feature == item.feature_name
            ]
            if not math.isclose(
                sum(d.contribution for d in details), item.contribution, abs_tol=1e-6
            ):
                raise ValueError("Detailed contributions must sum to each original feature")
        if any(item.source_feature not in names for item in self.detailed_contributions):
            raise ValueError("Detailed contributions must map to feature_schema")
        total = self.base_value + sum(item.contribution for item in self.contributions)
        if not math.isclose(total, self.log_odds, abs_tol=1e-6):
            raise ValueError("Base plus contributions must equal predicted log_odds")
        # Stable sigmoid, including logits with magnitude above the exp limit.
        exp = math.exp(-abs(self.log_odds))
        probability = 1 / (1 + exp) if self.log_odds >= 0 else exp / (1 + exp)
        if not math.isclose(probability, self.probability, abs_tol=1e-6):
            raise ValueError("Probability must equal sigmoid(log_odds)")
        return self
