from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Prediction(BaseModel):
    customer_id: str
    dataset_id: str
    scoring_run_id: str
    churn_probability: float
    model_version: str
    scored_at: datetime

    @field_validator("churn_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("churn_probability must be between 0 and 1")
        return v


class PredictionBatch(BaseModel):
    predictions: list[Prediction]


class Explanation(BaseModel):
    customer_id: str
    model_version: str
    feature_name: str
    feature_value: str | float
    contribution: float
    base_value: float
    explained_output: Literal["log_odds", "probability"]
    calibration_applied: Literal["original", "calibrated"]


class ExplanationList(BaseModel):
    explanations: list[Explanation]
