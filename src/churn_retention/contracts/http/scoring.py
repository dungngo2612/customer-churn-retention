from pydantic import (
    AwareDatetime,
    BaseModel,
    field_validator,
    model_validator,
)


class Prediction(BaseModel):
    customer_id: str
    dataset_id: str
    scoring_run_id: str
    churn_probability: float
    model_version: str
    scored_at: AwareDatetime

    @field_validator("churn_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("churn_probability must be between 0 and 1")
        return v


class PredictionBatch(BaseModel):
    scoring_run_id: str
    model_version: str
    predictions: list[Prediction]

    @model_validator(mode="after")
    def validate_consistency(self):
        """R04: Validate wrapper fields match all items."""
        for prediction in self.predictions:
            if prediction.scoring_run_id != self.scoring_run_id:
                raise ValueError(
                    f"Prediction has scoring_run_id '{prediction.scoring_run_id}' "
                    f"but wrapper has '{self.scoring_run_id}'"
                )
            if prediction.model_version != self.model_version:
                raise ValueError(
                    f"Prediction has model_version '{prediction.model_version}' "
                    f"but wrapper has '{self.model_version}'"
                )
        return self
