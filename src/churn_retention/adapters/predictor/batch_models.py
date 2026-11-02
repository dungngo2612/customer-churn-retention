"""Resolve exact bundle versions and bridge optional, bounded M2.3 explanations."""

from pathlib import Path

import pandas as pd

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.contracts.http.explanations import ModelExplanation


class BundleScoringModels:
    def __init__(self, bundle_root: Path, background_train: Path | None = None) -> None:
        self.bundle_root = bundle_root.resolve()
        self.background_train = background_train
        self._predictors: dict[str, BundlePredictor] = {}
        self._background: pd.DataFrame | None = None

    def predictor_for(self, model_version: str) -> BundlePredictor:
        if model_version not in self._predictors:
            path = (self.bundle_root / model_version).resolve()
            if path.parent != self.bundle_root:
                raise ValueError("model_version must identify a direct child of bundle_root")
            predictor = BundlePredictor()
            predictor.load(str(path))
            if predictor.model_version != model_version:
                raise ValueError("Bundle metadata does not match requested model_version")
            self._predictors[model_version] = predictor
        return self._predictors[model_version]

    def explain_one(self, model_version: str, row: pd.DataFrame) -> ModelExplanation:
        # No SHAP import or background file read when explain_limit == 0.
        from churn_retention.ml.explainability import explain_one, sample_background

        if self._background is None:
            if self.background_train is None or self.background_train.resolve().name != "train.csv":
                raise ValueError("SHAP requires --background-train pointing to train.csv")
            frame = pd.read_csv(self.background_train.resolve())
            frame.attrs["source_split"] = "train"
            self._background = sample_background(frame)
        predictor = self.predictor_for(model_version)
        return ModelExplanation.model_validate(
            explain_one(predictor.loaded_bundle, row, self._background)
        )
