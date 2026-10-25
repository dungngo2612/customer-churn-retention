from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    def load(self, artifact_path: str) -> None: ...

    def predict_batch(
        self, features: pd.DataFrame, customer_ids: Sequence[str]
    ) -> list[float]: ...
