import hashlib
from collections.abc import Sequence

import pandas as pd


class FakePredictor:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._model_version = "fake-v1"
        self._artifact_path: str | None = None

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self, artifact_path: str) -> None:
        self._artifact_path = artifact_path

    def predict_batch(
        self, features: pd.DataFrame, customer_ids: Sequence[str]
    ) -> list[float]:
        if len(customer_ids) != len(features):
            raise ValueError(
                f"Length mismatch: {len(customer_ids)} customer_ids vs {len(features)} features"
            )

        return [self._generate_probability(customer_id) for customer_id in customer_ids]

    def _generate_probability(self, customer_id: str) -> float:
        combined = f"{self._seed}:{customer_id}"
        hash_digest = hashlib.sha256(combined.encode()).digest()
        hash_int = int.from_bytes(hash_digest[:8], byteorder="big")
        return (hash_int % 100000) / 100000.0
