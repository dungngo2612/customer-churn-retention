from collections.abc import Sequence

import pandas as pd

from churn_retention.ml.training.bundle import LoadedBundle, load_bundle


class BundlePredictor:
    def __init__(self) -> None:
        self._bundle: LoadedBundle | None = None

    @property
    def model_version(self) -> str:
        if self._bundle is None:
            raise RuntimeError("chưa load bundle")
        return self._bundle.model_version

    def load(self, artifact_path: str) -> None:
        self._bundle = load_bundle(artifact_path)

    def predict_batch(
        self, features: pd.DataFrame, customer_ids: Sequence[str]
    ) -> list[float]:
        """Return probabilities in the supplied feature row/customer ID order."""
        if self._bundle is None:
            raise RuntimeError("chưa load bundle")
        if len(customer_ids) != len(features):
            raise ValueError(
                f"Length mismatch: {len(customer_ids)} customer_ids vs {len(features)} features"
            )
        if len(features) == 0:
            return []
        return self._bundle.predict_proba(features).tolist()
