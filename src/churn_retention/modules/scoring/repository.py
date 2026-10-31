from typing import Protocol

from churn_retention.contracts.http.scoring import PredictionBatch


class ScoringRunRepository(Protocol):
    def save(self, batch: PredictionBatch) -> None: ...

    def get(self, run_id: str) -> PredictionBatch | None: ...


class InMemoryScoringRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, PredictionBatch] = {}

    def save(self, batch: PredictionBatch) -> None:
        self._runs[batch.scoring_run_id] = batch

    def get(self, run_id: str) -> PredictionBatch | None:
        return self._runs.get(run_id)
