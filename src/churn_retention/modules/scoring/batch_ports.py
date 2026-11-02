"""Replacement boundaries for a dataset service, durable database and queue."""

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Protocol

import pandas as pd

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.contracts.http.scoring import Prediction, PredictionBatch
from churn_retention.modules.scoring.jobs import ScoringJob


class JobRepository(Protocol):
    def get(self, job_id: str) -> ScoringJob: ...
    def find_active(self, key: str) -> ScoringJob | None: ...
    def find_or_recover(self, key: str, now: datetime) -> ScoringJob | None:
        """Atomically fail expired RUNNING jobs and remove their staging."""
        ...

    def create_or_get(self, job: ScoringJob, now: datetime | None = None) -> ScoringJob: ...
    def claim(self, job_id: str, now: datetime) -> ScoringJob | None: ...
    def update_progress(self, job_id: str, processed_rows: int) -> ScoringJob: ...
    def renew_lease(self, job_id: str, now: datetime) -> ScoringJob: ...
    def fail_if_running(self, job_id: str, now: datetime, error: str) -> ScoringJob:
        """Fail and discard staging atomically; preserve an already terminal job."""
        ...

    def finish(self, job_id: str, now: datetime, error: str | None = None) -> ScoringJob:
        """Publish atomically on success; only after all staged predictions exist."""
        ...


class PredictionStore(Protocol):
    def stage(
        self,
        batch: PredictionBatch,
        explanations: Sequence[ModelExplanation],
        now: datetime,
    ) -> None: ...
    def discard(self, job_id: str) -> None: ...
    def predictions(self, job_id: str) -> list[Prediction]:
        """Raise ResultsNotReady until the job is SUCCEEDED."""
        ...

    def explanations(self, job_id: str) -> list[ModelExplanation]: ...


class JobQueue(Protocol):
    def enqueue(self, message: Mapping[str, object]) -> None: ...
    def dequeue(self) -> dict[str, str] | None: ...


class DatasetReader(Protocol):
    def total_rows(self, dataset_id: str) -> int: ...
    def chunks(self, dataset_id: str, chunk_size: int) -> Iterator[pd.DataFrame]: ...


def message_job_id(message: Mapping[str, object]) -> str:
    if set(message) != {"job_id"}:
        raise ValueError("Scoring messages must contain only job_id")
    job_id = message["job_id"]
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("Scoring message job_id must be a non-empty string")
    return job_id
