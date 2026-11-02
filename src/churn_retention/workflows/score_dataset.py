"""Idempotent orchestration around the existing synchronous scoring use case."""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.modules.scoring.batch_ports import (
    DatasetReader,
    JobQueue,
    JobRepository,
    PredictionStore,
)
from churn_retention.modules.scoring.jobs import (
    DEFAULT_LEASE_SECONDS,
    JobStatus,
    ScoringJob,
    idempotency_key,
    public_scoring_error,
)
from churn_retention.modules.scoring.ports import Predictor
from churn_retention.modules.scoring.use_cases import score_dataset


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    return str(uuid4())


def get_job_status(job_id: str, jobs: JobRepository) -> ScoringJob:
    """Read persisted state without loading inputs, enqueueing or recovering jobs."""
    return jobs.get(job_id)


class BatchScoringService:
    def __init__(
        self,
        jobs: JobRepository,
        predictions: PredictionStore,
        queue: JobQueue,
        reader: DatasetReader,
        predictor_for: Callable[[str], Predictor],
        *,
        explain: Callable[[str, pd.DataFrame], ModelExplanation] | None = None,
        clock: Callable[[], datetime] = utc_now,
        job_id_factory: Callable[[], str] = new_job_id,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self.jobs, self.predictions, self.queue, self.reader = (
            jobs,
            predictions,
            queue,
            reader,
        )
        self.predictor_for, self.explain = predictor_for, explain
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.clock, self.job_id_factory = clock, job_id_factory
        self.lease_seconds = lease_seconds

    def create_job(
        self,
        dataset_id: str,
        model_version: str,
        chunk_size: int = 200,
        *,
        explain_limit: int = 0,
        explain_customer_ids: Sequence[str] = (),
    ) -> ScoringJob:
        existing = self.jobs.find_or_recover(
            idempotency_key(dataset_id, model_version), self.clock()
        )
        if existing is not None:
            job = existing
        else:
            candidate = ScoringJob(
                job_id=self.job_id_factory(),
                dataset_id=dataset_id,
                model_version=model_version,
                chunk_size=chunk_size,
                total_rows=self.reader.total_rows(dataset_id),
                created_at=self.clock(),
                lease_seconds=self.lease_seconds,
                explain_limit=explain_limit,
                explain_customer_ids=tuple(explain_customer_ids),
            )
            job = self.jobs.create_or_get(
                candidate, self.clock()
            )  # Atomic duplicate check in storage.
        if job.status == JobStatus.PENDING:
            self.queue.enqueue({"job_id": job.job_id})
        return job

    def run_scoring_job(self, job_id: str) -> ScoringJob:
        job = self.jobs.claim(job_id, self.clock())
        if job is None:
            return self.jobs.get(job_id)
        try:
            predictor = self.predictor_for(job.model_version)
            if predictor.model_version != job.model_version:
                raise ValueError("Loaded model_version does not match pinned job model")
            explained: set[str] = set()
            requested = set(job.explain_customer_ids)
            for chunk in self.reader.chunks(job.dataset_id, job.chunk_size):
                if chunk.empty:
                    continue
                if len(chunk) > job.chunk_size:
                    raise ValueError("DatasetReader exceeded chunk_size")
                if predictor.model_version != job.model_version:
                    raise ValueError("Predictor model_version changed during job")

                def load_chunk(_: str, frame: pd.DataFrame = chunk) -> pd.DataFrame:
                    return frame

                batch = score_dataset(
                    job.dataset_id,
                    load_chunk,
                    predictor,
                    lambda: job_id,
                    self.clock,
                )
                for prediction in batch.predictions:
                    if not prediction.customer_id.strip():
                        raise ValueError("Customer IDs must be non-empty")
                    if (
                        not math.isfinite(prediction.churn_probability)
                        or not 0 <= prediction.churn_probability <= 1
                    ):
                        raise ValueError(
                            "Probabilities must be finite and within [0, 1]"
                        )
                explanations = []
                for offset, prediction in enumerate(batch.predictions):
                    if len(explained) >= job.explain_limit:
                        break
                    if requested and prediction.customer_id not in requested:
                        continue
                    if self.explain is None:
                        raise ValueError("SHAP requested but no explainer configured")
                    explanation = self.explain(job.model_version, chunk.iloc[[offset]])
                    if (
                        explanation.customer_id != prediction.customer_id
                        or explanation.model_version != job.model_version
                    ):
                        raise ValueError(
                            "Explanation customer/model does not match prediction"
                        )
                    explanations.append(explanation)
                    explained.add(prediction.customer_id)
                self.predictions.stage(batch, explanations, self.clock())
                job = self.jobs.update_progress(
                    job_id, job.processed_rows + len(batch.predictions)
                )
                job = self.jobs.renew_lease(job_id, self.clock())
            if requested - explained:
                raise ValueError(
                    f"{len(requested - explained)} requested explanation customer IDs not found"
                )
            return self.jobs.finish(job_id, self.clock())
        except Exception as exc:  # noqa: BLE001
            error = public_scoring_error(exc)
            return self.jobs.fail_if_running(job_id, self.clock(), error=error)
