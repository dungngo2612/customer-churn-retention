"""Immutable batch jobs and their legal lifecycle; no storage/queue imports."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

DEFAULT_LEASE_SECONDS = 300
LEASE_EXPIRED_ERROR = "lease expired"


class LeaseExpired(RuntimeError):
    pass


def public_scoring_error(error: Exception) -> str:
    """Keep the error category without serializing customer data or input values."""
    if isinstance(error, LeaseExpired):
        return LEASE_EXPIRED_ERROR
    return f"{type(error).__name__}: batch scoring failed"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class InvalidJobTransition(ValueError):
    pass


class ResultsNotReady(RuntimeError):
    pass


def idempotency_key(dataset_id: str, model_version: str) -> str:
    # JSON framing distinguishes ('ab', 'c') from ('a', 'bc').
    value = json.dumps([dataset_id, model_version], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoringJob:
    job_id: str
    dataset_id: str
    model_version: str
    chunk_size: int
    total_rows: int
    created_at: datetime
    processed_rows: int = 0
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    lease_expires_at: datetime | None = None
    explain_limit: int = 0
    explain_customer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.job_id, self.dataset_id, self.model_version)):
            raise ValueError("job_id, dataset_id and model_version must be non-empty")
        if self.chunk_size <= 0 or not 0 <= self.processed_rows <= self.total_rows:
            raise ValueError("chunk_size must be positive and 0 <= processed_rows <= total_rows")
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if self.explain_limit < 0:
            raise ValueError("explain_limit must be non-negative")
        if len(self.explain_customer_ids) > self.explain_limit:
            raise ValueError(
                "Requested customer IDs must fit within explain_limit (0 disables SHAP)"
            )
        if len(set(self.explain_customer_ids)) != len(self.explain_customer_ids) or any(
            not value.strip() for value in self.explain_customer_ids
        ):
            raise ValueError("Requested explanation customer IDs must be non-empty and unique")
        for value in (
            self.created_at,
            self.started_at,
            self.finished_at,
            self.lease_expires_at,
        ):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("Job timestamps must be timezone-aware")

    @property
    def idempotency_key(self) -> str:
        return idempotency_key(self.dataset_id, self.model_version)

    @property
    def processed_chunks(self) -> int:
        return (self.processed_rows + self.chunk_size - 1) // self.chunk_size

    def transition(
        self, status: JobStatus, now: datetime, error: str | None = None
    ) -> "ScoringJob":
        allowed = {
            JobStatus.PENDING: {JobStatus.RUNNING},
            JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED},
        }
        if status not in allowed.get(self.status, set()):
            raise InvalidJobTransition(f"Invalid job transition: {self.status} -> {status}")
        if status == JobStatus.RUNNING:
            return replace(
                self,
                status=status,
                started_at=now,
                lease_expires_at=now + timedelta(seconds=self.lease_seconds),
            )
        if status == JobStatus.SUCCEEDED and self.processed_rows != self.total_rows:
            raise InvalidJobTransition("Cannot succeed before all rows are processed")
        if status == JobStatus.FAILED and not error:
            raise ValueError("FAILED jobs require an error")
        return replace(self, status=status, finished_at=now, error=error, lease_expires_at=None)

    def has_active_lease(self, now: datetime) -> bool:
        return (
            self.status == JobStatus.RUNNING
            and self.lease_expires_at is not None
            and self.lease_expires_at > now
        )

    def require_lease(self, now: datetime) -> None:
        if self.status != JobStatus.RUNNING:
            raise InvalidJobTransition("Only RUNNING jobs may use a lease")
        if not self.has_active_lease(now):
            raise LeaseExpired(LEASE_EXPIRED_ERROR)

    def renew_lease(self, now: datetime) -> "ScoringJob":
        self.require_lease(now)
        return replace(self, lease_expires_at=now + timedelta(seconds=self.lease_seconds))

    def progress(self, processed_rows: int) -> "ScoringJob":
        if self.status != JobStatus.RUNNING:
            raise InvalidJobTransition("Only RUNNING jobs can record progress")
        if not self.processed_rows <= processed_rows <= self.total_rows:
            raise ValueError("Progress must be monotonic and cannot exceed total_rows")
        return replace(self, processed_rows=processed_rows)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["idempotency_key"] = self.idempotency_key
        for name in ("created_at", "started_at", "finished_at", "lease_expires_at"):
            value = getattr(self, name)
            result[name] = value.isoformat() if value is not None else None
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScoringJob":
        fields = dict(value)
        fields.pop("idempotency_key", None)
        fields.setdefault("lease_expires_at", None)
        fields["status"] = JobStatus(fields["status"])
        fields["explain_customer_ids"] = tuple(fields["explain_customer_ids"])
        for name in ("created_at", "started_at", "finished_at", "lease_expires_at"):
            fields[name] = (
                datetime.fromisoformat(fields[name]) if fields[name] is not None else None
            )
        return cls(**fields)
