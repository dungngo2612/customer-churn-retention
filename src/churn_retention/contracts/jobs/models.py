"""Job and Task Queue contracts and data structures for background processing."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Trạng thái vòng đời của một tác vụ trong hàng đợi."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


class JobType(str, Enum):
    """Loại tác vụ bất đồng bộ."""

    SCORING = "SCORING"
    CAMPAIGN_SIMULATION = "CAMPAIGN_SIMULATION"


@dataclass(frozen=True)
class TaskJob:
    """Đặc tả một tác vụ nền với khóa chống trùng (Idempotency Key) và chính sách Retry."""

    job_id: str
    job_type: JobType
    idempotency_key: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    max_retries: int = 3
    retry_count: int = 0
    error_reason: str | None = None
    error_type: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.idempotency_key.strip():
            raise ValueError("Idempotency key cannot be empty")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in (self.max_retries, self.retry_count)
            )
            or self.retry_count > self.max_retries
        ):
            raise ValueError(
                "Retry counts must be non-negative integers within the limit"
            )

    @classmethod
    def create(
        cls,
        job_type: JobType,
        idempotency_key: str,
        payload: dict[str, Any],
        max_retries: int = 3,
    ) -> TaskJob:
        """Khởi tạo một tác vụ mới với UUID duy nhất."""
        now_iso = datetime.now(UTC).isoformat()
        return cls(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            idempotency_key=idempotency_key,
            payload=payload,
            status=JobStatus.PENDING,
            max_retries=max_retries,
            retry_count=0,
            error_reason=None,
            error_type=None,
            result=None,
            created_at=now_iso,
            updated_at=now_iso,
        )

    def with_status(
        self,
        new_status: JobStatus,
        result: dict[str, Any] | None = None,
        error_reason: str | None = None,
        error_type: str | None = None,
        increment_retry: bool = False,
    ) -> TaskJob:
        """Tạo bản sao mới với trạng thái cập nhật (đảm bảo tính bất biến)."""
        now_iso = datetime.now(UTC).isoformat()
        new_retry = self.retry_count + (1 if increment_retry else 0)
        return TaskJob(
            job_id=self.job_id,
            job_type=self.job_type,
            idempotency_key=self.idempotency_key,
            payload=self.payload,
            status=new_status,
            max_retries=self.max_retries,
            retry_count=new_retry,
            error_reason=None
            if new_status == JobStatus.SUCCESS
            else (error_reason if error_reason is not None else self.error_reason),
            error_type=None
            if new_status == JobStatus.SUCCESS
            else (error_type if error_type is not None else self.error_type),
            result=result if result is not None else self.result,
            created_at=self.created_at,
            updated_at=now_iso,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["job_type"] = self.job_type.value
        d["status"] = self.status.value
        return d
