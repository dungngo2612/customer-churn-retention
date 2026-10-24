"""Job and Task Queue contracts and data structures for background processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid


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
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    max_retries: int = 3
    retry_count: int = 0
    error_reason: Optional[str] = None
    error_type: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def create(
        cls,
        job_type: JobType,
        idempotency_key: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> TaskJob:
        """Khởi tạo một tác vụ mới với UUID duy nhất."""
        now_iso = datetime.now(timezone.utc).isoformat()
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
        result: Optional[Dict[str, Any]] = None,
        error_reason: Optional[str] = None,
        error_type: Optional[str] = None,
        increment_retry: bool = False,
    ) -> TaskJob:
        """Tạo bản sao mới với trạng thái cập nhật (đảm bảo tính bất biến)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        new_retry = self.retry_count + (1 if increment_retry else 0)
        return TaskJob(
            job_id=self.job_id,
            job_type=self.job_type,
            idempotency_key=self.idempotency_key,
            payload=self.payload,
            status=new_status,
            max_retries=self.max_retries,
            retry_count=new_retry,
            error_reason=error_reason if error_reason is not None else self.error_reason,
            error_type=error_type if error_type is not None else self.error_type,
            result=result if result is not None else self.result,
            created_at=self.created_at,
            updated_at=now_iso,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["job_type"] = self.job_type.value
        d["status"] = self.status.value
        return d
