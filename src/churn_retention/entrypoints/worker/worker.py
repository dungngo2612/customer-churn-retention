"""Background worker with Idempotency, Retry policy, and Error classification."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from churn_retention.adapters.task_queue.queue import TaskQueue
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob

logger = logging.getLogger(__name__)


class PermanentError(Exception):
    """Lỗi vĩnh viễn không thể retry (ví dụ: sai schema, thiếu cột, dữ liệu hỏng)."""


class TransientError(Exception):
    """Lỗi tạm thời có thể retry (ví dụ: nghẽn mạng, database lock, timeout)."""


class JobWorker:
    """Tiến trình Worker xử lý các tác vụ từ hàng đợi với chính sách retry thông minh."""

    def __init__(
        self,
        queue: TaskQueue,
        *,
        handlers: dict[JobType, Callable[[TaskJob], dict[str, Any]]] | None = None,
        sanitize_errors: bool = False,
    ) -> None:
        self.queue = queue
        self.sanitize_errors = sanitize_errors
        self._handlers: dict[JobType, Callable[[TaskJob], dict[str, Any]]] = (
            {} if handlers is None else dict(handlers)
        )

    def register_handler(
        self,
        job_type: JobType,
        handler: Callable[[TaskJob], dict[str, Any]],
    ) -> None:
        self._handlers[job_type] = handler

    def process_one(self, job_id: str | None = None) -> TaskJob | None:
        """Lấy một tác vụ PENDING hoặc RETRYING và thực thi nó."""
        job = self.queue.claim_next(job_id)
        if job is None:
            return None

        # Chuyển trạng thái sang RUNNING
        running_job = job

        handler = self._handlers.get(running_job.job_type)
        if handler is None:
            failed_job = running_job.with_status(
                JobStatus.FAILED,
                error_reason=f"Không có handler cho job_type: {running_job.job_type}",
                error_type="ConfigurationError",
            )
            if not self.queue.complete_claim(failed_job, running_job.updated_at):
                return self.queue.get_job(running_job.job_id)
            return failed_job

        try:
            result = handler(running_job)
            success_job = running_job.with_status(JobStatus.SUCCESS, result=result)
            if not self.queue.complete_claim(success_job, running_job.updated_at):
                return self.queue.get_job(running_job.job_id)
            return success_job

        except (PermanentError, ValueError, KeyError) as e:
            # LỖI VĨNH VIỄN (Schema sai, thiếu cột): Chuyển FAILED ngay lập tức, KHÔNG RETRY
            failed_job = running_job.with_status(
                JobStatus.FAILED,
                error_reason="Tác vụ không hoàn tất; không tự gọi lại LLM."
                if self.sanitize_errors
                else str(e),
                error_type="PermanentError",
            )
            if not self.queue.complete_claim(failed_job, running_job.updated_at):
                return self.queue.get_job(running_job.job_id)
            return failed_job

        except Exception as e:  # noqa: BLE001 -- Worker boundary classifies and sanitizes handler failures.
            # LỖI TẠM THỜI (Transient): Cho phép retry có giới hạn (tối đa max_retries)
            if running_job.retry_count < running_job.max_retries:
                retry_job = running_job.with_status(
                    JobStatus.RETRYING,
                    error_reason="Tác vụ chưa hoàn tất."
                    if self.sanitize_errors
                    else f"Lỗi tạm thời ({type(e).__name__}): {e}",
                    error_type="TransientError",
                    increment_retry=True,
                )
                if not self.queue.complete_claim(retry_job, running_job.updated_at):
                    return self.queue.get_job(running_job.job_id)
                return retry_job
            else:
                # Đã vượt quá số lần retry cho phép
                exhausted_job = running_job.with_status(
                    JobStatus.FAILED,
                    error_reason="Tác vụ không hoàn tất; không tự gọi lại LLM."
                    if self.sanitize_errors
                    else f"Đã thử lại tối đa {running_job.max_retries} lần nhưng vẫn thất bại: {e}",
                    error_type="MaxRetriesExceeded",
                )
                if not self.queue.complete_claim(exhausted_job, running_job.updated_at):
                    return self.queue.get_job(running_job.job_id)
                return exhausted_job
