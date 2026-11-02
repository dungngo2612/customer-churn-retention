"""Task Queue adapter with SQLite persistence, Idempotency, and status tracking."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from churn_retention.adapters.database.local_sqlite import connect_local
from churn_retention.contracts.application_security import reject_credentials
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob


class TaskQueue(Protocol):
    """Giao diện hàng đợi tác vụ hỗ trợ khóa chống trùng và truy vết trạng thái."""

    def enqueue(self, job: TaskJob) -> TaskJob:
        """Đưa tác vụ vào hàng đợi. Nếu idempotency_key đã tồn tại, trả về job cũ (Idempotent)."""
        ...

    def get_job(self, job_id: str) -> TaskJob | None:
        """Lấy thông tin tác vụ theo ID."""
        ...

    def get_by_idempotency_key(self, idempotency_key: str) -> TaskJob | None:
        """Lấy thông tin tác vụ theo khóa chống trùng."""
        ...

    def fetch_next_pending(self) -> TaskJob | None:
        """Lấy tác vụ tiếp theo sẵn sàng xử lý (PENDING hoặc RETRYING)."""
        ...

    def update_job(self, job: TaskJob) -> None:
        """Cập nhật trạng thái và kết quả của tác vụ."""
        ...

    def claim_next(self, job_id: str | None = None) -> TaskJob | None: ...

    def complete_claim(self, job: TaskJob, token: str) -> bool: ...

    def list_jobs(self, limit: int = 50) -> list[TaskJob]:
        """Liệt kê danh sách các tác vụ gần đây."""
        ...


class SqliteTaskQueue:
    """Hiện thực TaskQueue sử dụng SQLite với ràng buộc UNIQUE cho idempotency_key."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parents[4]
            data_dir = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = data_dir / "task_queue.db"
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = connect_local(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error_reason TEXT,
                    error_type TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_jobs_status
                ON task_jobs(status, created_at ASC);
                """
            )
            conn.commit()

    def enqueue(self, job: TaskJob) -> TaskJob:
        """Đưa tác vụ vào hàng đợi. Nếu trùng idempotency_key, trả về bản ghi hiện có mà không tạo mới."""
        reject_credentials(job.to_dict())
        existing = self.get_by_idempotency_key(job.idempotency_key)
        if existing is not None:
            self._same_request(existing, job)
            return existing

        payload_json = json.dumps(job.payload, ensure_ascii=False, allow_nan=False)
        result_json = (
            json.dumps(job.result, ensure_ascii=False, allow_nan=False)
            if job.result
            else None
        )

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO task_jobs (
                        job_id, job_type, idempotency_key, payload, status,
                        max_retries, retry_count, error_reason, error_type,
                        result, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.job_type.value,
                        job.idempotency_key,
                        payload_json,
                        job.status.value,
                        job.max_retries,
                        job.retry_count,
                        job.error_reason,
                        job.error_type,
                        result_json,
                        job.created_at,
                        job.updated_at,
                    ),
                )
                conn.commit()
                return job
            except sqlite3.IntegrityError:
                # Trùng lặp do race condition giữa 2 luồng đồng thời
                existing_after_race = self.get_by_idempotency_key(job.idempotency_key)
                if existing_after_race is not None:
                    self._same_request(existing_after_race, job)
                    return existing_after_race
                raise

    @staticmethod
    def _same_request(existing: TaskJob, submitted: TaskJob) -> None:
        if (
            existing.job_type != submitted.job_type
            or json.dumps(existing.payload, sort_keys=True, allow_nan=False)
            != json.dumps(submitted.payload, sort_keys=True, allow_nan=False)
            or existing.max_retries != submitted.max_retries
        ):
            raise ValueError("Idempotency key already belongs to a different request")

    def get_job(self, job_id: str) -> TaskJob | None:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM task_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def get_by_idempotency_key(self, idempotency_key: str) -> TaskJob | None:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM task_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def fetch_next_pending(self) -> TaskJob | None:
        """Lấy và khóa tác vụ đầu tiên đang ở trạng thái PENDING hoặc RETRYING."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM task_jobs
                WHERE status IN ('PENDING', 'RETRYING')
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def update_job(self, job: TaskJob) -> None:
        reject_credentials(job.to_dict())
        current = self.get_job(job.job_id)
        if current is None or current.status in {
            JobStatus.RUNNING,
            JobStatus.SUCCESS,
            JobStatus.FAILED,
        }:
            raise ValueError(
                "Use fenced completion; running/terminal jobs cannot be overwritten"
            )
        result_json = (
            json.dumps(job.result, ensure_ascii=False, allow_nan=False)
            if job.result
            else None
        )
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE task_jobs SET
                    status = ?,
                    retry_count = ?,
                    error_reason = ?,
                    error_type = ?,
                    result = ?,
                    updated_at = ?
                WHERE job_id = ? AND status IN ('PENDING','RETRYING')
                """,
                (
                    job.status.value,
                    job.retry_count,
                    job.error_reason,
                    job.error_type,
                    result_json,
                    job.updated_at,
                    job.job_id,
                ),
            )
            conn.commit()

    def claim_next(self, job_id: str | None = None) -> TaskJob | None:
        """Select and claim in one transaction, optionally ONLY a submitted job."""
        with self._get_connection() as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM task_jobs WHERE status IN ('PENDING','RETRYING') "
                "AND (? IS NULL OR job_id=?) ORDER BY created_at,job_id LIMIT 1",
                (job_id, job_id),
            ).fetchone()
            if row is None:
                return None
            claimed = self._row_to_job(row).with_status(JobStatus.RUNNING)
            conn.execute(
                "UPDATE task_jobs SET status='RUNNING',updated_at=? WHERE job_id=?",
                (claimed.updated_at, claimed.job_id),
            )
            return claimed

    def complete_claim(self, job: TaskJob, token: str) -> bool:
        """Fenced completion: an obsolete worker cannot replace terminal state."""
        reject_credentials(job.to_dict())
        with self._get_connection() as conn, conn:
            return (
                conn.execute(
                    "UPDATE task_jobs SET status=?,retry_count=?,error_reason=?,error_type=?,result=?,updated_at=? "
                    "WHERE job_id=? AND status='RUNNING' AND updated_at=?",
                    (
                        job.status.value,
                        job.retry_count,
                        job.error_reason,
                        job.error_type,
                        json.dumps(job.result, ensure_ascii=False, allow_nan=False)
                        if job.result
                        else None,
                        job.updated_at,
                        job.job_id,
                        token,
                    ),
                ).rowcount
                == 1
            )

    def fail_interrupted(self, job_id: str) -> bool:
        """Explicit operator action; NEVER requeue potentially billed API calls."""
        with self._get_connection() as conn, conn:
            return (
                conn.execute(
                    "UPDATE task_jobs SET status='FAILED',error_type='Interrupted',"
                    "error_reason='Interrupted; no automatic API retry' WHERE job_id=? AND status='RUNNING'",
                    (job_id,),
                ).rowcount
                == 1
            )

    def list_jobs(self, limit: int = 50) -> list[TaskJob]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM task_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_job(row) for row in cur.fetchall()]

    def list_running_jobs(self) -> list[TaskJob]:
        """Recovery must include older RUNNING jobs outside the recent-jobs panel."""
        with self._get_connection() as conn:
            return [
                self._row_to_job(row)
                for row in conn.execute(
                    "SELECT * FROM task_jobs WHERE status='RUNNING' ORDER BY created_at,job_id"
                )
            ]

    def _row_to_job(self, row: sqlite3.Row) -> TaskJob:
        reject_credentials(dict(row))
        payload = json.loads(row["payload"]) if row["payload"] else {}
        result = json.loads(row["result"]) if row["result"] else None

        return TaskJob(
            job_id=row["job_id"],
            job_type=JobType(row["job_type"]),
            idempotency_key=row["idempotency_key"],
            payload=payload,
            status=JobStatus(row["status"]),
            max_retries=row["max_retries"],
            retry_count=row["retry_count"],
            error_reason=row["error_reason"],
            error_type=row["error_type"],
            result=result,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
