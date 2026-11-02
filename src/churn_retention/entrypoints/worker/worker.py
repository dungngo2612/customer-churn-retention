"""Background worker with Idempotency, Retry policy, and Error classification."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from churn_retention.adapters.task_queue.queue import TaskQueue
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob

logger = logging.getLogger(__name__)


class PermanentError(Exception):
    """Lỗi vĩnh viễn không thể retry (ví dụ: sai schema, thiếu cột, dữ liệu hỏng)."""


class TransientError(Exception):
    """Lỗi tạm thời có thể retry (ví dụ: nghẽn mạng, database lock, timeout)."""


class IdempotentPredictionStore:
    """Kho lưu trữ dự đoán bảo đảm ràng buộc duy nhất (scoring_run_id, customer_id)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parents[4]
            data_dir = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = data_dir / "predictions.db"
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scoring_predictions (
                    scoring_run_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    churn_probability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scoring_run_id, customer_id)
                );
                """
            )
            conn.commit()

    def save_predictions_batch(
        self,
        scoring_run_id: str,
        predictions_df: pd.DataFrame,
        model_version: str,
    ) -> int:
        """Ghi kết quả dự đoán với cơ chế chống ghi trùng ON CONFLICT DO NOTHING.

        Trả về số dòng thực sự được chèn mới.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        inserted_count = 0

        with self._get_connection() as conn:
            for _, row in predictions_df.iterrows():
                cid = str(row["customer_id"])
                prob = float(row["churn_prob"])
                cur = conn.execute(
                    """
                    INSERT INTO scoring_predictions (
                        scoring_run_id, customer_id, churn_probability, model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(scoring_run_id, customer_id) DO NOTHING;
                    """,
                    (scoring_run_id, cid, prob, model_version, now_iso),
                )
                inserted_count += cur.rowcount
            conn.commit()

        return inserted_count

    def count_predictions(self, scoring_run_id: str) -> int:
        """Đếm số bản ghi dự đoán của một scoring_run_id cụ thể."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM scoring_predictions WHERE scoring_run_id = ?",
                (scoring_run_id,),
            )
            return cur.fetchone()[0]


class JobWorker:
    """Tiến trình Worker xử lý các tác vụ từ hàng đợi với chính sách retry thông minh."""

    def __init__(
        self,
        queue: TaskQueue,
        prediction_store: IdempotentPredictionStore | None = None,
        *,
        handlers: dict[JobType, Callable[[TaskJob], dict[str, Any]]] | None = None,
        sanitize_errors: bool = False,
    ) -> None:
        self.queue = queue
        self.sanitize_errors = sanitize_errors
        self.prediction_store = (
            (prediction_store or IdempotentPredictionStore())
            if handlers is None
            else prediction_store
        )
        self._handlers: dict[JobType, Callable[[TaskJob], dict[str, Any]]] = (
            {JobType.SCORING: self._handle_scoring}
            if handlers is None
            else dict(handlers)
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

    def _handle_scoring(self, job: TaskJob) -> dict[str, Any]:
        """Thực thi tác vụ chấm điểm scoring."""
        payload = job.payload
        dataset_path = payload.get("dataset_path")
        bundle_path = payload.get("bundle_path")
        scoring_run_id = payload.get("scoring_run_id", f"run-{job.job_id[:8]}")

        if not dataset_path or not Path(dataset_path).exists():
            raise PermanentError(f"File dataset không tồn tại: {dataset_path}")

        # Kiểm tra schema hợp lệ
        df = pd.read_csv(dataset_path)
        if "customerID" not in df.columns:
            raise PermanentError("Lỗi Schema: Thiếu cột bắt buộc 'customerID'")

        from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor

        if not bundle_path or not Path(bundle_path).exists():
            raise PermanentError(f"Model bundle không tồn tại: {bundle_path}")

        bp = BundlePredictor()
        bp.load(str(bundle_path))

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)

        probs = bp.predict_batch(features=df, customer_ids=df["customerID"].tolist())
        pred_df = pd.DataFrame(
            {
                "customer_id": df["customerID"].tolist(),
                "churn_prob": probs,
            }
        )

        # Lưu an toàn vào DB với ràng buộc duy nhất (scoring_run_id, customer_id)
        assert self.prediction_store is not None
        inserted = self.prediction_store.save_predictions_batch(
            scoring_run_id=scoring_run_id,
            predictions_df=pred_df,
            model_version=bp.model_version,
        )

        total_stored = self.prediction_store.count_predictions(scoring_run_id)

        return {
            "scoring_run_id": scoring_run_id,
            "model_version": bp.model_version,
            "total_customers": len(df),
            "inserted_new": inserted,
            "total_stored_in_db": total_stored,
        }
