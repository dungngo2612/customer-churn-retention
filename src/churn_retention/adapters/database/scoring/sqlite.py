"""SQLite implements both job and prediction ports with atomic publication."""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Self

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.contracts.http.scoring import Prediction, PredictionBatch
from churn_retention.modules.scoring.jobs import (
    InvalidJobTransition,
    JobStatus,
    ResultsNotReady,
    ScoringJob,
)


class SQLiteScoringStore:
    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS scoring_jobs (
                job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS scoring_active_key
                ON scoring_jobs(idempotency_key)
                WHERE status IN ('PENDING', 'RUNNING', 'SUCCEEDED');
            CREATE TABLE IF NOT EXISTS scoring_predictions (
                job_id TEXT NOT NULL REFERENCES scoring_jobs(job_id),
                customer_id TEXT NOT NULL, position INTEGER NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (job_id, customer_id), UNIQUE (job_id, position)
            );
            CREATE TABLE IF NOT EXISTS scoring_explanations (
                job_id TEXT NOT NULL, customer_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (job_id, customer_id),
                FOREIGN KEY (job_id, customer_id)
                    REFERENCES scoring_predictions(job_id, customer_id) ON DELETE CASCADE
            );
        """)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()

    @staticmethod
    def _get(db: sqlite3.Connection, job_id: str) -> ScoringJob:
        row = db.execute(
            "SELECT payload FROM scoring_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Scoring job not found: {job_id}")
        return ScoringJob.from_dict(json.loads(row["payload"]))

    @staticmethod
    def _write(db: sqlite3.Connection, job: ScoringJob) -> None:
        db.execute(
            "UPDATE scoring_jobs SET status = ?, payload = ? WHERE job_id = ?",
            (job.status.value, json.dumps(job.to_dict()), job.job_id),
        )

    def get(self, job_id: str) -> ScoringJob:
        with self._lock:
            return self._get(self._db, job_id)

    @staticmethod
    def _active(db: sqlite3.Connection, key: str) -> ScoringJob | None:
        row = db.execute(
            """SELECT payload FROM scoring_jobs WHERE idempotency_key = ?
                            AND status IN ('PENDING', 'RUNNING', 'SUCCEEDED')""",
            (key,),
        ).fetchone()
        return ScoringJob.from_dict(json.loads(row["payload"])) if row else None

    def find_active(self, key: str) -> ScoringJob | None:
        with self._lock:
            return self._active(self._db, key)

    def create_or_get(self, job: ScoringJob) -> ScoringJob:
        if job.status != JobStatus.PENDING:
            raise InvalidJobTransition("New jobs must be PENDING")
        with self._transaction() as db:
            existing = self._active(db, job.idempotency_key)
            if existing is not None:
                return existing
            db.execute(
                "INSERT INTO scoring_jobs VALUES (?, ?, ?, ?)",
                (
                    job.job_id,
                    job.idempotency_key,
                    job.status.value,
                    json.dumps(job.to_dict()),
                ),
            )
        return job

    def claim(self, job_id: str, now: datetime) -> ScoringJob | None:
        with self._transaction() as db:
            job = self._get(db, job_id)
            if job.status != JobStatus.PENDING:
                return None  # Duplicate delivery never runs a job twice.
            running = job.transition(JobStatus.RUNNING, now)
            self._write(db, running)
            return running

    def update_progress(self, job_id: str, processed_rows: int) -> ScoringJob:
        with self._transaction() as db:
            job = self._get(db, job_id).progress(processed_rows)
            self._write(db, job)
            return job

    def finish(
        self, job_id: str, now: datetime, error: str | None = None
    ) -> ScoringJob:
        with self._transaction() as db:
            job = self._get(db, job_id)
            if error is None:
                count = db.execute(
                    "SELECT COUNT(*) FROM scoring_predictions WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
                if count != job.total_rows:
                    raise ValueError("Cannot publish an incomplete prediction set")
            status = JobStatus.FAILED if error is not None else JobStatus.SUCCEEDED
            job = job.transition(status, now, error)
            # Readers gate on this status in the same DB. No partial publication.
            self._write(db, job)
            return job

    def stage(
        self, batch: PredictionBatch, explanations: Sequence[ModelExplanation]
    ) -> None:
        with self._transaction() as db:
            job = self._get(db, batch.scoring_run_id)
            if job.status != JobStatus.RUNNING:
                raise InvalidJobTransition("Only RUNNING jobs may stage predictions")
            if batch.model_version != job.model_version:
                raise ValueError(
                    "Prediction model_version does not match pinned job model"
                )
            for offset, prediction in enumerate(batch.predictions):
                if (
                    prediction.dataset_id != job.dataset_id
                    or prediction.scoring_run_id != job.job_id
                ):
                    raise ValueError("Prediction dataset/run does not match job")
                try:
                    db.execute(
                        "INSERT INTO scoring_predictions VALUES (?, ?, ?, ?)",
                        (
                            job.job_id,
                            prediction.customer_id,
                            job.processed_rows + offset,
                            prediction.model_dump_json(),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "Duplicate customer_id within scoring run"
                    ) from exc
            count = db.execute(
                "SELECT COUNT(*) FROM scoring_explanations WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()[0]
            if count + len(explanations) > job.explain_limit:
                raise ValueError("Explanation count exceeds explain_limit")
            for explanation in explanations:
                if explanation.model_version != job.model_version:
                    raise ValueError("Explanation model_version does not match job")
                db.execute(
                    "INSERT INTO scoring_explanations VALUES (?, ?, ?)",
                    (
                        job.job_id,
                        explanation.customer_id,
                        explanation.model_dump_json(by_alias=True),
                    ),
                )

    def discard(self, job_id: str) -> None:
        with self._transaction() as db:
            if self._get(db, job_id).status == JobStatus.SUCCEEDED:
                raise InvalidJobTransition("Published results are immutable")
            db.execute("DELETE FROM scoring_predictions WHERE job_id = ?", (job_id,))

    def _ready(self, job_id: str) -> None:
        if self._get(self._db, job_id).status != JobStatus.SUCCEEDED:
            raise ResultsNotReady(
                "Predictions/explanations are visible only for SUCCEEDED jobs"
            )

    def predictions(self, job_id: str) -> list[Prediction]:
        with self._lock:
            self._ready(job_id)
            rows = self._db.execute(
                "SELECT payload FROM scoring_predictions WHERE job_id = ? ORDER BY position",
                (job_id,),
            )
            return [Prediction.model_validate_json(row[0]) for row in rows]

    def explanations(self, job_id: str) -> list[ModelExplanation]:
        with self._lock:
            self._ready(job_id)
            rows = self._db.execute(
                """SELECT e.payload FROM scoring_explanations e JOIN scoring_predictions p
                                      ON e.job_id = p.job_id AND e.customer_id = p.customer_id
                                      WHERE e.job_id = ? ORDER BY p.position""",
                (job_id,),
            )
            return [ModelExplanation.model_validate_json(row[0]) for row in rows]
