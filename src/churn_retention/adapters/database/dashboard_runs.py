"""Isolated SQLite ledger for accepted Dashboard LLM runs, never model bundles."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from churn_retention.adapters.database.local_sqlite import connect_local
from churn_retention.contracts.application_security import reject_credentials
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_failure import FAILURE_MESSAGES, failure_message


def canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SavedRun:
    run_id: str
    request_key: str
    source_name: str
    cohort_json: str
    dataset_digest: str
    reference_version: str
    reference_sha256: str
    execution_mode: str
    predictor_ref: str
    status: str


class DashboardRunStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS dashboard_runs (
                    run_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                    source_name TEXT NOT NULL, cohort_json TEXT NOT NULL,
                    dataset_digest TEXT NOT NULL, reference_version TEXT NOT NULL,
                    reference_sha256 TEXT NOT NULL, execution_mode TEXT NOT NULL,
                    predictor_ref TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dashboard_outcomes (
                    run_id TEXT NOT NULL, customer_id TEXT NOT NULL,
                    state TEXT NOT NULL, result_json TEXT,
                    PRIMARY KEY(run_id, customer_id),
                    FOREIGN KEY(run_id) REFERENCES dashboard_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS dashboard_failures (
                    run_id TEXT NOT NULL, customer_id TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    PRIMARY KEY(run_id, customer_id),
                    FOREIGN KEY(run_id) REFERENCES dashboard_runs(run_id)
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect_local(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def reserve(
        self,
        frame: pd.DataFrame,
        source_name: str,
        request_id: str,
        reference_version: str,
        pin: str,
        mode: str,
        *,
        run_id: str | None = None,
    ) -> SavedRun:
        # pandas.to_json defaults to 10 decimal places and changes query vectors.
        # Native JSON float repr round-trips the exact validated Python float.
        reject_credentials([frame.to_dict(orient="records"), source_name, request_id])
        cohort = canonical(frame.to_dict(orient="records"))
        digest = hashlib.sha256(cohort.encode()).hexdigest()
        key = hashlib.sha256(
            canonical([request_id, digest, reference_version, pin, mode]).encode()
        ).hexdigest()
        identifier = run_id or str(uuid4())
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT run_id FROM dashboard_runs WHERE request_key=?", (key,)
            ).fetchone()
            if existing is not None:
                return self.get(existing["run_id"])
            # A collision is an error, never ON CONFLICT DO NOTHING with old scores.
            conn.execute(
                "INSERT INTO dashboard_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    key,
                    source_name,
                    cohort,
                    digest,
                    reference_version,
                    pin,
                    mode,
                    f"llm-rag-v2/{mode}/{reference_version}",
                    "PENDING",
                    datetime.now(UTC).isoformat(),
                ),
            )
        return self.get(identifier)

    def get(self, run_id: str) -> SavedRun:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM dashboard_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Run not found")
        reject_credentials(dict(row))
        return SavedRun(**{name: row[name] for name in SavedRun.__dataclass_fields__})

    def start(self, run_id: str) -> bool:
        with self.connect() as conn:
            return (
                conn.execute(
                    (
                        "UPDATE dashboard_runs SET status='RUNNING' WHERE run_id=? AN"
                        "D status='PENDING'"
                    ),
                    (run_id,),
                ).rowcount
                == 1
            )

    def begin_attempt(self, run_id: str, customer_id: str) -> None:
        # Persist BEFORE provider invocation. A crashed attempt cannot auto-repeat.
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT status,cohort_json FROM dashboard_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if (
                run is None
                or run["status"] != "RUNNING"
                or customer_id not in {row["customerID"] for row in json.loads(run["cohort_json"])}
            ):
                raise ValueError("Attempt must belong to a running accepted cohort")
            conn.execute(
                "INSERT INTO dashboard_outcomes VALUES (?,?,'CALLING',NULL)",
                (run_id, customer_id),
            )

    def outcome(
        self,
        run_id: str,
        customer_id: str,
        result: LLMChurnPrediction | None,
        *,
        error_code: str = "prediction_failed",
    ) -> None:
        if result is not None:
            reject_credentials(result.model_dump())
        with self.connect() as conn:
            changed = conn.execute(
                (
                    "UPDATE dashboard_outcomes SET state=?,result_json=? WHERE ru"
                    "n_id=? AND customer_id=? AND state='CALLING'"
                ),
                (
                    "SUCCESS" if result else "FAILED",
                    canonical(result.model_dump()) if result else None,
                    run_id,
                    customer_id,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("Outcome already finalized or attempt missing")
            if result is None:
                code = error_code if error_code in FAILURE_MESSAGES else "prediction_failed"
                conn.execute(
                    "INSERT INTO dashboard_failures VALUES (?,?,?)",
                    (run_id, customer_id, code),
                )

    def reject(self, run_id: str, code: str) -> None:
        """Fail closed without parsing a corrupt snapshot or changing accepted runs."""
        code = code if code in FAILURE_MESSAGES else "prediction_failed"
        with self.connect() as conn:
            if conn.execute(
                (
                    "UPDATE dashboard_runs SET status='FAILED' WHERE run_id=? AND"
                    " status IN ('PENDING','RUNNING')"
                ),
                (run_id,),
            ).rowcount:
                conn.execute(
                    "INSERT OR IGNORE INTO dashboard_failures VALUES (?, '', ?)",
                    (run_id, code),
                )

    def error_code(self, run_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                (
                    "SELECT error_code FROM dashboard_failures WHERE run_id=? ORD"
                    "ER BY customer_id='' DESC,customer_id LIMIT 1"
                ),
                (run_id,),
            ).fetchone()
        return row[0] if row is not None and row[0] in FAILURE_MESSAGES else "prediction_failed"

    def progress(self, run_id: str) -> pd.DataFrame:
        run = self.get(run_id)
        with self.connect() as conn:
            rows = conn.execute(
                (
                    "SELECT o.customer_id,o.state,f.error_code FROM dashboard_out"
                    "comes o LEFT JOIN dashboard_failures f ON f.run_id=o.run_id "
                    "AND f.customer_id=o.customer_id WHERE o.run_id=?"
                ),
                (run_id,),
            ).fetchall()
        states = {row["customer_id"]: row for row in rows}
        result = []
        for customer in json.loads(run.cohort_json):
            cid = customer["customerID"]
            row = states.get(cid)
            state = row["state"] if row else "NOT_RUN"
            code = (
                (row["error_code"] or "prediction_failed")
                if row is not None and state == "FAILED"
                else "interrupted"
                if state == "UNKNOWN"
                else "not_run"
                if state == "NOT_RUN"
                else ""
            )
            result.append(
                {
                    "customer_id": cid,
                    "state": state,
                    "error_code": code,
                    "message": failure_message(code) if code else "",
                }
            )
        return pd.DataFrame(result, columns=["customer_id", "state", "error_code", "message"])

    def finish(self, run_id: str, *, interrupted: bool = False) -> SavedRun:
        run = self.get(run_id)
        with self.connect() as conn:
            if interrupted:
                conn.execute(
                    (
                        "UPDATE dashboard_outcomes SET state='UNKNOWN' WHERE run_id=?"
                        " AND state='CALLING'"
                    ),
                    (run_id,),
                )
            succeeded = conn.execute(
                "SELECT COUNT(*) FROM dashboard_outcomes WHERE run_id=? AND state='SUCCESS'",
                (run_id,),
            ).fetchone()[0]
            complete = succeeded == len(json.loads(run.cohort_json)) and not interrupted
            status = "COMPLETE" if complete else "PARTIAL" if succeeded else "FAILED"
            conn.execute(
                (
                    "UPDATE dashboard_runs SET status=? WHERE run_id=? AND status"
                    " IN ('PENDING','RUNNING')"
                ),
                (status, run_id),
            )
        return self.get(run_id)

    def recover_interrupted_job(self, job_id: str) -> bool:
        """Operator-confirmed stop; fence job and run together, never resubmit API calls."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                "SELECT payload,status FROM task_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None or job["status"] != "RUNNING":
                return False
            run_id = json.loads(job["payload"])["run_id"]
            run = conn.execute(
                "SELECT status FROM dashboard_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError("Interrupted job has no linked run")
            if run["status"] in {"PENDING", "RUNNING"}:
                conn.execute(
                    (
                        "UPDATE dashboard_outcomes SET state='UNKNOWN' WHERE run_id=?"
                        " AND state='CALLING'"
                    ),
                    (run_id,),
                )
                count = conn.execute(
                    "SELECT COUNT(*) FROM dashboard_outcomes WHERE run_id=? AND state='SUCCESS'",
                    (run_id,),
                ).fetchone()[0]
                conn.execute(
                    "UPDATE dashboard_runs SET status=? WHERE run_id=?",
                    ("PARTIAL" if count else "FAILED", run_id),
                )
            # A worker can crash after committing COMPLETE but before completing its job.
            # Repair that job from durable outcomes without rerunning prediction.
            success = run["status"] == "COMPLETE"
            conn.execute(
                (
                    "UPDATE task_jobs SET status=?,error_type=?,error_reason=?,up"
                    "dated_at=? WHERE job_id=?"
                ),
                (
                    "SUCCESS" if success else "FAILED",
                    None if success else "Interrupted",
                    None if success else "Interrupted; no automatic API retry",
                    datetime.now(UTC).isoformat(),
                    job_id,
                ),
            )
            return True

    def frames(self, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        run = self.get(run_id)
        with self.connect() as conn:
            rows = conn.execute(
                (
                    "SELECT customer_id,result_json FROM dashboard_outcomes WHERE"
                    " run_id=? AND state='SUCCESS'"
                ),
                (run_id,),
            ).fetchall()
        for row in rows:
            reject_credentials(dict(row))
        outcomes = {
            row["customer_id"]: LLMChurnPrediction.model_validate_json(row["result_json"])
            for row in rows
        }
        for result in outcomes.values():
            reject_credentials(result.model_dump())
        customers = pd.DataFrame(json.loads(run.cohort_json))
        customers = customers[customers["customerID"].isin(outcomes)].copy().reset_index(drop=True)
        records: list[dict[str, Any]] = []
        for cid in customers["customerID"]:
            result = outcomes[cid]
            records.append(
                {
                    "customer_id": cid,
                    "churn_prob": result.churn_probability,
                    "churn_prediction": result.churn_prediction,
                    "reason": result.reason,
                    "provider": f"MOCK ({result.provider} route)"
                    if run.execution_mode == "mock"
                    else result.provider,
                    "reference_count": result.reference_count,
                    "reference_version": result.reference_version,
                    "probability_calibrated": result.probability_calibrated,
                    "execution_mode": run.execution_mode,
                }
            )
        return customers, pd.DataFrame(
            records,
            columns=[
                "customer_id",
                "churn_prob",
                "churn_prediction",
                "reason",
                "provider",
                "reference_count",
                "reference_version",
                "probability_calibrated",
                "execution_mode",
            ],
        )
