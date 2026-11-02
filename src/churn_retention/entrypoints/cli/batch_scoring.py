"""Local batch commands. Queue is drained in this process; SQLite survives it."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from churn_retention.adapters.database.scoring.csv_reader import CsvDatasetReader
from churn_retention.adapters.database.scoring.sqlite import SQLiteScoringStore
from churn_retention.adapters.predictor.batch_models import BundleScoringModels
from churn_retention.adapters.task_queue.in_process import InProcessJobQueue
from churn_retention.entrypoints.worker.tasks.scoring import run_worker_once
from churn_retention.modules.scoring.jobs import (
    DEFAULT_LEASE_SECONDS,
    JobStatus,
    public_scoring_error,
)
from churn_retention.workflows.score_dataset import BatchScoringService, get_job_status


def register_commands(commands: Any) -> None:
    score = commands.add_parser("score-batch", help="Score CSV chunks using an in-process worker")
    score.add_argument("--dataset", required=True, type=Path)
    score.add_argument(
        "--dataset-id",
        help="Optional immutable dataset ID; default is resolved CSV path",
    )
    score.add_argument("--model-version", required=True)
    score.add_argument("--bundle-root", type=Path, default=Path("artifacts/bundles"))
    score.add_argument("--chunk-size", type=int, default=200)
    score.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    score.add_argument("--explain-limit", type=int, default=0)
    score.add_argument("--explain-customer-id", action="append", default=[])
    score.add_argument("--background-train", type=Path)
    status = commands.add_parser("job-status", help="Read a persisted job from SQLite")
    status.add_argument("--job-id", required=True)
    for parser in (score, status):
        parser.add_argument(
            "--state-db", type=Path, default=Path("artifacts/scoring/scoring.sqlite3")
        )


def run_command(args: argparse.Namespace) -> int:
    started = perf_counter()
    try:
        with SQLiteScoringStore(args.state_db) as store:
            if args.command == "job-status":
                job = get_job_status(args.job_id, store)
            else:
                path = args.dataset.resolve()
                dataset_id = args.dataset_id or str(path)
                queue = InProcessJobQueue()
                models = BundleScoringModels(args.bundle_root, args.background_train)
                service = BatchScoringService(
                    store,
                    store,
                    queue,
                    CsvDatasetReader({dataset_id: path}),
                    models.predictor_for,
                    explain=models.explain_one,
                    lease_seconds=args.lease_seconds,
                )
                job = service.create_job(
                    dataset_id,
                    args.model_version,
                    args.chunk_size,
                    explain_limit=args.explain_limit,
                    explain_customer_ids=args.explain_customer_id,
                )
                completed = run_worker_once(queue, service)
                if completed is not None:
                    job = completed
            print(
                json.dumps(
                    {
                        **job.to_dict(),
                        "processed_chunks": job.processed_chunks,
                        "elapsed_seconds": round(perf_counter() - started, 6),
                    },
                    ensure_ascii=False,
                )
            )
            return 1 if job.status == JobStatus.FAILED else 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": public_scoring_error(exc)}))
        return 1
