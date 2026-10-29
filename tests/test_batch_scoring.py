"""Batch boundaries and failure publication; no real test set is read."""

import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from churn_retention.adapters.database.scoring.csv_reader import CsvDatasetReader
from churn_retention.adapters.database.scoring.sqlite import SQLiteScoringStore
from churn_retention.adapters.predictor.batch_models import BundleScoringModels
from churn_retention.adapters.task_queue.in_process import InProcessJobQueue
from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.entrypoints.cli.__main__ import main
from churn_retention.entrypoints.worker.tasks.scoring import (
    handle_scoring_message,
    run_worker_once,
)
from churn_retention.ml.training.baseline import train_logistic
from churn_retention.ml.training.bundle import create_feature_schema, save_bundle
from churn_retention.modules.scoring.jobs import (
    InvalidJobTransition,
    JobStatus,
    ResultsNotReady,
    ScoringJob,
    idempotency_key,
)
from churn_retention.modules.scoring.use_cases import score_dataset
from churn_retention.workflows.score_dataset import BatchScoringService

NOW = datetime(2026, 10, 29, tzinfo=UTC)


class CountingPredictor:
    def __init__(self, version="v1"):
        self.model_version = version
        self.calls = []
        self.bad_on_call = None
        self.bad_value = float("nan")

    def load(self, artifact_path: str) -> None:
        pass

    def predict_batch(
        self, features: pd.DataFrame, customer_ids: Sequence[str]
    ) -> list[float]:
        self.calls.append(list(customer_ids))
        if len(self.calls) == self.bad_on_call:
            return [self.bad_value] * len(customer_ids)
        return (features["signal"].astype(float) / 100).tolist()


@pytest.fixture
def setup(tmp_path):
    frame = pd.DataFrame(
        {"customer_id": [f"c{i}" for i in range(7)], "signal": np.arange(1, 8)}
    )
    csv = tmp_path / "customers.csv"
    frame.to_csv(csv, index=False)
    with SQLiteScoringStore(tmp_path / "state.sqlite3") as store:
        queue, predictor = InProcessJobQueue(), CountingPredictor()
        service = BatchScoringService(
            store,
            store,
            queue,
            CsvDatasetReader({"dataset": csv}),
            lambda version: predictor,
            clock=lambda: NOW,
        )
        yield frame, store, queue, predictor, service


def explanation_for(version, row):
    customer = str(row.customer_id.iloc[0])
    probability = float(row.signal.iloc[0]) / 100
    base = math.log(probability / (1 - probability))
    return ModelExplanation.model_validate(
        {
            "customer_id": customer,
            "model_version": version,
            "output_type": "log_odds",
            "calibration_applied": "original",
            "base_value": base,
            "log_odds": base,
            "probability": probability,
            "background": {"seed": 42, "rows": 1, "customer_ids": ["train-0"]},
            "feature_schema": [
                {"name": "signal", "kind": "numeric", "dtype": "float64"}
            ],
            "contributions": [
                {
                    "customer_id": customer,
                    "model_version": version,
                    "feature_name": "signal",
                    "feature_value": float(row.signal.iloc[0]),
                    "contribution": 0.0,
                    "base_value": base,
                    "explained_output": "log_odds",
                    "calibration_applied": "original",
                }
            ],
            "detailed_contributions": [
                {
                    "feature_name": "num__signal",
                    "source_feature": "signal",
                    "transformed_value": 1.0,
                    "contribution": 0.0,
                }
            ],
        }
    )


def test_state_machine_and_key():
    job = ScoringJob("job", "dataset", "v1", 2, 3, NOW)
    running = job.transition(JobStatus.RUNNING, NOW)
    complete = running.progress(3).transition(JobStatus.SUCCEEDED, NOW)
    assert complete.started_at == NOW and complete.finished_at == NOW
    assert ScoringJob.from_dict(complete.to_dict()) == complete
    assert job.idempotency_key == idempotency_key("dataset", "v1")
    assert idempotency_key("ab", "c") != idempotency_key("a", "bc")
    for original, status in [
        (job, JobStatus.FAILED),
        (complete, JobStatus.RUNNING),
        (complete, JobStatus.SUCCEEDED),
        (running, JobStatus.PENDING),
        (running.transition(JobStatus.FAILED, NOW, "failure"), JobStatus.RUNNING),
    ]:
        with pytest.raises(InvalidJobTransition):
            original.transition(status, NOW)
    with pytest.raises(InvalidJobTransition, match="all rows"):
        running.transition(JobStatus.SUCCEEDED, NOW)
    with pytest.raises(ValueError, match="monotonic"):
        running.progress(2).progress(1)
    with pytest.raises(ValueError, match="require an error"):
        running.transition(JobStatus.FAILED, NOW)


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"explain_limit": -1},
        {"explain_limit": 0, "explain_customer_ids": ("c1",)},
        {"explain_limit": 2, "explain_customer_ids": ("c1", "c1")},
    ],
)
def test_invalid_job_configuration(changes):
    with pytest.raises(ValueError):
        replace(ScoringJob("j", "d", "v1", 2, 3, NOW), **changes)


def test_pending_idempotency_and_message_has_only_reference(setup):
    _, _, queue, _, service = setup
    first = service.create_job("dataset", "v1", 2)
    repeated = service.create_job("dataset", "v1", 5, explain_limit=1)
    assert (
        repeated == first and repeated.chunk_size == 2 and repeated.explain_limit == 0
    )
    assert queue.dequeue() == {"job_id": first.job_id}
    assert queue.dequeue() is None
    assert run_worker_once(queue, service) is None


def test_running_idempotency_and_duplicate_delivery(setup):
    _, store, queue, predictor, service = setup
    job = service.create_job("dataset", "v1", 2)
    running = store.claim(job.job_id, NOW)
    assert service.create_job("dataset", "v1") == running
    assert service.run_scoring_job(job.job_id) == running
    assert run_worker_once(queue, service) == running
    assert predictor.calls == []


def test_chunks_progress_visibility_and_one_shot_equivalence(setup, monkeypatch):
    frame, store, queue, predictor, service = setup
    job = service.create_job("dataset", "v1", 2)
    progress = []
    update = store.update_progress

    def record(job_id, rows):
        updated = update(job_id, rows)
        assert updated.status == JobStatus.RUNNING
        with pytest.raises(ResultsNotReady):
            store.predictions(job_id)
        progress.append(updated.processed_rows)
        return updated

    monkeypatch.setattr(store, "update_progress", record)
    result = run_worker_once(queue, service)
    assert result.status == JobStatus.SUCCEEDED
    assert result.processed_rows == 7 and result.processed_chunks == 4
    assert progress == [2, 4, 6, 7]
    assert list(map(len, predictor.calls)) == [2, 2, 2, 1]
    expected = score_dataset(
        "dataset", lambda _: frame, CountingPredictor(), lambda: job.job_id, lambda: NOW
    )
    assert store.predictions(job.job_id) == expected.predictions


def test_succeeded_idempotency_and_no_duplicate_predictions(setup):
    _, store, queue, predictor, service = setup
    job = service.create_job("dataset", "v1", 2)
    run_worker_once(queue, service)
    assert service.create_job("dataset", "v1").job_id == job.job_id
    assert service.run_scoring_job(job.job_id).status == JobStatus.SUCCEEDED
    assert len(predictor.calls) == 4
    assert len(store.predictions(job.job_id)) == 7
    assert queue.dequeue() is None
    with pytest.raises(InvalidJobTransition):
        store.finish(job.job_id, NOW)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_probability_fails_and_discards_partial_results(setup, bad):
    _, store, queue, predictor, service = setup
    predictor.bad_on_call, predictor.bad_value = 2, bad
    job = service.create_job("dataset", "v1", 2)
    failed = run_worker_once(queue, service)
    assert failed.status == JobStatus.FAILED and failed.error
    assert failed.processed_rows == 2 and failed.finished_at == NOW
    with pytest.raises(ResultsNotReady):
        store.predictions(job.job_id)
    assert (
        store._db.execute(
            "SELECT COUNT(*) FROM scoring_predictions WHERE job_id = ?", (job.job_id,)
        ).fetchone()[0]
        == 0
    )
    predictor.bad_on_call = None
    retry = service.create_job("dataset", "v1", 2)
    assert retry.job_id != job.job_id
    assert run_worker_once(queue, service).status == JobStatus.SUCCEEDED
    assert service.run_scoring_job(job.job_id) == failed
    assert len(store.predictions(retry.job_id)) == 7


@pytest.mark.parametrize(
    "limit,requested,expected",
    [
        (0, (), []),
        (1, (), ["c0"]),
        (3, (), ["c0", "c1", "c2"]),
        (2, ("c4", "c6"), ["c4", "c6"]),
    ],
)
def test_explain_limit_and_requested_ids(setup, limit, requested, expected):
    _, store, queue, _, service = setup
    calls = []

    def explain(version, row):
        calls.append(str(row.customer_id.iloc[0]))
        return explanation_for(version, row)

    service.explain = explain
    job = service.create_job(
        "dataset", "v1", 2, explain_limit=limit, explain_customer_ids=requested
    )
    assert run_worker_once(queue, service).status == JobStatus.SUCCEEDED
    assert calls == expected
    assert [item.customer_id for item in store.explanations(job.job_id)] == expected


def test_missing_requested_explanation_fails_without_publishing(setup):
    _, store, queue, _, service = setup
    service.explain = explanation_for
    job = service.create_job(
        "dataset", "v1", 2, explain_limit=1, explain_customer_ids=["absent"]
    )
    assert run_worker_once(queue, service).status == JobStatus.FAILED
    with pytest.raises(ResultsNotReady):
        store.explanations(job.job_id)


@pytest.mark.parametrize(
    "message",
    [
        {"job_id": "j", "customer_id": "private"},
        {"job_id": "j", "probability": 0.5},
        {"job_id": ""},
        {},
    ],
)
def test_worker_rejects_messages_with_payload(setup, message):
    _, _, queue, _, service = setup
    with pytest.raises(ValueError):
        queue.enqueue(message)
    with pytest.raises(ValueError):
        handle_scoring_message(message, service)


def test_atomic_idempotency_and_claim_across_connections(tmp_path):
    path = tmp_path / "shared.sqlite3"
    with SQLiteScoringStore(path) as first, SQLiteScoringStore(path) as second:
        jobs = [
            ScoringJob("a", "d", "v1", 2, 3, NOW),
            ScoringJob("b", "d", "v1", 2, 3, NOW),
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            created = list(
                pool.map(
                    lambda item: item[0].create_or_get(item[1]),
                    [(first, jobs[0]), (second, jobs[1])],
                )
            )
            assert created[0].job_id == created[1].job_id
            claimed = list(
                pool.map(
                    lambda store: store.claim(created[0].job_id, NOW), [first, second]
                )
            )
        assert sum(job is not None for job in claimed) == 1


@pytest.mark.parametrize("ids", [["same", "same", "other"], ["same", "other", "same"]])
def test_duplicate_customer_ids_within_or_across_chunks_fail(setup, tmp_path, ids):
    _, store, queue, _, service = setup
    path = tmp_path / "duplicate.csv"
    pd.DataFrame({"customer_id": ids, "signal": [1, 2, 3]}).to_csv(path, index=False)
    service.reader = CsvDatasetReader({"dataset": path})
    job = service.create_job("dataset", "v1", 2)
    assert run_worker_once(queue, service).status == JobStatus.FAILED
    with pytest.raises(ResultsNotReady):
        store.predictions(job.job_id)


def test_reader_preserves_string_ids_alias_and_excludes_label(tmp_path):
    path = tmp_path / "customers.csv"
    path.write_text("customerID,signal,Churn\n001,1,Yes\n002,2,No\n", encoding="utf-8")
    reader = CsvDatasetReader({"id": path})
    assert reader.total_rows("id") == 2
    chunks = list(reader.chunks("id", 1))
    assert [str(frame.customer_id.iloc[0]) for frame in chunks] == ["001", "002"]
    assert all("Churn" not in frame for frame in chunks)


def test_model_version_mismatch_is_failed(setup):
    _, _, queue, predictor, service = setup
    predictor.model_version = "unexpected"
    service.create_job("dataset", "v1", 2)
    failed = run_worker_once(queue, service)
    assert failed.status == JobStatus.FAILED and "pinned" in failed.error


def test_probability_endpoints_are_valid(setup, tmp_path):
    _, store, queue, _, service = setup
    path = tmp_path / "endpoints.csv"
    pd.DataFrame({"customer_id": ["zero", "one"], "signal": [0, 100]}).to_csv(
        path, index=False
    )
    service.reader = CsvDatasetReader({"dataset": path})
    job = service.create_job("dataset", "v1", 1)
    assert run_worker_once(queue, service).status == JobStatus.SUCCEEDED
    assert [p.churn_probability for p in store.predictions(job.job_id)] == [0.0, 1.0]


def test_changed_row_count_never_publishes_partial_results(setup, tmp_path):
    frame, store, queue, _, service = setup
    path = tmp_path / "mutable.csv"
    frame.to_csv(path, index=False)
    service.reader = CsvDatasetReader({"dataset": path})
    job = service.create_job("dataset", "v1", 2)
    frame.iloc[:-1].to_csv(path, index=False)
    failed = run_worker_once(queue, service)
    assert failed.status == JobStatus.FAILED and failed.processed_rows == 6
    with pytest.raises(ResultsNotReady):
        store.predictions(job.job_id)


@pytest.fixture
def local_bundle(tmp_path):
    train = pd.DataFrame(
        {
            "customerID": [f"train-{i}" for i in range(40)],
            "signal": np.arange(40, dtype=float),
            "group": ["a", "b"] * 20,
        }
    )
    frame = train[["signal", "group"]]
    config = {
        "numeric_features": ["signal"],
        "categorical_features": ["group"],
        "logistic": {"C": 1.0, "max_iter": 500},
    }
    pipeline = train_logistic(frame, (train.signal > 20).astype(int), config, 42, False)
    schema = create_feature_schema(frame, ["signal"], ["group"])
    root = tmp_path / "bundles"
    save_bundle(
        pipeline,
        root / "v1",
        schema,
        {"model_version": "v1", "model_type": "logistic", "class_weight": False},
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {
            "customerID": ["val-a", "val-b", "val-c"],
            "signal": [1.0, 20.0, 35.0],
            "group": ["a", "b", "a"],
        }
    ).to_csv(tmp_path / "val.csv", index=False)
    return root, pipeline, schema


def test_exact_model_version_remains_pinned_after_new_bundle(
    setup, local_bundle, tmp_path
):
    _, store, queue, _, service = setup
    root, pipeline, schema = local_bundle
    models = BundleScoringModels(root)
    service.reader = CsvDatasetReader({"dataset": tmp_path / "val.csv"})
    service.predictor_for = models.predictor_for
    job = service.create_job("dataset", "v1", 2)
    save_bundle(
        pipeline, root / "v2", schema, {"model_version": "v2", "model_type": "logistic"}
    )
    assert run_worker_once(queue, service).status == JobStatus.SUCCEEDED
    assert store.get(job.job_id).model_version == "v1"
    assert {p.model_version for p in store.predictions(job.job_id)} == {"v1"}
    full = next(
        CsvDatasetReader({"dataset": tmp_path / "val.csv"}).chunks("dataset", 100)
    )
    expected = models.predictor_for("v1").predict_batch(
        full.drop(columns="customer_id"), full.customer_id.tolist()
    )
    assert [
        p.churn_probability for p in store.predictions(job.job_id)
    ] == pytest.approx(expected, abs=1e-12, rel=0)


def test_cli_persists_status_and_uses_m23_only_for_requested_rows(
    local_bundle, tmp_path, capsys
):
    root, _, _ = local_bundle
    database = tmp_path / "cli.sqlite3"
    arguments = [
        "score-batch",
        "--dataset",
        str(tmp_path / "val.csv"),
        "--model-version",
        "v1",
        "--bundle-root",
        str(root),
        "--state-db",
        str(database),
        "--chunk-size",
        "2",
        "--explain-limit",
        "2",
        "--background-train",
        str(tmp_path / "train.csv"),
    ]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SUCCEEDED" and result["processed_chunks"] == 2
    assert (
        main(["job-status", "--job-id", result["job_id"], "--state-db", str(database)])
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["job_id"] == result["job_id"] and status["processed_rows"] == 3
    with SQLiteScoringStore(database) as store:
        assert len(store.predictions(result["job_id"])) == 3
        explanations = store.explanations(result["job_id"])
        assert [item.customer_id for item in explanations] == ["val-a", "val-b"]
        assert not any(
            set(item.background.customer_ids) & {"val-a", "val-b", "val-c"}
            for item in explanations
        )
