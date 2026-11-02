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
from churn_retention.entrypoints.cli import batch_scoring as batch_cli
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
from churn_retention.workflows.score_dataset import BatchScoringService, get_job_status

NOW = datetime(2026, 10, 29, tzinfo=UTC)


class CountingPredictor:
    def __init__(self, version="v1"):
        self.model_version = version
        self.calls = []
        self.bad_on_call = None
        self.bad_value = float("nan")

    def load(self, artifact_path: str) -> None:
        pass

    def predict_batch(self, features: pd.DataFrame, customer_ids: Sequence[str]) -> list[float]:
        self.calls.append(list(customer_ids))
        if len(self.calls) == self.bad_on_call:
            return [self.bad_value] * len(customer_ids)
        return (features["signal"].astype(float) / 100).tolist()


@pytest.fixture
def setup(tmp_path):
    frame = pd.DataFrame({"customer_id": [f"c{i}" for i in range(7)], "signal": np.arange(1, 8)})
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
            "feature_schema": [{"name": "signal", "kind": "numeric", "dtype": "float64"}],
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
    assert repeated == first and repeated.chunk_size == 2 and repeated.explain_limit == 0
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
    job = service.create_job("dataset", "v1", 2, explain_limit=1, explain_customer_ids=["absent"])
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
                pool.map(lambda store: store.claim(created[0].job_id, NOW), [first, second])
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
    assert failed.status == JobStatus.FAILED
    assert failed.error == "ValueError: batch scoring failed"
    assert predictor.calls == []


def test_probability_endpoints_are_valid(setup, tmp_path):
    _, store, queue, _, service = setup
    path = tmp_path / "endpoints.csv"
    pd.DataFrame({"customer_id": ["zero", "one"], "signal": [0, 100]}).to_csv(path, index=False)
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


def test_exact_model_version_remains_pinned_after_new_bundle(setup, local_bundle, tmp_path):
    _, store, queue, _, service = setup
    root, pipeline, schema = local_bundle
    models = BundleScoringModels(root)
    service.reader = CsvDatasetReader({"dataset": tmp_path / "val.csv"})
    service.predictor_for = models.predictor_for
    job = service.create_job("dataset", "v1", 2)
    save_bundle(pipeline, root / "v2", schema, {"model_version": "v2", "model_type": "logistic"})
    assert run_worker_once(queue, service).status == JobStatus.SUCCEEDED
    assert store.get(job.job_id).model_version == "v1"
    assert {p.model_version for p in store.predictions(job.job_id)} == {"v1"}
    full = next(CsvDatasetReader({"dataset": tmp_path / "val.csv"}).chunks("dataset", 100))
    expected = models.predictor_for("v1").predict_batch(
        full.drop(columns="customer_id"), full.customer_id.tolist()
    )
    assert [p.churn_probability for p in store.predictions(job.job_id)] == pytest.approx(
        expected, abs=1e-12, rel=0
    )


def test_cli_persists_status_and_uses_m23_only_for_requested_rows(local_bundle, tmp_path, capsys):
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
    assert main(["job-status", "--job-id", result["job_id"], "--state-db", str(database)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["job_id"] == result["job_id"] and status["processed_rows"] == 3
    with SQLiteScoringStore(database) as store:
        assert len(store.predictions(result["job_id"])) == 3
        explanations = store.explanations(result["job_id"])
        assert [item.customer_id for item in explanations] == ["val-a", "val-b"]
        assert not any(
            set(item.background.customer_ids) & {"val-a", "val-b", "val-c"} for item in explanations
        )


class FakeClock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now

    def advance(self, seconds):
        from datetime import timedelta

        self.now += timedelta(seconds=seconds)


def crash_after_first_chunk(service, predictor, monkeypatch):
    original = predictor.predict_batch
    calls = 0

    def predict(features, customer_ids):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit("simulated worker death")
        return original(features, customer_ids)

    monkeypatch.setattr(predictor, "predict_batch", predict)
    job = service.create_job("dataset", "v1", 2)
    with pytest.raises(SystemExit):
        service.run_scoring_job(job.job_id)
    monkeypatch.setattr(predictor, "predict_batch", original)
    return job


def test_running_lease_before_expiry_reuses_crashed_job(setup, monkeypatch):
    _, store, _, predictor, service = setup
    clock = FakeClock()
    service.clock, service.lease_seconds = clock, 30
    job = crash_after_first_chunk(service, predictor, monkeypatch)
    clock.advance(29)
    running = service.create_job("dataset", "v1", 2)
    assert running.job_id == job.job_id and running.status == JobStatus.RUNNING
    assert running.processed_rows == 2
    assert store._db.execute("SELECT COUNT(*) FROM scoring_jobs").fetchone()[0] == 1
    assert service.run_scoring_job(job.job_id) == running
    assert len(predictor.calls) == 1


def test_expired_lease_discards_staging_and_restarts_from_zero(setup, monkeypatch):
    _, store, _, predictor, service = setup
    clock = FakeClock()
    service.clock, service.lease_seconds = clock, 30
    service.explain = explanation_for
    # Include explanation staging so recovery also proves FK cascade cleanup.
    original_create = service.create_job

    def create(*args, **kwargs):
        return original_create(*args, **kwargs, explain_limit=1)

    monkeypatch.setattr(service, "create_job", create)
    old = crash_after_first_chunk(service, predictor, monkeypatch)
    assert store._db.execute("SELECT COUNT(*) FROM scoring_explanations").fetchone()[0] == 1
    clock.advance(30)  # Equality is expired, not a grace period.
    fresh = service.create_job("dataset", "v1", 2)
    failed = store.get(old.job_id)
    assert failed.status == JobStatus.FAILED and failed.error == "lease expired"
    assert failed.finished_at == clock() and failed.processed_rows == 2
    assert fresh.job_id != old.job_id and fresh.processed_rows == 0
    assert store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0] == 0
    assert store._db.execute("SELECT COUNT(*) FROM scoring_explanations").fetchone()[0] == 0
    assert service.run_scoring_job(fresh.job_id).status == JobStatus.SUCCEEDED
    assert len(store.predictions(fresh.job_id)) == 7
    assert store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0] == 7
    assert len({(p.scoring_run_id, p.customer_id) for p in store.predictions(fresh.job_id)}) == 7
    assert service.run_scoring_job(old.job_id) == failed
    assert service.run_scoring_job(fresh.job_id).status == JobStatus.SUCCEEDED
    assert len(store.predictions(fresh.job_id)) == 7


def test_lease_renews_after_every_chunk(setup, monkeypatch):
    from datetime import timedelta

    _, store, _, predictor, service = setup
    clock = FakeClock()
    service.clock, service.lease_seconds = clock, 30
    predict = predictor.predict_batch
    renew = store.renew_lease
    deadlines = []

    def slow_predict(features, customer_ids):
        clock.advance(5)
        return predict(features, customer_ids)

    def record_renewal(job_id, now):
        job = renew(job_id, now)
        deadlines.append(job.lease_expires_at)
        assert job.lease_expires_at == now + timedelta(seconds=30)
        assert job.processed_rows == min(len(deadlines) * 2, 7)
        return job

    monkeypatch.setattr(predictor, "predict_batch", slow_predict)
    monkeypatch.setattr(store, "renew_lease", record_renewal)
    job = service.create_job("dataset", "v1", 2)
    done = service.run_scoring_job(job.job_id)
    assert done.status == JobStatus.SUCCEEDED
    assert deadlines == [NOW + timedelta(seconds=s) for s in (35, 40, 45, 50)]
    assert done.lease_expires_at is None


def test_expired_chunk_cannot_publish_or_revive_lease(setup, monkeypatch):
    _, store, _, predictor, service = setup
    clock = FakeClock()
    service.clock, service.lease_seconds = clock, 5
    original = predictor.predict_batch

    def slow(features, customer_ids):
        clock.advance(5)
        return original(features, customer_ids)

    monkeypatch.setattr(predictor, "predict_batch", slow)
    job = service.create_job("dataset", "v1", 2)
    failed = service.run_scoring_job(job.job_id)
    assert failed.status == JobStatus.FAILED and failed.error == "lease expired"
    assert store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0] == 0


def test_stale_worker_cannot_overwrite_reclaimed_job(setup, monkeypatch):
    _, store, _, predictor, service = setup
    clock = FakeClock()
    service.clock, service.lease_seconds = clock, 5
    original = predictor.predict_batch
    fresh_jobs = []

    def reclaim_while_old_worker_is_paused(features, customer_ids):
        clock.advance(5)
        fresh = service.create_job("dataset", "v1", 2)
        fresh_jobs.append(fresh)
        monkeypatch.setattr(predictor, "predict_batch", original)
        assert service.run_scoring_job(fresh.job_id).status == JobStatus.SUCCEEDED
        return original(features, customer_ids)

    monkeypatch.setattr(predictor, "predict_batch", reclaim_while_old_worker_is_paused)
    old = service.create_job("dataset", "v1", 2)
    failed = service.run_scoring_job(old.job_id)
    assert failed.error == "lease expired" and failed.status == JobStatus.FAILED
    assert len(store.predictions(fresh_jobs[0].job_id)) == 7
    assert store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0] == 7


def test_legacy_sqlite_migration_treats_missing_lease_as_expired(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    old = ScoringJob("old", "dataset", "v1", 2, 7, NOW).transition(JobStatus.RUNNING, NOW)
    payload = old.to_dict()
    payload.pop("lease_expires_at")
    payload.pop("lease_seconds")
    with sqlite3.connect(path) as db:
        db.execute(
            (
                "CREATE TABLE scoring_jobs (job_id TEXT PRIMARY KEY, idempote"
                "ncy_key TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NO"
                "T NULL)"
            )
        )
        db.execute(
            "INSERT INTO scoring_jobs VALUES (?, ?, ?, ?)",
            (old.job_id, old.idempotency_key, old.status.value, json.dumps(payload)),
        )
    with SQLiteScoringStore(path) as store:
        assert store.get("old").lease_expires_at is None
        assert store.find_or_recover(old.idempotency_key, NOW) is None
        assert store.get("old").error == "lease expired"
        new = store.create_or_get(ScoringJob("new", "dataset", "v1", 2, 7, NOW))
        assert new.job_id == "new"
    with SQLiteScoringStore(path) as store:
        assert store.get("new").status == JobStatus.PENDING


def test_expired_recovery_is_atomic_across_connections(tmp_path):
    from datetime import timedelta

    path = tmp_path / "shared-lease.sqlite3"
    with SQLiteScoringStore(path) as first, SQLiteScoringStore(path) as second:
        old = ScoringJob("old", "dataset", "v1", 2, 7, NOW, lease_seconds=5)
        first.create_or_get(old)
        first.claim(old.job_id, NOW)
        now = NOW + timedelta(seconds=5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = list(
                pool.map(
                    lambda pair: pair[0].create_or_get(
                        ScoringJob(pair[1], "dataset", "v1", 2, 7, now), now
                    ),
                    [(first, "new-a"), (second, "new-b")],
                )
            )
        assert jobs[0].job_id == jobs[1].job_id
        assert first.get(old.job_id).error == "lease expired"


@pytest.mark.parametrize("source", ["load", "read", "predict", "explain", "stage"])
def test_worker_failures_redact_exception_data(setup, monkeypatch, source):
    _, store, queue, predictor, service = setup
    service.explain = explanation_for

    def fail(*args, **kwargs):
        raise ValueError("customer_id=private-customer signal=private-feature-value")

    targets = {
        "load": (service, "predictor_for"),
        "read": (service.reader, "chunks"),
        "predict": (predictor, "predict_batch"),
        "explain": (service, "explain"),
        "stage": (store, "stage"),
    }
    target, method = targets[source]
    monkeypatch.setattr(target, method, fail)
    job = service.create_job("dataset", "v1", 2, explain_limit=1)
    failed = run_worker_once(queue, service)
    assert failed.status == JobStatus.FAILED
    assert failed.error == "ValueError: batch scoring failed"
    assert store.get(job.job_id).error == failed.error
    payload = store._db.execute(
        "SELECT payload FROM scoring_jobs WHERE job_id = ?", (job.job_id,)
    ).fetchone()[0]
    assert "private-customer" not in payload
    assert "private-feature-value" not in payload
    with pytest.raises(ResultsNotReady):
        store.predictions(job.job_id)


def test_cli_validation_error_redacts_customer_and_feature_values(local_bundle, tmp_path, capsys):
    root, _, _ = local_bundle
    dataset = tmp_path / "invalid.csv"
    dataset.write_text(
        "customer_id,signal,group\nprivate-customer,private-feature-value,a\n",
        encoding="utf-8",
    )
    database = tmp_path / "invalid.sqlite3"
    assert (
        main(
            [
                "score-batch",
                "--dataset",
                str(dataset),
                "--model-version",
                "v1",
                "--bundle-root",
                str(root),
                "--state-db",
                str(database),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["status"] == "FAILED"
    assert result["error"] == "ValueError: batch scoring failed"
    assert "private-customer" not in output.out + output.err
    assert "private-feature-value" not in output.out + output.err
    with SQLiteScoringStore(database) as store:
        assert store.get(result["job_id"]).error == result["error"]


@pytest.mark.parametrize(
    "error_type", [FileNotFoundError, OSError, ValueError, RuntimeError, KeyError]
)
def test_cli_failures_redact_exception_data(tmp_path, capsys, monkeypatch, error_type):
    def fail(*args, **kwargs):
        raise error_type("customer_id=private-customer signal=private-feature-value")

    monkeypatch.setattr(SQLiteScoringStore, "__init__", fail)
    assert (
        main(
            [
                "job-status",
                "--job-id",
                "job",
                "--state-db",
                str(tmp_path / "state.sqlite3"),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": f"{error_type.__name__}: batch scoring failed"}
    assert "private-customer" not in output.out + output.err
    assert "private-feature-value" not in output.out + output.err


@pytest.mark.parametrize("status", list(JobStatus))
def test_cli_job_status_uses_read_use_case_without_loading_inputs(
    setup, tmp_path, capsys, monkeypatch, status
):
    from datetime import timedelta

    _, store, queue, predictor, service = setup
    if status == JobStatus.RUNNING:
        job = crash_after_first_chunk(service, predictor, monkeypatch)
    else:
        job = service.create_job("dataset", "v1", 2)
        if status == JobStatus.FAILED:
            predictor.bad_on_call = 1
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            run_worker_once(queue, service)
    expected = store.get(job.job_id)
    assert expected.status == status
    if status == JobStatus.RUNNING:
        assert not expected.has_active_lease(NOW + timedelta(seconds=301))
    staged = store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0]
    calls = []

    def query(job_id, jobs):
        calls.append(job_id)
        return get_job_status(job_id, jobs)

    def forbid(*args, **kwargs):
        pytest.fail("job-status must only read persisted job state")

    monkeypatch.setattr(batch_cli, "get_job_status", query)
    for name in (
        "CsvDatasetReader",
        "BundleScoringModels",
        "InProcessJobQueue",
        "run_worker_once",
    ):
        monkeypatch.setattr(batch_cli, name, forbid)
    for name in ("find_or_recover", "create_or_get", "claim", "renew_lease", "finish"):
        monkeypatch.setattr(SQLiteScoringStore, name, forbid)
    (tmp_path / "customers.csv").unlink()
    exit_code = main(
        [
            "job-status",
            "--job-id",
            job.job_id,
            "--state-db",
            str(tmp_path / "state.sqlite3"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert calls == [job.job_id]
    assert exit_code == (1 if status == JobStatus.FAILED else 0)
    for field, value in json.loads(json.dumps(expected.to_dict())).items():
        assert result[field] == value
    assert result["processed_chunks"] == expected.processed_chunks
    assert store.get(job.job_id) == expected
    assert store._db.execute("SELECT COUNT(*) FROM scoring_predictions").fetchone()[0] == staged


def test_cli_job_status_missing_job_returns_safe_error(setup, tmp_path, capsys, monkeypatch):
    calls = []

    def query(job_id, jobs):
        calls.append(job_id)
        return get_job_status(job_id, jobs)

    monkeypatch.setattr(batch_cli, "get_job_status", query)
    assert (
        main(
            [
                "job-status",
                "--job-id",
                "private-customer",
                "--state-db",
                str(tmp_path / "state.sqlite3"),
            ]
        )
        == 1
    )
    assert calls == ["private-customer"]
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "KeyError: batch scoring failed"}
    assert "private-customer" not in output.out + output.err
