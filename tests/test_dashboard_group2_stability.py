"""Group 2 acceptance: real service/router/provider adapters, fake SDK I/O only."""

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from google.genai import errors
from groq import APIConnectionError, APITimeoutError, RateLimitError
from streamlit.testing.v1 import AppTest
from test_dashboard_llm_integration import setup as setup  # noqa: PLC0414 -- Re-export the shared pytest fixture.
from test_dashboard_llm_integration import source as source  # noqa: PLC0414 -- Re-export the shared pytest fixture.
from test_llm_prediction_service import PRIVATE, VALID
from test_llm_prediction_service import sdk as sdk  # noqa: PLC0414 -- Re-export the shared pytest fixture.
from test_llm_router import Clock

from churn_retention.adapters.database.dashboard_runs import canonical
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as ui
from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
    DashboardLLMAdapter,
    validate_frame,
)
from churn_retention.entrypoints.streamlit.llm_input import (
    DEFAULT_REFERENCE_SHA256,
)
from churn_retention.entrypoints.worker.worker import JobWorker
from churn_retention.modules.scoring import llm_router

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src/churn_retention/dashboard/app.py"


def live_adapter(setup, monkeypatch):
    base, _ = setup
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    monkeypatch.setenv("CHURN_DASHBOARD_PREVIEW", "0")
    return DashboardLLMAdapter(base.resources, base.pin, base.store, mode="live")


def sdk_failure(kind: str, role: str):
    request = httpx.Request("POST", "https://offline.invalid")
    if kind == "json":
        return PRIVATE
    if role == "Groq":
        return {
            "rate_limit": RateLimitError(
                PRIVATE, response=httpx.Response(429, request=request), body={}
            ),
            "timeout": APITimeoutError(request=request),
            "connection": APIConnectionError(message=PRIVATE, request=request),
        }[kind]
    return {
        "rate_limit": errors.ClientError(429, {"message": PRIVATE}),
        "timeout": httpx.ReadTimeout(PRIVATE),
        "connection": httpx.ConnectError(PRIVATE),
    }[kind]


def test_real_artifact_is_exactly_train_4930_and_retrieval_never_heldout(
    approved_rag_reference: Path,
) -> None:
    store = ReferenceStore.load(
        approved_rag_reference, expected_artifact_sha256=DEFAULT_REFERENCE_SHA256
    )
    splits = pd.read_csv(ROOT / "artifacts/splits/dataset_splits.csv", dtype={"customerID": str})
    train = pd.read_csv(ROOT / "data/processed/train.csv", dtype={"customerID": str})
    ref_ids = {row.customer_id for row in store.records}
    split_train = set(splits.loc[splits["split"] == "train", "customerID"])
    heldout = set(splits.loc[splits["split"] != "train", "customerID"])
    exact_train = ref_ids == set(train.customerID) == split_train
    no_heldout = ref_ids.isdisjoint(heldout)
    assert len(ref_ids) == len(store.records) == 4930 and exact_train and no_heldout
    assert store.matrix.shape == (4930, 45)
    assert store.manifest.reference_split == "train" and store.manifest.fit.target_used is False
    assert (
        hashlib.sha256((ROOT / "data/processed/train.csv").read_bytes()).hexdigest()
        == store.manifest.source_train_sha256
    )
    query = {
        str(name): value for name, value in train.iloc[0].drop(["customerID", "Churn"]).items()
    }
    neighbors = SimilarCustomerRetriever(store, k=5).retrieve(query)
    only_train = all(
        row.customer_id in ref_ids and row.customer_id not in heldout for row in neighbors
    )
    assert len(neighbors) == 5 and only_train


def test_snapshot_round_trip_preserves_every_float_and_query_vector(setup) -> None:
    adapter, calls = setup
    frame = ui.synthetic_frame().iloc[:1].copy()
    frame["MonthlyCharges"] = 19.123456789012345
    frame["TotalCharges"] = 229.48148146814813
    clean = validate_frame(frame, adapter.resources)
    before = {name: clean.iloc[0][name] for name in adapter.resources.feature_names}
    run = adapter.prepare(frame, "synthetic-high-precision.csv", "precision")
    restored = json.loads(run.cohort_json)[0]
    assert restored["MonthlyCharges"] == before["MonthlyCharges"]
    expected = adapter.resources.store.prepare_query(before).vector
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.SUCCESS
    observed = calls.call_args.args[1]
    assert observed == before
    np.testing.assert_array_equal(adapter.resources.store.prepare_query(observed).vector, expected)


@pytest.mark.parametrize("mode", ["unknown", "LIVE", "", None])
def test_unknown_mode_cannot_bypass_live_gate(setup, mode) -> None:
    adapter, calls = setup
    with pytest.raises(ValueError, match="mode"):
        DashboardLLMAdapter(
            adapter.resources, adapter.pin, adapter.store, mode=mode, predict_one=calls
        )
    calls.assert_not_called()


def test_worker_rejects_oversized_saved_batch_even_when_prepare_was_bypassed(
    setup,
) -> None:
    adapter, calls = setup
    frame = pd.concat([ui.synthetic_frame().iloc[:1]] * 6, ignore_index=True)
    frame["customerID"] = [f"SYNTHETIC-{i}" for i in range(6)]
    run = adapter.store.reserve(
        frame,
        "synthetic",
        "bypass",
        adapter.resources.store.reference_version,
        adapter.pin,
        "mock",
    )
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.FAILED
    assert adapter.store.get(run.run_id).status == "FAILED"
    calls.assert_not_called()


@pytest.mark.parametrize("kind", ["json", "digest", "feature", "reference"])
def test_worker_rejects_corrupt_persisted_snapshot_without_calling_provider(setup, kind) -> None:
    adapter, calls = setup
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "corrupt")
    with adapter.store.connect() as conn:
        if kind == "json":
            conn.execute(
                "UPDATE dashboard_runs SET cohort_json='{' WHERE run_id=?",
                (run.run_id,),
            )
        elif kind == "digest":
            conn.execute(
                "UPDATE dashboard_runs SET dataset_digest=? WHERE run_id=?",
                ("0" * 64, run.run_id),
            )
        elif kind == "reference":
            conn.execute(
                "UPDATE dashboard_runs SET reference_version='rag-v2-wrong' WHERE run_id=?",
                (run.run_id,),
            )
        else:
            values = json.loads(run.cohort_json)
            values[0]["tenure"] = -1
            payload = canonical(values)
            conn.execute(
                "UPDATE dashboard_runs SET cohort_json=?,dataset_digest=? WHERE run_id=?",
                (payload, hashlib.sha256(payload.encode()).hexdigest(), run.run_id),
            )
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.FAILED
    assert adapter.store.get(run.run_id).status == "FAILED"
    calls.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [("provider", None), ("reference_count", 4), ("reference_version", "rag-v2-wrong")],
)
def test_response_provenance_is_required_before_success_persistence(setup, field, value) -> None:
    adapter, calls = setup
    response = {
        **json.loads(VALID),
        "provider": "Groq",
        "reference_count": 5,
        "reference_version": adapter.resources.store.reference_version,
        field: value,
    }
    calls.return_value = LLMChurnPrediction.model_validate(response)
    calls.side_effect = None
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "bad-response")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.FAILED
    assert adapter.store.frames(run.run_id)[1].empty
    assert adapter.store.error_code(run.run_id) == "provenance"


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "connection", "json"])
def test_groq_failure_falls_back_to_gemini_once_and_persists_complete_output(
    setup, sdk, monkeypatch, kind
) -> None:
    adapter = live_adapter(setup, monkeypatch)
    sdk.outcomes["Groq"] = sdk_failure(kind, "Groq")
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "fallback")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.SUCCESS
    _, results = adapter.store.frames(run.run_id)
    row = results.iloc[0]
    assert row["provider"] == "Gemini" and row["execution_mode"] == "live"
    assert row["churn_prediction"] == 1 and row["churn_prob"] == 0.6 and row["reason"]
    assert row["reference_count"] == 5 and row["reference_version"] == run.reference_version
    assert all(c.request.await_count == 1 for c in sdk.clients)
    ui.execute(adapter, run)
    assert len(sdk.clients) == 2


@pytest.mark.parametrize(
    "kind,code",
    [
        ("rate_limit", "rate_limit"),
        ("timeout", "timeout"),
        ("connection", "connection"),
        ("json", "invalid_response"),
    ],
)
def test_both_provider_failures_are_not_success_and_are_safe_durable_and_nonretrying(
    setup, sdk, monkeypatch, kind, code
) -> None:
    adapter = live_adapter(setup, monkeypatch)
    sdk.outcomes.update(Groq=sdk_failure(kind, "Groq"), Gemini=sdk_failure(kind, "Gemini"))
    run = adapter.prepare(ui.synthetic_frame().iloc[:3], "synthetic", "both-fail")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.FAILED
    assert adapter.store.get(run.run_id).status == "FAILED"
    assert adapter.store.frames(run.run_id)[1].empty
    assert adapter.store.progress(run.run_id).state.tolist() == [
        "FAILED",
        "NOT_RUN",
        "NOT_RUN",
    ]
    assert adapter.store.error_code(run.run_id) == code
    assert all(c.request.await_count == 1 for c in sdk.clients)
    queue = SqliteTaskQueue(adapter.store.path)
    repeat = queue.enqueue(
        TaskJob.create(
            JobType.SCORING,
            "different-job-same-failed-run",
            {"run_id": run.run_id},
            max_retries=0,
        )
    )
    worker = JobWorker(
        queue,
        handlers={JobType.SCORING: lambda item: adapter.process(str(item.payload["run_id"]))},
        sanitize_errors=True,
    )
    replay = worker.process_one(repeat.job_id)
    assert replay is not None and replay.status == JobStatus.FAILED
    assert len(sdk.clients) == 2 and PRIVATE.encode() not in adapter.store.path.read_bytes()


def test_hedge_at_60_runs_providers_in_parallel_and_drains_loser(setup, sdk, monkeypatch) -> None:
    adapter = live_adapter(setup, monkeypatch)
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def slow_primary():
        clock.advance(60)
        await asyncio.Event().wait()

    async def fallback():
        assert sdk.clients[0].active  # Both requests overlap, not sequential retry.
        return VALID

    sdk.outcomes.update(Groq=slow_primary, Gemini=fallback)
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "hedge")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.SUCCESS
    assert adapter.store.frames(run.run_id)[1].iloc[0]["provider"] == "Gemini"
    assert sdk.clients[0].cancelled and all(c.request.await_count == 1 for c in sdk.clients)


def test_hedged_gemini_failure_still_accepts_healthy_groq(setup, sdk, monkeypatch) -> None:
    adapter = live_adapter(setup, monkeypatch)
    clock = Clock()
    release = asyncio.Event()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def primary():
        clock.advance(60)
        await release.wait()
        return VALID

    async def fallback():
        assert sdk.clients[0].active
        release.set()
        raise httpx.ConnectError(PRIVATE)

    sdk.outcomes.update(Groq=primary, Gemini=fallback)
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "hedged-failure")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.SUCCESS
    assert adapter.store.frames(run.run_id)[1].iloc[0]["provider"] == "Groq"


def test_overall_120s_deadline_does_not_trigger_worker_retry(setup, sdk, monkeypatch) -> None:
    adapter = live_adapter(setup, monkeypatch)
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def blocked():
        clock.advance(120)
        await asyncio.Event().wait()

    sdk.outcomes["Groq"] = blocked
    run = adapter.prepare(ui.synthetic_frame().iloc[:2], "synthetic", "deadline")
    job = ui.execute(adapter, run)
    assert job is not None and job.status == JobStatus.FAILED
    assert adapter.store.error_code(run.run_id) == "deadline"
    assert adapter.store.frames(run.run_id)[1].empty
    assert len(sdk.clients) == 2 and all(c.request.await_count <= 1 for c in sdk.clients)


def test_exactly_five_rows_and_repeated_submit_do_not_exceed_request_budget(
    setup, sdk, monkeypatch
) -> None:
    adapter = live_adapter(setup, monkeypatch)
    frame = pd.concat(
        [
            ui.synthetic_frame(),
            ui.synthetic_frame().iloc[:1].assign(customerID="SYNTHETIC-FIFTH"),
        ],
        ignore_index=True,
    )
    run = adapter.prepare(frame, "synthetic", "five")
    job = ui.execute(adapter, run)
    again = ui.execute(adapter, run)
    assert job is not None and again is not None and job.job_id == again.job_id
    assert len(adapter.store.frames(run.run_id)[1]) == 5
    assert len(sdk.clients) == 10 and sum(c.request.await_count for c in sdk.clients) == 5


def test_concurrent_double_submit_only_invokes_generation_once(setup) -> None:
    adapter, calls = setup
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "synthetic", "concurrent")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: ui.execute(adapter, run), range(2)))
    assert calls.call_count == 1 and adapter.store.get(run.run_id).status == "COMPLETE"


@pytest.mark.parametrize(
    "feature", ui.synthetic_frame().columns.drop(["customerID", "Churn"]).tolist()
)
def test_all_nineteen_features_are_required_before_any_generation(setup, feature) -> None:
    adapter, calls = setup
    with pytest.raises(ValueError):
        adapter.prepare(ui.synthetic_frame().drop(columns=feature), "synthetic", "missing")
    calls.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", True),
        ("MonthlyCharges", float("inf")),
        ("MonthlyCharges", float("nan")),
        ("MonthlyCharges", -0.1),
        ("TotalCharges", ""),
        ("PaymentMethod", "unknown"),
        ("Contract", ["Month-to-month"]),
    ],
)
def test_wrong_types_missing_values_and_outliers_rejected(setup, field, value) -> None:
    adapter, calls = setup
    frame = ui.synthetic_frame().iloc[:1].astype(object)
    frame.at[0, field] = value
    with pytest.raises(ValueError):
        adapter.prepare(frame, "synthetic", "invalid")
    calls.assert_not_called()


def test_zero_tenure_missing_total_charges_uses_existing_preprocessing_rule(
    setup,
) -> None:
    adapter, _ = setup
    frame = ui.synthetic_frame().iloc[:1].copy().astype(object)
    frame.at[0, "tenure"] = 0
    frame.at[0, "TotalCharges"] = ""
    run = adapter.prepare(frame, "synthetic", "zero-tenure")
    assert json.loads(run.cohort_json)[0]["TotalCharges"] == 0.0


def test_live_confirmation_resets_and_failure_ui_keeps_mock_snapshot_honest(
    setup, sdk, monkeypatch
) -> None:
    base, _ = setup
    mock_run = base.prepare(ui.synthetic_frame().iloc[:1], "synthetic-mock", "mock-seed")
    ui.execute(base, mock_run)
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", mock_run.run_id)
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    monkeypatch.setenv("CHURN_DASHBOARD_PREVIEW", "0")
    monkeypatch.setattr(ui, "resources", lambda: base.resources)
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.radio(key="cuong_llm.provider_mode").set_value("Live API (Groq/Gemini)").run()
    assert not at.exception and not at.checkbox(key="cuong_llm.confirm").value
    assert at.button(key="cuong_llm.submit").disabled and not sdk.clients
    sdk.outcomes.update(
        Groq=sdk_failure("rate_limit", "Groq"),
        Gemini=sdk_failure("rate_limit", "Gemini"),
    )
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.button(key="cuong_llm.submit").click().run()
    assert not at.exception and at.error
    assert any("giới hạn" in error.value for error in at.error)
    assert at.session_state[ui.ACTIVE_RUN_KEY] == mock_run.run_id
    latest = base.store.get(at.session_state[ui.LAST_ATTEMPT_KEY])
    assert latest.status == "FAILED" and base.store.frames(latest.run_id)[1].empty
    assert any("MOCK provider" in entry.value for entry in at.markdown)
    at.run()
    assert not at.exception and at.error and len(sdk.clients) == 2
    at.button(key="cuong_llm.new_request").click().run()
    assert not at.checkbox(key="cuong_llm.confirm").value and len(sdk.clients) == 2
