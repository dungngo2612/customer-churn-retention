"""Cường Dashboard → v2 mock service → SQLite → Retention/Campaigns, offline."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Lock
from unittest.mock import Mock

import pandas as pd
import pytest
from dashboard_navigation import apply_settings, go, plan_setup, sample_analysis, save_campaign
from llm_v2_fixtures import make_source
from streamlit.testing.v1 import AppTest

from churn_retention.adapters.database.campaigns.repository import (
    SqliteCampaignRepository,
)
from churn_retention.adapters.database.dashboard_runs import DashboardRunStore
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as ui
from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
    DashboardLLMAdapter,
    mock_prediction,
    read_csv,
    validate_frame,
)
from churn_retention.entrypoints.streamlit.llm_input import form_resources
from churn_retention.entrypoints.streamlit.llm_prediction_service import (
    predict_customer,
)
from churn_retention.entrypoints.worker.worker import JobWorker
from churn_retention.modules.campaigns import CampaignStatus, create_campaign
from churn_retention.modules.retention.budget import select_actions_under_budget

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src/churn_retention/dashboard/app.py"


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    source = make_source(tmp_path_factory.mktemp("cuong-dashboard-v2"), n=8)
    source.build()
    return source


@pytest.fixture
def setup(source, tmp_path, monkeypatch):
    # AppTest installs a mocked Runtime, but opens no listener. Transport policy
    # is verified separately with real sockets; preserve business-only assertions.
    from churn_retention.entrypoints.streamlit import demo_security

    monkeypatch.setattr(demo_security, "require_safe_runtime", lambda: None)
    pin = hashlib.sha256(source.output.read_bytes()).hexdigest()
    form = form_resources(ReferenceStore.load(source.output, expected_artifact_sha256=pin))
    monkeypatch.setenv("CHURN_LLM_REFERENCE_ARTIFACT", str(source.output))
    monkeypatch.setenv("CHURN_LLM_REFERENCE_SHA256", pin)
    monkeypatch.setenv("CHURN_DASHBOARD_STATE", str(tmp_path / "preview.sqlite3"))
    monkeypatch.setenv("CHURN_DASHBOARD_PREVIEW", "1")
    monkeypatch.delenv("CHURN_DASHBOARD_DEMO_RUN", raising=False)
    ledger = DashboardRunStore(tmp_path / "preview.sqlite3")
    calls = Mock(side_effect=mock_prediction)
    adapter = DashboardLLMAdapter(form, pin, ledger, predict_one=calls)
    return adapter, calls


def test_upload_selected_rows_persist_alignment_and_retention(setup) -> None:
    adapter, calls = setup
    frame = ui.synthetic_frame()
    frame["customerID"] = ["0001", "0002", "0003", "0004"]
    loaded = read_csv(frame.to_csv(index=False).encode(), adapter.resources)
    assert loaded.customerID.tolist() == ["0001", "0002", "0003", "0004"]
    selected = loaded.iloc[[1, 0]].copy()
    run = adapter.prepare(selected, "new-upload.csv", "request-upload")
    job = ui.execute(adapter, run)
    assert job is not None
    assert job.status == JobStatus.SUCCESS and calls.call_count == 2
    assert adapter.store.get(run.run_id).status == "COMPLETE"
    for call in calls.call_args_list:
        assert len(call.args[1]) == 19 and not {"Churn", "customerID"} & set(call.args[1])
    customers, predictions = adapter.store.frames(run.run_id)
    assert customers.customerID.tolist() == predictions.customer_id.tolist() == ["0002", "0001"]
    assert predictions.churn_prediction.tolist() == [0, 1]
    assert customers.Churn.tolist() == [
        "No",
        "No",
    ]  # Predicted 1 is still observed active.
    allocation = select_actions_under_budget(customers, predictions, budget=20.0)
    assert len(allocation) == 2 and allocation.cost.sum() <= 20.0
    assert all(predictions.reason.str.contains("MOCK"))
    # Re-open SQLite; accepted outputs/provenance survive process/UI reruns.
    reopened = DashboardRunStore(adapter.store.path)
    pd.testing.assert_frame_equal(predictions, reopened.frames(run.run_id)[1])


def test_same_request_is_idempotent_different_payload_and_new_request_get_new_run(
    setup,
) -> None:
    adapter, calls = setup
    frame = ui.synthetic_frame().iloc[:1]
    first = adapter.prepare(frame, "same.csv", "same-request")
    ui.execute(adapter, first)
    repeated = adapter.prepare(frame, "same.csv", "same-request")
    ui.execute(adapter, repeated)
    assert repeated.run_id == first.run_id and calls.call_count == 1
    changed = frame.copy()
    changed["MonthlyCharges"] = 80.0
    assert adapter.prepare(changed, "same.csv", "same-request").run_id != first.run_id
    assert adapter.prepare(frame, "same.csv", "explicit-new-request").run_id != first.run_id


def test_run_id_collision_is_error_not_silent_old_result(setup) -> None:
    adapter, calls = setup
    frame = validate_frame(ui.synthetic_frame().iloc[:1], adapter.resources)
    run = adapter.prepare(frame, "first.csv", "first")
    ui.execute(adapter, run)
    changed = frame.copy()
    changed["MonthlyCharges"] = 90.0
    with pytest.raises(sqlite3.IntegrityError):
        adapter.store.reserve(
            changed,
            "second.csv",
            "second",
            run.reference_version,
            run.reference_sha256,
            "mock",
            run_id=run.run_id,
        )
    assert adapter.store.get(run.run_id).cohort_json == run.cohort_json
    assert calls.call_count == 1


@pytest.mark.parametrize(
    "problem",
    [
        "missing_feature",
        "extra_column",
        "duplicate_id",
        "missing_id",
        "unknown_category",
        "invalid_churn",
        "bad_numeric",
        "missing_total",
    ],
)
def test_invalid_csv_never_calls_llm(setup, problem) -> None:
    adapter, calls = setup
    frame = ui.synthetic_frame()
    if problem == "missing_feature":
        frame = frame.drop(columns="gender")
    elif problem == "extra_column":
        frame["unexpected"] = 1
    elif problem == "duplicate_id":
        frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    elif problem == "missing_id":
        frame.loc[0, "customerID"] = ""
    elif problem == "unknown_category":
        frame.loc[0, "Contract"] = "not-a-contract"
    elif problem == "invalid_churn":
        frame.loc[0, "Churn"] = "predicted-label"
    elif problem == "bad_numeric":
        frame["MonthlyCharges"] = "oops"
    elif problem == "missing_total":
        frame["TotalCharges"] = ""
    with pytest.raises(ValueError):
        read_csv(frame.to_csv(index=False).encode(), adapter.resources)
    calls.assert_not_called()


@pytest.mark.parametrize("content", [b"", b"\xff", b"a,a\n1,2\n", b"x" * 2_000_001])
def test_malformed_csv_is_rejected_before_inference(setup, content) -> None:
    adapter, calls = setup
    with pytest.raises(ValueError):
        read_csv(content, adapter.resources)
    calls.assert_not_called()


def test_large_batch_and_live_preview_are_blocked(setup, monkeypatch) -> None:
    adapter, calls = setup
    frame = pd.concat([ui.synthetic_frame()] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="1–5"):
        adapter.prepare(frame, "big.csv", "big")
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    with pytest.raises(ValueError, match="disabled"):
        DashboardLLMAdapter(adapter.resources, adapter.pin, adapter.store, mode="live")
    monkeypatch.setenv("CHURN_DASHBOARD_PREVIEW", "0")
    live = DashboardLLMAdapter(adapter.resources, adapter.pin, adapter.store, mode="live")
    assert live.predict_one is predict_customer  # Wiring verified; never invoke the real API.
    calls.assert_not_called()


def test_partial_error_is_fail_fast_saved_and_not_retried(setup) -> None:
    adapter, calls = setup

    def fail_second(store, features):
        if calls.call_count == 2:
            raise RuntimeError("PRIVATE_SDK_ERROR_NEVER_STORE")
        return mock_prediction(store, features)

    calls.side_effect = fail_second
    run = adapter.prepare(ui.synthetic_frame().iloc[:3], "partial.csv", "partial")
    job = ui.execute(adapter, run)
    assert job is not None
    assert job.status == JobStatus.FAILED and job.max_retries == 0
    assert (
        adapter.store.get(run.run_id).status == "PARTIAL"
        and len(adapter.store.frames(run.run_id)[0]) == 1
    )
    assert calls.call_count == 2
    ui.execute(adapter, run)
    assert calls.call_count == 2
    assert b"PRIVATE_SDK_ERROR" not in adapter.store.path.read_bytes()


def test_worker_atomic_claim_only_once_and_only_requested_job(tmp_path) -> None:
    queue = SqliteTaskQueue(tmp_path / "queue.sqlite3")
    ignored = queue.enqueue(TaskJob.create(JobType.SCORING, "unrelated", {}))
    target = queue.enqueue(TaskJob.create(JobType.SCORING, "target", {}, max_retries=0))
    counter: list[int] = []
    lock = Lock()

    def handler(job):
        with lock:
            counter.append(1)
        return {"ok": True}

    def process(_):
        return JobWorker(queue, handlers={JobType.SCORING: handler}).process_one(target.job_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(process, range(2)))
    completed, pending = queue.get_job(target.job_id), queue.get_job(ignored.job_id)
    assert completed is not None and pending is not None
    assert len(counter) == 1 and completed.status == JobStatus.SUCCESS
    assert pending.status == JobStatus.PENDING


def test_interrupted_calls_are_fenced_and_never_auto_repeated(setup) -> None:
    adapter, calls = setup
    calls.side_effect = SystemExit("synthetic crash")
    run = adapter.prepare(ui.synthetic_frame().iloc[:1], "crash.csv", "crash")
    with pytest.raises(SystemExit):
        ui.execute(adapter, run)
    queue = SqliteTaskQueue(adapter.store.path)
    job = queue.list_jobs()[0]
    assert job.status == JobStatus.RUNNING
    assert queue.fail_interrupted(job.job_id)
    assert not queue.complete_claim(job.with_status(JobStatus.SUCCESS), job.updated_at)
    adapter.store.finish(run.run_id, interrupted=True)
    with pytest.raises(ValueError):
        adapter.process(run.run_id)
    assert calls.call_count == 1 and adapter.store.get(run.run_id).status == "FAILED"


def test_campaign_snapshot_cannot_be_replaced(tmp_path) -> None:
    repo = SqliteCampaignRepository(tmp_path / "campaigns.sqlite3")
    original = create_campaign(
        name="synthetic",
        dataset_id="sha256-test",
        model_version="llm-config/mock",
        scoring_run_id="uuid-test",
        budget=20.0,
        assumptions={"execution_mode": "mock"},
        results={"allocations": []},
        repository=repo,
    )
    with pytest.raises(ValueError, match="immutable"):
        repo.save(replace(original, scoring_run_id="different-run"))
    repo.save(original.with_status(CampaignStatus.EXPORTED))
    restored = repo.get(original.campaign_id)
    assert restored is not None and restored.scoring_run_id == "uuid-test"


def test_dashboard_preserves_tabs_persistence_campaigns_and_no_inference_on_rerun(
    setup, monkeypatch
) -> None:
    adapter, calls = setup
    run = adapter.prepare(ui.synthetic_frame(), "synthetic.csv", "seed-ui")
    ui.execute(adapter, run)
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    guard = Mock(side_effect=AssertionError("No inference on render/rerun"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception and not at.error
    plan_setup(at)
    for key in ["c1_c", "c2_c", "c3_c"]:
        at.number_input(key=key).set_value(1000000.0)
    apply_settings(at)
    assert not at.exception and not at.error  # All-A0 is a valid business scenario.
    save_campaign(at)
    assert not at.exception and not at.error
    camp = ui.list_campaigns()[0]
    assert camp.scoring_run_id == run.run_id
    assert (
        camp.assumptions["execution_mode"] == "mock"
        and camp.assumptions["probability_calibrated"] is False
    )
    assert camp.status == CampaignStatus.SIMULATED  # Rendering CSV must be read-only.
    at.run()
    assert not at.exception
    assert ui.get_campaign(camp.campaign_id).status == CampaignStatus.SIMULATED
    guard.assert_not_called()
    assert calls.call_count == 4


def test_dashboard_submit_single_and_repeated_click_is_idempotent(setup, monkeypatch) -> None:
    adapter, _ = setup
    monkeypatch.setattr(ui, "resources", lambda: adapter.resources)
    calls = Mock(wraps=mock_prediction)
    import churn_retention.entrypoints.streamlit.dashboard_llm_service as service

    monkeypatch.setattr(service, "mock_prediction", calls)
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception and not calls.called
    sample_analysis(at)
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.button(key="cuong_llm.submit").click().run()
    assert not at.exception and not at.error and calls.call_count == 1
    at.button(key="cuong_llm.submit").click().run()
    plan_setup(at)
    at.number_input(key="e2e_budget_input").set_value(1000.0)
    apply_settings(at)
    assert not at.exception and calls.call_count == 1


def test_uploaded_batch_uses_uploaded_bytes_and_only_selected_rows(setup, monkeypatch) -> None:
    adapter, _ = setup
    monkeypatch.setattr(ui, "resources", lambda: adapter.resources)
    frame = ui.synthetic_frame().iloc[:3].copy()
    frame["customerID"] = ["UPLOAD-001", "UPLOAD-002", "UPLOAD-003"]
    upload = BytesIO(frame.to_csv(index=False).encode())
    upload.name = "uploaded.csv"
    monkeypatch.setattr(ui.st, "file_uploader", Mock(return_value=upload))
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    go(at, "customers")
    at.button(key="data.accept").click().run()
    go(at, "analysis")
    at.radio(key="cuong_llm.scope").set_value("Batch nhỏ (tối đa 5)").run()
    at.multiselect(key="cuong_llm.selected_ids").set_value(["UPLOAD-002", "UPLOAD-001"]).run()
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.button(key="cuong_llm.submit").click().run()
    assert not at.exception and not at.error
    run_id = at.session_state[ui.ACTIVE_RUN_KEY]
    customers, _ = adapter.store.frames(run_id)
    assert customers.customerID.tolist() == ["UPLOAD-002", "UPLOAD-001"]
    assert adapter.store.get(run_id).source_name == "uploaded.csv"


def test_bad_upload_blocks_dashboard_instead_of_using_default_dataset(setup, monkeypatch) -> None:
    adapter, calls = setup
    monkeypatch.setattr(ui, "resources", lambda: adapter.resources)
    upload = BytesIO(b"bad,columns\n1,2\n")
    upload.name = "invalid.csv"
    monkeypatch.setattr(ui.st, "file_uploader", Mock(return_value=upload))
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    go(at, "customers")
    assert not at.exception and at.error
    go(at, "analysis")
    assert not at.button
    calls.assert_not_called()


def test_primary_dashboard_in_clean_minimal_deployment(source, tmp_path) -> None:
    from test_llm_standalone_app import RUNTIME_FILES

    root = tmp_path / "deployment"
    relatives = set(RUNTIME_FILES) | {
        "dashboard/app.py",
        "adapters/database/dashboard_runs.py",
        "adapters/database/local_sqlite.py",
        "entrypoints/streamlit/dashboard_llm_service.py",
        "entrypoints/streamlit/dashboard_llm_ui.py",
        "entrypoints/streamlit/demo_security.py",
        "adapters/database/campaigns/__init__.py",
        "adapters/database/campaigns/repository.py",
        "adapters/task_queue/__init__.py",
        "adapters/task_queue/queue.py",
        "contracts/jobs/__init__.py",
        "contracts/jobs/models.py",
        "entrypoints/worker/__init__.py",
        "entrypoints/worker/worker.py",
        "modules/campaigns/__init__.py",
        "modules/campaigns/models.py",
        "modules/campaigns/use_cases.py",
    }
    relatives |= {
        str(p.relative_to(ROOT / "src/churn_retention"))
        for p in (ROOT / "src/churn_retention/dashboard").rglob("*.py")
    }
    files = {Path("src/churn_retention") / rel for rel in relatives}
    for file in tuple(files):
        for parent in file.parents:
            if parent == Path("src"):
                break
            init = parent / "__init__.py"
            if (ROOT / init).is_file():
                files.add(init)
    for file in files:
        target = root / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / file, target)
    pin = hashlib.sha256(source.output.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tests/dashboard_llm_probe.py"),
            str(root),
            str(source.output),
            pin,
        ],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert (
        report["primary_dashboard_pages"] == 5
        and report["legacy_imports"] == report["real_api_calls"] == 0
    )


def test_dashboard_uses_existing_prediction_service_with_mock_router(setup, monkeypatch) -> None:
    from contextlib import asynccontextmanager

    from churn_retention.entrypoints.streamlit import llm_prediction_service
    from churn_retention.entrypoints.streamlit.dashboard_llm_service import MockProvider
    from churn_retention.modules.scoring.llm_router import LLMRouter

    adapter, _ = setup
    monkeypatch.setenv("CHURN_DASHBOARD_PREVIEW", "0")
    monkeypatch.setenv("CHURN_DASHBOARD_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "fake-offline-only")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-offline-only")

    @asynccontextmanager
    async def router():
        async with LLMRouter(
            MockProvider(),
            MockProvider(),
            primary_provider="Groq",
            fallback_provider="Gemini",
        ) as instance:
            yield instance

    monkeypatch.setattr(llm_prediction_service, "groq_gemini_router", router)
    original_service = Mock(wraps=llm_prediction_service.predict_customer)
    live = DashboardLLMAdapter(
        adapter.resources,
        adapter.pin,
        adapter.store,
        mode="live",
        predict_one=original_service,
    )
    run = live.prepare(
        ui.synthetic_frame().iloc[:1],
        "synthetic-service-contract.csv",
        "service-contract",
    )
    job = ui.execute(live, run)
    assert job is not None and job.status == JobStatus.SUCCESS
    original_service.assert_called_once()
    customers, predictions = adapter.store.frames(run.run_id)
    assert len(customers) == len(predictions) == 1
    assert predictions.iloc[0]["reference_count"] == 5
    assert predictions.iloc[0]["reason"].startswith("MOCK provider")


def test_dashboard_page_reload_auto_restores_latest_run_and_allows_reset(
    setup, monkeypatch
) -> None:
    adapter, _ = setup
    monkeypatch.setattr(ui, "resources", lambda: adapter.resources)
    run = adapter.prepare(ui.synthetic_frame().iloc[:2], "auto-saved.csv", "auto-reload-req")
    ui.execute(adapter, run)

    # Simulate fresh page load / F5 reload without CHURN_DASHBOARD_DEMO_RUN
    at = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not at.exception and not at.error
    assert at.session_state.get(ui.ACTIVE_RUN_KEY) == run.run_id
    assert at.session_state.get("retention.input") is not None
    assert len(at.session_state.get("retention.input").frame) == 2

    # Click reset workspace in sidebar
    at.button(key="reset_workspace_btn").click().run()
    assert not at.exception
    assert at.session_state.get("retention.session_cleared") is True
    assert at.session_state.get(ui.ACTIVE_RUN_KEY) is None
    assert at.session_state.get("retention.input") is None

