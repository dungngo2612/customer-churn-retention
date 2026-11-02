"""Group 4: real business engines/SQLite/UI with synthetic inputs and no API."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import StringIO
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_dashboard_llm_integration import PAGE
from test_dashboard_llm_integration import setup as setup  # noqa: PLC0414 -- pytest fixture export
from test_dashboard_llm_integration import source as source  # noqa: PLC0414 -- fixture dependency

from churn_retention.adapters.database.campaigns.repository import (
    SqliteCampaignRepository,
)
from churn_retention.adapters.database.dashboard_runs import DashboardRunStore
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as ui
from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
    DashboardLLMAdapter,
)
from churn_retention.entrypoints.worker.worker import (
    JobWorker,
    PermanentError,
    TransientError,
)
from churn_retention.modules.campaigns.models import CampaignStatus
from churn_retention.modules.campaigns.use_cases import (
    create_campaign,
    export_campaign_csv,
)
from churn_retention.modules.retention.budget import (
    select_actions_under_budget,
    solve_optimal_budget_brute_force,
)
from churn_retention.modules.retention.policy import RetentionPolicy, compute_proxy_clv
from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    ValidationError,
    default_actions,
    evaluate_actions,
    recommend,
)


def frames():
    customers = ui.synthetic_frame()
    predictions = pd.DataFrame(
        {"customer_id": customers.customerID, "churn_prob": [0.9, 0.7, 0.4, 0.8]}
    )
    return customers, predictions


def kwargs_for(adapter, run, budget=20.0):
    customers, predictions = adapter.store.frames(run.run_id)
    allocations = select_actions_under_budget(customers, predictions, budget)
    selected = allocations[allocations.action_code != "A0"]
    return {
        "name": "Synthetic campaign",
        "dataset_id": hashlib.sha256(customers.to_json(orient="records").encode()).hexdigest(),
        "model_version": run.predictor_ref,
        "scoring_run_id": run.run_id,
        "budget": budget,
        "assumptions": {
            "margin": 0.5,
            "horizon_months": 12,
            "actions": [asdict(a) for a in default_actions()],
            "prediction_source": "llm-rag-v2",
            "execution_mode": run.execution_mode,
            "probability_calibrated": False,
            "reference_version": run.reference_version,
            "reference_sha256": run.reference_sha256,
            "source_run_dataset_digest": run.dataset_digest,
        },
        "results": {
            "allocations": json.loads(allocations.to_json(orient="records")),
            "summary": {
                "total_cost": float(selected.cost.sum()),
                "total_benefit": float(selected.benefit.sum()),
                "total_net_value": float(selected.net_value.sum()),
                "n_intervened": len(selected),
                "n_total": len(allocations),
            },
        },
        "status": CampaignStatus.SIMULATED,
    }


def completed(setup):
    adapter, calls = setup
    run = adapter.prepare(ui.synthetic_frame(), "DỮ LIỆU GIẢ", "group4")
    ui.execute(adapter, run)
    return adapter, adapter.store.get(run.run_id), calls


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1, True])
@pytest.mark.parametrize("solver", [select_actions_under_budget, solve_optimal_budget_brute_force])
def test_invalid_budget_rejected(bad, solver):
    with pytest.raises(ValidationError):
        solver(*frames(), budget=bad)


@pytest.mark.parametrize("field", ["contact_cost", "offer_cost", "acceptance_rate", "efficacy"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_action_config_rejects_nonfinite(field, bad):
    values = {
        "code": "X",
        "name": "synthetic",
        "contact_cost": 1.0,
        "offer_cost": 1.0,
        "acceptance_rate": 0.5,
        "efficacy": 0.5,
    }
    values[field] = bad
    with pytest.raises(ValidationError):
        ActionConfig(**values)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("tenure", float("inf")),
        ("tenure", -1),
        ("tenure", 1.5),
        ("tenure", True),
        ("MonthlyCharges", float("inf")),
        ("MonthlyCharges", True),
        ("churn_prob", True),
    ],
)
def test_economic_inputs_reject_bad_values(field, bad):
    customers, predictions = frames()
    frame = predictions if field == "churn_prob" else customers
    frame[field] = frame[field].astype(object)
    frame.loc[0, field] = bad
    with pytest.raises(ValidationError):
        evaluate_actions(customers, predictions)


def test_policy_equations_and_zero_denominator():
    policy = RetentionPolicy(480, 0.7, 8, 0.1)
    assert policy.expected_benefit() == pytest.approx(33.6)
    assert policy.expected_net_value() == pytest.approx(25.6)
    assert policy.breakeven_prob() == pytest.approx(0.1667)
    assert not RetentionPolicy(100, 0.5, 10, 0.2).should_intervene()  # exact break-even
    assert RetentionPolicy(0, 1, 0, 1).breakeven_prob() is None
    for bad in (float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError):
            RetentionPolicy(bad, 0.5, 1, 0.1)
        with pytest.raises(ValueError):
            compute_proxy_clv(bad)
    with pytest.raises(ValueError):
        compute_proxy_clv(1e308, 120)


def test_manual_equations_alignment_and_observed_status():
    customers, predictions = frames()
    options = evaluate_actions(customers, predictions.iloc[::-1])
    active = customers[(customers.Churn == "No") & (customers.tenure > 0)]
    assert set(options.customerID) == set(active.customerID)
    for row in options.itertuples():
        customer = customers.set_index("customerID").loc[row.customerID]
        probability = predictions.set_index("customer_id").loc[row.customerID, "churn_prob"]
        action = next(a for a in default_actions() if a.code == row.action_code)
        value = customer.MonthlyCharges * 12 * 0.5
        cost = action.contact_cost + action.acceptance_rate * action.offer_cost
        reduction = action.acceptance_rate * action.efficacy
        assert row.proxy_value == pytest.approx(value)
        assert row.cost == pytest.approx(cost)
        assert row.benefit == pytest.approx(probability * reduction * value)
        assert row.net_value == pytest.approx(row.benefit - cost)
        assert row.churn_prob_after == pytest.approx(probability * (1 - reduction))
    assert (recommend(customers, predictions).net_value >= 0).all()


@pytest.mark.parametrize("budget", [0.0, 0.0000000001, 7.99, 8.0, 9.0, 19.0, 20.0, 100.0])
def test_allocation_invariants_and_deterministic_order(budget):
    customers, predictions = frames()
    first = select_actions_under_budget(customers, predictions, budget)
    second = select_actions_under_budget(customers.iloc[::-1], predictions.iloc[::-1], budget)
    pd.testing.assert_frame_equal(first, second)
    assert first.customerID.is_unique and first.cost.sum() <= budget
    assert (first.net_value >= 0).all()
    if budget == 0:
        assert first.action_code.eq("A0").all()
    _, optimal_value = solve_optimal_budget_brute_force(customers, predictions, budget)
    assert first.net_value.sum() <= optimal_value + 0.0001


def test_zero_cost_positive_action_and_no_epsilon_overspend():
    customers, predictions = frames()
    actions = (default_actions()[0], ActionConfig("FREE", "synthetic free", 0, 0, 1, 1))
    result = select_actions_under_budget(customers, predictions, 0, actions)
    assert result.action_code.eq("FREE").all() and result.cost.sum() == 0
    tiny = (default_actions()[0], ActionConfig("TINY", "tiny", 5e-10, 0, 1, 1))
    assert select_actions_under_budget(customers, predictions, 0, tiny).action_code.eq("A0").all()


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1, True])
def test_campaign_rejects_invalid_budget(tmp_path, budget):
    repo = SqliteCampaignRepository(tmp_path / "campaign.sqlite3")
    with pytest.raises(ValueError):
        create_campaign("synthetic", "dataset", "model", "run", budget, {}, repository=repo)
    assert repo.list_all() == []


def test_campaign_end_to_end_duplicates_restart_and_export(setup):
    adapter, run, calls = completed(setup)
    kwargs = kwargs_for(adapter, run)
    with ThreadPoolExecutor(max_workers=3) as pool:
        campaigns = list(pool.map(lambda _: ui.create_campaign(**kwargs), range(3)))
    assert len({c.campaign_id for c in campaigns}) == 1 and len(ui.list_campaigns()) == 1
    campaign = campaigns[0]
    original = ui.get_campaign(campaign.campaign_id).to_dict()
    ui.export_campaign_csv(campaign.campaign_id)
    assert ui.get_campaign(campaign.campaign_id).to_dict() == original  # render is read-only
    ui.mark_exported(campaign.campaign_id)
    assert ui.create_campaign(**kwargs).status == CampaignStatus.EXPORTED  # never regress status
    reopened = SqliteCampaignRepository(ui._repo().db_path)
    saved = reopened.get(campaign.campaign_id)
    assert saved is not None and saved.scoring_run_id == run.run_id
    csv = pd.read_csv(StringIO(ui.export_campaign_csv(campaign.campaign_id)))
    assert set(csv.customer_id) <= set(adapter.store.frames(run.run_id)[0].customerID)
    assert not csv.action_code.eq("A0").any() and calls.call_count == 4
    assert (
        ui.create_campaign(
            **{
                **kwargs,
                "budget": 0.0,
                "results": kwargs_for(adapter, run, 0)["results"],
            }
        ).campaign_id
        != campaign.campaign_id
    )


@pytest.mark.parametrize(
    "problem",
    [
        "dataset",
        "run",
        "model",
        "pin",
        "mode",
        "summary",
        "customer",
        "probability",
        "duplicate",
    ],
)
def test_campaign_rejects_inconsistent_links_and_results(setup, problem):
    adapter, run, _ = completed(setup)
    values = kwargs_for(adapter, run)
    if problem in {"dataset", "run", "model"}:
        values[
            {
                "dataset": "dataset_id",
                "run": "scoring_run_id",
                "model": "model_version",
            }[problem]
        ] = "wrong"
    elif problem in {"pin", "mode"}:
        values["assumptions"][{"pin": "reference_sha256", "mode": "execution_mode"}[problem]] = (
            "wrong"
        )
    elif problem == "summary":
        values["results"]["summary"]["total_net_value"] += 1
    elif problem == "customer":
        values["results"]["allocations"][0]["customerID"] = "different"
    elif problem == "probability":
        values["results"]["allocations"][0]["churn_prob"] = 0.01
    else:
        values["results"]["allocations"].append(values["results"]["allocations"][0])
    with pytest.raises(ValueError):
        ui.create_campaign(**values)
    assert ui.list_campaigns() == []


def test_campaign_rollback_on_database_failure(setup):
    adapter, run, _ = completed(setup)
    repo = ui._repo()
    with sqlite3.connect(repo.db_path) as db:
        db.execute(
            (
                "CREATE TRIGGER fail_insert BEFORE INSERT ON campaigns BEGIN "
                "SELECT RAISE(ABORT, 'synthetic'); END;"
            )
        )
    with pytest.raises(sqlite3.IntegrityError):
        ui.create_campaign(**kwargs_for(adapter, run))
    assert repo.list_all() == []
    with sqlite3.connect(repo.db_path) as db:
        db.execute("DROP TRIGGER fail_insert")
    assert ui.create_campaign(**kwargs_for(adapter, run)).scoring_run_id == run.run_id


@pytest.mark.parametrize("value", ["=1+1", " +cmd", "\t@SUM(1)", "\r=1", "\n=1"])
def test_csv_formulas_escaped_without_changing_identity(tmp_path, value):
    repo = SqliteCampaignRepository(tmp_path / "campaign.sqlite3")
    campaign = create_campaign(
        "synthetic",
        "dataset",
        "model",
        "run",
        0,
        {},
        {
            "allocations": [
                {
                    "customer_id": value,
                    "action_code": "A1",
                    "action_name": value,
                    "cost": 0,
                    "benefit": 1,
                    "net_value": 1,
                }
            ]
        },
        repository=repo,
    )
    frame = pd.read_csv(StringIO(export_campaign_csv(campaign.campaign_id, repository=repo)))
    assert frame.customer_id[0] == "'" + value and frame.action_name[0] == "'" + value
    assert repo.get(campaign.campaign_id).results["allocations"][0]["customer_id"] == value


def test_queue_retry_budget_error_cleanup_and_collision(tmp_path):
    queue = SqliteTaskQueue(tmp_path / "queue.sqlite3")
    job = queue.enqueue(
        TaskJob.create(JobType.CAMPAIGN_SIMULATION, "intent", {"id": 1}, max_retries=2)
    )
    with pytest.raises(ValueError):
        queue.enqueue(
            TaskJob.create(JobType.CAMPAIGN_SIMULATION, "intent", {"id": 2}, max_retries=2)
        )
    calls = Mock(
        side_effect=[
            TransientError("synthetic"),
            TransientError("synthetic"),
            {"ok": True},
        ]
    )
    worker = JobWorker(queue, handlers={JobType.CAMPAIGN_SIMULATION: calls})
    assert worker.process_one().status == JobStatus.RETRYING
    assert worker.process_one().status == JobStatus.RETRYING
    result = worker.process_one()
    assert result.status == JobStatus.SUCCESS and result.retry_count == 2
    assert result.error_type is None and result.error_reason is None
    assert worker.process_one() is None and calls.call_count == 3
    with pytest.raises(ValueError):
        queue.update_job(job)
    assert queue.get_job(job.job_id).status == JobStatus.SUCCESS


def test_worker_permanent_and_exhausted(tmp_path):
    queue = SqliteTaskQueue(tmp_path / "queue.sqlite3")
    for error, expected_calls in (
        (PermanentError("synthetic"), 1),
        (TransientError("synthetic"), 2),
    ):
        job = queue.enqueue(
            TaskJob.create(JobType.CAMPAIGN_SIMULATION, str(expected_calls), {}, max_retries=1)
        )
        calls = Mock(side_effect=error)
        worker = JobWorker(queue, handlers={JobType.CAMPAIGN_SIMULATION: calls})
        for _ in range(expected_calls):
            result = worker.process_one(job.job_id)
        assert result.status == JobStatus.FAILED
        assert worker.process_one(job.job_id) is None and calls.call_count == expected_calls


def test_atomic_recovery_rollback_fencing_and_partial_results(setup):
    adapter, _ = setup
    run = adapter.prepare(ui.synthetic_frame(), "synthetic", "crash")
    queue = SqliteTaskQueue(adapter.store.path)
    job = queue.enqueue(
        TaskJob.create(JobType.SCORING, "crash", {"run_id": run.run_id}, max_retries=0)
    )
    claim = queue.claim_next(job.job_id)
    assert adapter.store.start(run.run_id)
    adapter.store.begin_attempt(run.run_id, ui.synthetic_frame().customerID[0])
    from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
        mock_prediction,
    )

    adapter.store.outcome(
        run.run_id,
        ui.synthetic_frame().customerID[0],
        mock_prediction(adapter.resources.store, ui.synthetic_frame().iloc[0].to_dict()),
    )
    adapter.store.begin_attempt(run.run_id, ui.synthetic_frame().customerID[1])
    with adapter.store.connect() as db:
        db.execute(
            (
                "CREATE TRIGGER recovery_abort BEFORE UPDATE ON task_jobs BEG"
                "IN SELECT RAISE(ABORT, 'synthetic'); END;"
            )
        )
    with pytest.raises(sqlite3.IntegrityError):
        adapter.store.recover_interrupted_job(job.job_id)
    assert adapter.store.get(run.run_id).status == "RUNNING"
    assert "CALLING" in set(adapter.store.progress(run.run_id).state)
    with adapter.store.connect() as db:
        db.execute("DROP TRIGGER recovery_abort")
    assert adapter.store.recover_interrupted_job(job.job_id)
    assert not queue.complete_claim(claim.with_status(JobStatus.SUCCESS), claim.updated_at)
    assert not adapter.store.recover_interrupted_job(job.job_id)
    assert queue.get_job(job.job_id).status == JobStatus.FAILED
    assert adapter.store.get(run.run_id).status == "PARTIAL"
    assert adapter.store.progress(run.run_id).state.tolist() == [
        "SUCCESS",
        "UNKNOWN",
        "NOT_RUN",
        "NOT_RUN",
    ]
    assert len(DashboardRunStore(adapter.store.path).frames(run.run_id)[0]) == 1
    with pytest.raises(ValueError):
        adapter.store.begin_attempt(run.run_id, ui.synthetic_frame().customerID[2])


def test_recover_job_after_run_committed_never_predicts_again(setup):
    adapter, run, calls = completed(setup)
    queue = SqliteTaskQueue(adapter.store.path)
    job = queue.enqueue(
        TaskJob.create(JobType.SCORING, "after-commit", {"run_id": run.run_id}, max_retries=0)
    )
    queue.claim_next(job.job_id)
    assert adapter.store.recover_interrupted_job(job.job_id)
    assert queue.get_job(job.job_id).status == JobStatus.SUCCESS
    assert adapter.store.get(run.run_id).status == "COMPLETE" and calls.call_count == 4


def test_all_tabs_zero_budget_campaign_repeat_and_no_prediction_rerun(setup, monkeypatch):
    _adapter, run, calls = completed(setup)
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    guard = Mock(side_effect=AssertionError("No inference on business operations"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    assert len(at.tabs) == 4 and not at.exception and not at.error
    assert at.button(key="btn_apply").disabled
    assert any("Chưa có dịch vụ thực thi" in info.value for info in at.info)
    at.slider(key="e2e_budget_slider").set_value(0).run()
    assert not at.exception and not at.error
    at.button(key="btn_save_e2e_camp").click().run()
    at.button(key="btn_save_e2e_camp").click().run()
    assert len(ui.list_campaigns()) == 1 and not at.exception and not at.error
    campaign = ui.list_campaigns()[0]
    assert campaign.budget == 0 and campaign.results["summary"]["total_cost"] == 0
    ui.mark_exported(campaign.campaign_id)
    at.run()
    assert not at.exception and not at.error
    assert ui.get_campaign(campaign.campaign_id).status == CampaignStatus.EXPORTED
    assert calls.call_count == 4
    guard.assert_not_called()


@pytest.mark.parametrize("scenario", ["inactive", "new-customers", "zero-value", "free-actions"])
def test_dashboard_business_edge_scenarios(setup, monkeypatch, scenario):
    adapter, calls = setup
    frame = ui.synthetic_frame()
    if scenario == "inactive":
        frame["Churn"] = "Yes"
    elif scenario == "new-customers":
        frame["tenure"] = 0
    elif scenario == "zero-value":
        frame["MonthlyCharges"] = 0.0
    run = adapter.prepare(frame, "DỮ LIỆU GIẢ", scenario)
    ui.execute(adapter, run)
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    guard = Mock(side_effect=AssertionError("No inference during business views"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    if scenario == "free-actions":
        for key in ["c1_c", "c1_o", "c2_c", "c2_o", "c3_c", "c3_o"]:
            at.number_input(key=key).set_value(0.0)
        at.run()
        assert "N/A (chi phí bằng 0)" in "\n".join(m.value for m in at.markdown)
    assert len(at.tabs) == 4 and not at.exception and not at.error
    at.button(key="btn_save_e2e_camp").click().run()
    assert not at.exception and not at.error
    assert len(ui.list_campaigns()) == 1
    assert ui.list_campaigns()[0].results["summary"]["total_cost"] == 0
    assert calls.call_count == 4
    guard.assert_not_called()


def test_dashboard_save_failure_is_safe_and_preserves_previous_campaign(setup, monkeypatch):
    adapter, run, _calls = completed(setup)
    existing = ui.create_campaign(**kwargs_for(adapter, run))
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    monkeypatch.setattr(
        ui,
        "create_campaign",
        Mock(side_effect=sqlite3.OperationalError("PRIVATE_DB_PATH")),
    )
    at.button(key="btn_save_e2e_camp").click().run()
    assert not at.exception and at.error
    assert "PRIVATE_DB_PATH" not in "\n".join(error.value for error in at.error)
    assert [c.campaign_id for c in ui.list_campaigns()] == [existing.campaign_id]


def test_old_running_job_visible_for_recovery(tmp_path):
    queue = SqliteTaskQueue(tmp_path / "queue.sqlite3")
    first = queue.enqueue(TaskJob.create(JobType.CAMPAIGN_SIMULATION, "older-running", {}))
    queue.claim_next(first.job_id)
    for index in range(6):
        queue.enqueue(TaskJob.create(JobType.CAMPAIGN_SIMULATION, f"new-{index}", {}))
    assert first.job_id not in {job.job_id for job in queue.list_jobs(5)}
    assert [job.job_id for job in queue.list_running_jobs()] == [first.job_id]
