"""Workspace acceptance: navigation, stable pagination and preserved run context."""

from unittest.mock import Mock

import pandas as pd
import pytest
from dashboard_navigation import apply_settings, go, plan_setup, sample_analysis, save_campaign
from streamlit.testing.v1 import AppTest
from test_dashboard_llm_integration import PAGE
from test_dashboard_llm_integration import setup as setup  # noqa: PLC0414 -- shared fixture
from test_dashboard_llm_integration import source as source  # noqa: PLC0414 -- shared fixture

from churn_retention.dashboard import state
from churn_retention.dashboard.components import page_slice, percent, risk_label
from churn_retention.dashboard.planning import Settings, campaign_arguments
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.entrypoints.streamlit.dashboard_llm_service import DashboardLLMAdapter
from churn_retention.modules.retention.use_cases import default_actions


@pytest.mark.parametrize("size", [10, 25, 50])
def test_pagination_preserves_every_id_once_in_stable_order(size):
    frame = pd.DataFrame({"customerID": [f"KH-{n:04d}" for n in range(101)]})
    ids = []
    _, _, pages = page_slice(frame, size, 1)
    for page in range(1, pages + 1):
        part, actual, total = page_slice(frame, size, page)
        assert actual == page and total == pages
        ids.extend(part.customerID)
    assert ids == frame.customerID.tolist() and len(set(ids)) == 101
    assert page_slice(frame.iloc[:1], size, 20)[1:] == (1, 1)
    assert page_slice(frame.iloc[:0], size, 10)[1:] == (1, 1)


def test_invalid_pagination_and_missing_values_are_distinct_from_zero():
    with pytest.raises(ValueError):
        page_slice(pd.DataFrame(), 0, 1)
    assert percent(0) == "0.0%" and percent(None) == "—"
    assert risk_label(0.7) == "Trung bình" and risk_label(0.7001) == "Cao"
    assert Settings().actions == default_actions()  # Original package conditions/defaults.


def test_empty_workspace_all_pages_never_runs_prediction(setup, monkeypatch):
    _, calls = setup
    guard = Mock(side_effect=AssertionError("No inference on navigation"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    for page in ["overview", "customers", "analysis", "plan", "campaigns", "overview"]:
        go(at, page)
        assert not at.error
    assert state.INPUT not in at.session_state
    assert services.ACTIVE_RUN_KEY not in at.session_state
    calls.assert_not_called()
    guard.assert_not_called()


def test_customer_search_resets_page_and_preserves_selected_id(setup):
    _, calls = setup
    frame = pd.concat([services.synthetic_frame()] * 16, ignore_index=True)
    frame["customerID"] = [f"KH-{n:04d}" for n in range(1, 65)]
    at = AppTest.from_file(str(PAGE), default_timeout=40)
    at.session_state[state.INPUT] = state.InputData(frame, "Hồ sơ tổng hợp", sample=True)
    at.run()
    go(at, "customers")
    at.number_input(key="customers.list.page").set_value(4).run()
    assert at.dataframe[0].value["Mã khách hàng"].tolist() == frame.customerID.tolist()[30:40]
    at.selectbox(key="customers.profile").select("KH-0031").run()
    assert at.session_state[state.CUSTOMER] == "KH-0031"
    at.text_input(key="customers.search").set_value("KH-0001").run()
    assert not at.error and not at.exception
    assert at.number_input(key="customers.list.page").value == 1
    assert at.dataframe[0].value["Mã khách hàng"].tolist() == ["KH-0001"]
    assert at.session_state[state.CUSTOMER] == "KH-0031"
    at.text_input(key="customers.search").set_value("KH-9999").run()
    assert not at.exception and not at.error
    assert not at.number_input
    go(at, "overview")
    go(at, "customers")
    assert at.text_input(key="customers.search").value == "KH-9999"
    assert at.session_state[state.INPUT].frame.customerID.tolist() == frame.customerID.tolist()
    calls.assert_not_called()


def test_new_input_keeps_saved_snapshot_settings_and_campaign_source(setup, monkeypatch):
    adapter, calls = setup
    run = adapter.prepare(services.synthetic_frame(), "Source A.csv", "saved-context")
    services.execute(adapter, run)
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    guard = Mock(side_effect=AssertionError("No repeated prediction"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    plan_setup(at)
    at.number_input(key="e2e_budget_input").set_value(10.0)
    apply_settings(at)
    changed = services.synthetic_frame()
    changed["MonthlyCharges"] += 5
    at.session_state[state.INPUT] = state.InputData(changed, "Source B.csv")
    for page in ["customers", "analysis", "overview", "campaigns", "plan"]:
        go(at, page)
        assert not at.error
        if page == "analysis":
            rows = next(item.value for item in at.dataframe if "Đề xuất" in item.value.columns)
            assert rows["Đề xuất"].notna().all()
            assert "Không thuộc diện can thiệp" in set(rows["Đề xuất"])
        assert at.session_state[services.ACTIVE_RUN_KEY] == run.run_id
    assert at.session_state[state.SETTINGS].budget == 10
    assert at.warning and "snapshot" in at.warning[0].value
    save_campaign(at)
    assert not at.error and not at.exception
    campaign = services.list_campaigns()[0]
    assert campaign.scoring_run_id == run.run_id
    assert campaign.assumptions["source_run_dataset_digest"] == run.dataset_digest
    assert campaign.budget == 10
    assert calls.call_count == 4
    guard.assert_not_called()


def test_batch_selection_survives_page_navigation_and_submit_stays_idempotent(setup, monkeypatch):
    adapter, _ = setup
    calls = Mock(wraps=adapter.predict_one)
    import churn_retention.entrypoints.streamlit.dashboard_llm_service as service

    monkeypatch.setattr(service, "mock_prediction", calls)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    sample_analysis(at)
    at.radio(key="cuong_llm.scope").set_value("Batch nhỏ (tối đa 5)").run()
    at.multiselect(key="cuong_llm.selected_ids").set_value(["DEMO-002", "DEMO-001"]).run()
    go(at, "overview")
    go(at, "analysis")
    assert at.multiselect(key="cuong_llm.selected_ids").value == ["DEMO-002", "DEMO-001"]
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.button(key="cuong_llm.submit").click().run()
    assert not at.error and not at.exception and calls.call_count == 2
    run_id = at.session_state[services.ACTIVE_RUN_KEY]
    for page in ["customers", "plan", "campaigns", "overview", "analysis"]:
        go(at, page)
        assert not at.error
    # Confirmation may need to be repeated; request identity and cohort stay unchanged.
    at.checkbox(key="cuong_llm.confirm").check().run()
    at.button(key="cuong_llm.submit").click().run()
    assert at.session_state[services.ACTIVE_RUN_KEY] == run_id and calls.call_count == 2
    assert adapter.store.frames(run_id)[0].customerID.tolist() == ["DEMO-002", "DEMO-001"]


def test_history_opens_exact_selected_run_without_prediction(setup, monkeypatch):
    adapter, calls = setup
    runs = []
    for i in range(2):
        run = adapter.prepare(
            services.synthetic_frame().iloc[[i]], f"Source {i}.csv", f"history-{i}"
        )
        services.execute(adapter, run)
        runs.append(run)
    guard = Mock(side_effect=AssertionError("History is read-only"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40).run()
    go(at, "analysis")
    at.selectbox(key="history.selected").select(runs[0].run_id).run()
    at.button(key="history.open").click().run()
    assert not at.error and not at.exception
    assert at.session_state[services.ACTIVE_RUN_KEY] == runs[0].run_id
    assert calls.call_count == 2
    guard.assert_not_called()


def test_campaign_selector_can_open_another_saved_snapshot(setup, monkeypatch):
    adapter, calls = setup
    run = adapter.prepare(services.synthetic_frame(), "Campaign source.csv", "two-campaigns")
    services.execute(adapter, run)
    customers, predictions = adapter.store.frames(run.run_id)
    view = services.DashboardView(adapter.store.get(run.run_id), customers, predictions)
    first = services.create_campaign(**campaign_arguments(view, Settings(), "Chiến dịch A"))
    second = services.create_campaign(**campaign_arguments(view, Settings(), "Chiến dịch B"))
    monkeypatch.setenv("CHURN_DASHBOARD_DEMO_RUN", run.run_id)
    guard = Mock(side_effect=AssertionError("Campaign selection must be read-only"))
    monkeypatch.setattr(DashboardLLMAdapter, "process", guard)
    at = AppTest.from_file(str(PAGE), default_timeout=40)
    at.session_state[state.CAMPAIGN] = second.campaign_id
    at.run()
    go(at, "campaigns")
    at.selectbox(key="campaign.selected").select(first.campaign_id).run()
    assert not at.exception and not at.error
    assert at.session_state[state.CAMPAIGN] == first.campaign_id
    assert at.selectbox(key="campaign.selected").value == first.campaign_id
    assert calls.call_count == 4
    guard.assert_not_called()
