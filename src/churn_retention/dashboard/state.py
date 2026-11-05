"""Session drafts and accepted snapshots, backed by the existing ledger and queue."""

import hashlib
import json
import os
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd
import streamlit as st

from churn_retention.adapters.database.dashboard_runs import DashboardRunStore, SavedRun, canonical
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.dashboard.planning import Settings
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.entrypoints.streamlit import demo_security
from churn_retention.entrypoints.streamlit.llm_input import LLMInputResources

INPUT = "retention.input"
INPUT_ERROR = "retention.input_error"
SETTINGS = "retention.settings"
CUSTOMER = "retention.customer"
CAMPAIGN = "retention.campaign"


@dataclass(frozen=True)
class InputData:
    frame: pd.DataFrame
    source: str
    sample: bool = False


@dataclass(frozen=True)
class Context:
    ledger: DashboardRunStore
    form: LLMInputResources | None
    pin: str
    data: InputData | None
    view: services.DashboardView | None
    error: str = ""


def initialize() -> None:
    if services.REQUEST_KEY not in st.session_state:
        st.session_state[services.REQUEST_KEY] = str(uuid4())
    if SETTINGS not in st.session_state:
        st.session_state[SETTINGS] = Settings()
    if services.ACTIVE_RUN_KEY not in st.session_state and os.environ.get(
        "CHURN_DASHBOARD_DEMO_RUN"
    ):
        st.session_state[services.ACTIVE_RUN_KEY] = os.environ["CHURN_DASHBOARD_DEMO_RUN"]


def context() -> Context:
    try:
        demo_security.require_safe_runtime()
    except ValueError:
        st.error("Ứng dụng chỉ hoạt động qua launcher cục bộ được hỗ trợ.")
        st.stop()
    initialize()
    ledger = DashboardRunStore(services.runtime_path())
    form, pin, error = None, "", ""
    try:
        form = services.resources()
        pin = services.reference_artifact_sha256()
    except Exception:  # noqa: BLE001 -- Do not display local paths or raw configuration.
        error = "Dữ liệu tham chiếu chưa sẵn sàng hoặc không hợp lệ. Chưa thể chạy phân tích."
    # Auto-restore input data from persistent workspace if session has none
    if st.session_state.get(INPUT) is None and not st.session_state.get("retention.session_cleared"):
        try:
            saved_input = ledger.load_active_input()
            if saved_input is not None:
                frame, source, is_sample = saved_input
                if not frame.empty:
                    st.session_state[INPUT] = InputData(frame=frame, source=source, sample=is_sample)
        except Exception:
            pass

    view = None
    run_id = st.session_state.get(services.ACTIVE_RUN_KEY)
    if not run_id and not st.session_state.get("retention.session_cleared") and pin:
        try:
            saved_run_id = ledger.load_active_run_id()
            if saved_run_id:
                try:
                    cand = ledger.get(saved_run_id)
                    if cand.execution_mode in {"mock", "live"} and cand.reference_sha256 == pin:
                        run_id = saved_run_id
                        st.session_state[services.ACTIVE_RUN_KEY] = run_id
                except Exception:
                    pass
            if not run_id:
                with ledger.connect() as conn:
                    row = conn.execute(
                        """
                        SELECT run_id FROM dashboard_runs
                        WHERE status IN ('COMPLETE', 'PARTIAL')
                          AND reference_sha256 = ?
                          AND execution_mode IN ('mock', 'live')
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (pin,),
                    ).fetchone()
                    if row is not None:
                        run_id = row["run_id"]
                        st.session_state[services.ACTIVE_RUN_KEY] = run_id
        except Exception:
            pass
    if run_id:
        try:
            run = ledger.get(run_id)
            if run.execution_mode not in {"mock", "live"} or run.reference_sha256 != pin:
                raise ValueError("Reference mismatch")
            customers, predictions = ledger.frames(run_id)
            if not customers.empty:
                view = services.DashboardView(run, customers, predictions)
                if st.session_state.get(INPUT) is None:
                    is_sample = (
                        run.source_name.startswith("Dữ liệu mẫu")
                        or "synthetic" in run.source_name.lower()
                    )
                    data = InputData(
                        frame=customers,
                        source=run.source_name,
                        sample=is_sample,
                    )
                    st.session_state[INPUT] = data
                    ledger.save_active_input(customers, run.source_name, is_sample)
        except Exception:  # noqa: BLE001 -- Keep saved state on corrupt/stale context.
            error = (
                "Không thể mở kết quả đã lưu với dữ liệu tham chiếu hiện tại. Dữ liệu vẫn được giữ."
            )
    return Context(ledger, form, pin, st.session_state.get(INPUT), view, error)


def set_input(data: InputData) -> None:
    st.session_state[INPUT] = data
    st.session_state.pop(INPUT_ERROR, None)
    st.session_state["retention.session_cleared"] = False
    # Never clear an accepted run or infer when the input changes.
    st.session_state["cuong_llm.confirm"] = False
    try:
        ledger = DashboardRunStore(services.runtime_path())
        ledger.save_active_input(data.frame, data.source, data.sample)
    except Exception:
        pass


def clear_session() -> None:
    st.session_state.pop(INPUT, None)
    st.session_state.pop(INPUT_ERROR, None)
    st.session_state.pop(services.ACTIVE_RUN_KEY, None)
    st.session_state.pop("retention.customer", None)
    st.session_state.pop("retention.campaign", None)
    st.session_state.pop("cuong_llm.confirm", None)
    st.session_state.pop("analysis.tabs", None)
    st.session_state["retention.session_cleared"] = True
    try:
        ledger = DashboardRunStore(services.runtime_path())
        ledger.clear_active_workspace()
    except Exception:
        pass


def matches_input(run: SavedRun, data: InputData) -> bool:
    ids = [row["customerID"] for row in json.loads(run.cohort_json)]
    try:
        selected = data.frame.set_index("customerID", drop=False).loc[ids]
        digest = hashlib.sha256(canonical(selected.to_dict(orient="records")).encode()).hexdigest()
    except KeyError:
        return False
    return digest == run.dataset_digest


def recent_runs(ctx: Context) -> list[SavedRun]:
    ids = dict.fromkeys(
        str(job.payload["run_id"])
        for job in SqliteTaskQueue(ctx.ledger.path).list_jobs(50)
        if "run_id" in job.payload
    )
    return [ctx.ledger.get(identifier) for identifier in ids]


def open_run(run_id: str) -> None:
    st.session_state[services.ACTIVE_RUN_KEY] = run_id
    st.session_state["retention.session_cleared"] = False
    try:
        ledger = DashboardRunStore(services.runtime_path())
        ledger.save_active_run(run_id)
    except Exception:
        pass


def plan_step(step: str) -> None:
    st.session_state["retention.plan_step"] = step
    st.session_state["_plan_step"] = step
