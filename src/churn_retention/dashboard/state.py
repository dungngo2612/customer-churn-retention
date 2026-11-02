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
    initialize()
    ledger = DashboardRunStore(services.runtime_path())
    form, pin, error = None, "", ""
    try:
        form = services.resources()
        pin = services.reference_artifact_sha256()
    except Exception:  # noqa: BLE001 -- Do not display local paths or raw configuration.
        error = "Dữ liệu tham chiếu chưa sẵn sàng hoặc không hợp lệ. Chưa thể chạy phân tích."
    view = None
    run_id = st.session_state.get(services.ACTIVE_RUN_KEY)
    if run_id:
        try:
            run = ledger.get(run_id)
            if run.execution_mode not in {"mock", "live"} or run.reference_sha256 != pin:
                raise ValueError("Reference mismatch")
            customers, predictions = ledger.frames(run_id)
            if not customers.empty:
                view = services.DashboardView(run, customers, predictions)
        except Exception:  # noqa: BLE001 -- Keep saved state on corrupt/stale context.
            error = (
                "Không thể mở kết quả đã lưu với dữ liệu tham chiếu hiện tại. Dữ liệu vẫn được giữ."
            )
    return Context(ledger, form, pin, st.session_state.get(INPUT), view, error)


def set_input(data: InputData) -> None:
    st.session_state[INPUT] = data
    st.session_state.pop(INPUT_ERROR, None)
    # Never clear an accepted run or infer when the input changes.
    st.session_state["cuong_llm.confirm"] = False


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


def plan_step(step: str) -> None:
    st.session_state["retention.plan_step"] = step
    st.session_state["_plan_step"] = step
