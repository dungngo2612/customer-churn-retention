"""Prediction controls injected into Cường's Dashboard, not a replacement UI."""

import hashlib
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from churn_retention.adapters.database.campaigns.repository import (
    SqliteCampaignRepository,
)
from churn_retention.adapters.database.dashboard_runs import (
    DashboardRunStore,
    SavedRun,
    canonical,
)
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.contracts.jobs.models import JobType, TaskJob
from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
    DashboardLLMAdapter,
)
from churn_retention.entrypoints.streamlit.llm_input import (
    LLMInputResources,
    load_form_resources,
    reference_artifact_path,
    reference_artifact_sha256,
    reference_fingerprint,
)
from churn_retention.entrypoints.worker.worker import JobWorker
from churn_retention.modules.campaigns import (
    Campaign,
    CampaignStatus,
)
from churn_retention.modules.campaigns import (
    create_campaign as _create,
)
from churn_retention.modules.campaigns import (
    export_campaign_dataframe as _dataframe,
)
from churn_retention.modules.campaigns import (
    get_campaign as _get,
)
from churn_retention.modules.campaigns import (
    list_campaigns as _list,
)
from churn_retention.modules.campaigns.use_cases import safe_export_frame
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.retention.demo import demo_inputs
from churn_retention.modules.retention.use_cases import actions_from_dicts

ACTIVE_RUN_KEY = "cuong_llm.active_run"
REQUEST_KEY = "cuong_llm.request_id"
LAST_ATTEMPT_KEY = "cuong_llm.last_attempt"


def execution_label(mode: object) -> str:
    return {"mock": "MOCK", "live": "LIVE"}.get(str(mode), "UNKNOWN")


def _new_request() -> None:
    st.session_state[REQUEST_KEY] = str(uuid4())
    st.session_state["cuong_llm.confirm"] = False


def runtime_path() -> Path:
    # Separate namespace/path; never reuse data/predictions.db or an existing DB.
    return Path(
        os.environ.get(
            "CHURN_DASHBOARD_STATE",
            str(Path(__file__).resolve().parents[4] / ".dashboard-preview/state.sqlite3"),
        )
    )


@st.cache_resource
def _resources(path: str, fingerprint: object, pin: str) -> LLMInputResources:
    resources = load_form_resources(path)
    if reference_fingerprint(path) != fingerprint or reference_artifact_sha256() != pin:
        raise ValueError("Reference changed during load")
    return resources


def resources() -> LLMInputResources:
    path = reference_artifact_path()
    return _resources(str(path), reference_fingerprint(path), reference_artifact_sha256())


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": context.customer_ref,
                "Churn": "No" if context.observed_active else "Yes",
                **record.features(),
            }
            for record, context in demo_inputs()
        ]
    )


def execute(adapter: DashboardLLMAdapter, run: SavedRun) -> TaskJob | None:
    queue = SqliteTaskQueue(adapter.store.path)
    job = queue.enqueue(
        TaskJob.create(
            JobType.SCORING,
            f"dashboard-llm:{run.request_key}",
            {"run_id": run.run_id},
            max_retries=0,
        )
    )
    worker = JobWorker(
        queue,
        handlers={JobType.SCORING: lambda item: adapter.process(str(item.payload["run_id"]))},
        sanitize_errors=True,
    )
    return worker.process_one(job.job_id) or queue.get_job(job.job_id)


@dataclass(frozen=True)
class DashboardView:
    run: SavedRun
    customers: pd.DataFrame
    predictions: pd.DataFrame


def _repo() -> SqliteCampaignRepository:
    return SqliteCampaignRepository(runtime_path().with_name("campaigns-preview.sqlite3"))


def create_campaign(**kwargs) -> Campaign:
    ledger = DashboardRunStore(runtime_path())
    run = ledger.get(kwargs["scoring_run_id"])
    if run.status not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("Campaign requires accepted predictions")
    customers, predictions = ledger.frames(run.run_id)
    assumptions = kwargs["assumptions"]
    dataset_id = hashlib.sha256(customers.to_json(orient="records").encode()).hexdigest()
    required = {
        "prediction_source": "llm-rag-v2",
        "execution_mode": run.execution_mode,
        "probability_calibrated": False,
        "reference_version": run.reference_version,
        "reference_sha256": run.reference_sha256,
        "source_run_dataset_digest": run.dataset_digest,
    }
    if (
        not len(customers)
        or kwargs["dataset_id"] != dataset_id
        or kwargs["model_version"] != run.predictor_ref
        or any(assumptions.get(k) != v for k, v in required.items())
    ):
        raise ValueError("Campaign provenance does not match its accepted run")
    expected = select_actions_under_budget(
        customers,
        predictions,
        kwargs["budget"],
        actions_from_dicts(assumptions["actions"]),
        assumptions["margin"],
        assumptions["horizon_months"],
    )
    actual = pd.DataFrame(kwargs["results"]["allocations"])
    if expected.empty:
        if not actual.empty:
            raise ValueError("Campaign allocations do not match run")
    else:
        try:
            wanted = expected.sort_values("customerID").reset_index(drop=True)
            given = actual[expected.columns].sort_values("customerID").reset_index(drop=True)
            if len(wanted) != len(given):
                raise ValueError("Campaign allocation count differs")
            for left, right in zip(
                wanted.to_dict("records"), given.to_dict("records"), strict=True
            ):
                for column in expected.columns:
                    want, got = left[column], right[column]
                    if pd.isna(want) or (isinstance(want, Real) and not math.isfinite(want)):
                        valid = pd.isna(got)
                    elif isinstance(want, Real):
                        valid = (
                            isinstance(got, Real)
                            and not isinstance(got, bool)
                            and math.isfinite(got)
                            and math.isclose(want, got, rel_tol=1e-8, abs_tol=1e-8)
                        )
                    else:
                        valid = want == got
                    if not valid:
                        raise ValueError("Campaign allocations do not match run")
        except (KeyError, TypeError) as exc:
            raise ValueError("Campaign allocations do not match run") from exc
    selected = expected[expected["action_code"] != "A0"]
    summary = {
        "total_cost": float(selected["cost"].sum()),
        "total_benefit": float(selected["benefit"].sum()),
        "total_net_value": float(selected["net_value"].sum()),
        "n_intervened": len(selected),
        "n_total": len(expected),
    }
    if kwargs["results"]["summary"] != summary:
        raise ValueError("Campaign summary does not match allocations")
    # Exact repeated intent maps to one row, even across concurrent sessions/restarts.
    key = hashlib.sha256(canonical(kwargs).encode()).hexdigest()
    return _create(**kwargs, repository=_repo(), idempotency_key=key)


def get_campaign(campaign_id: str) -> Campaign:
    return _get(campaign_id, repository=_repo())


def list_campaigns() -> list[Campaign]:
    return _list(repository=_repo())


def export_campaign_dataframe(campaign_id: str) -> pd.DataFrame:
    return _dataframe(campaign_id, repository=_repo())


def export_campaign_csv(campaign_id: str, only_interventions: bool = True) -> str:
    """Read-only preview; status changes only on explicit download click."""
    frame = export_campaign_dataframe(campaign_id)
    if only_interventions and not frame.empty:
        frame = frame[frame["action_code"] != "A0"].copy()
    frame = safe_export_frame(frame)
    return frame.to_csv(index=False)


def mark_exported(campaign_id: str) -> None:
    try:
        _repo().save(get_campaign(campaign_id).with_status(CampaignStatus.EXPORTED))
    except Exception:  # noqa: BLE001 -- Callback errors must not expose raw archive data.
        st.error(
            (
                "Không thể ghi nhận xuất CSV. Trạng thái chiến dịch chưa được"
                " xác nhận; dữ liệu được giữ nguyên."
            )
        )
