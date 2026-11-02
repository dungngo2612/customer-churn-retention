"""Prediction controls injected into Cường's Dashboard, not a replacement UI."""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal
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
from churn_retention.contracts.application_security import safe_source_name
from churn_retention.contracts.jobs.models import JobType, TaskJob
from churn_retention.contracts.llm_failure import failure_message
from churn_retention.entrypoints.streamlit.dashboard_llm_service import (
    DashboardLLMAdapter,
    read_csv,
    validate_frame,
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
            str(
                Path(__file__).resolve().parents[4] / ".dashboard-preview/state.sqlite3"
            ),
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
    return _resources(
        str(path), reference_fingerprint(path), reference_artifact_sha256()
    )


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
        handlers={
            JobType.SCORING: lambda item: adapter.process(str(item.payload["run_id"]))
        },
        sanitize_errors=True,
    )
    return worker.process_one(job.job_id) or queue.get_job(job.job_id)


@dataclass(frozen=True)
class DashboardView:
    run: SavedRun
    customers: pd.DataFrame
    predictions: pd.DataFrame


def prediction_controls() -> DashboardView | None:
    try:
        return _prediction_controls()
    except Exception:  # noqa: BLE001 -- Database and validation details must not reach UI/logs.
        st.error(
            "Không thể đọc dữ liệu Dashboard an toàn. Không gọi lại API hoặc tạo lại database tự động."
        )
        return None


def _prediction_controls() -> DashboardView | None:
    st.info(
        "Demo một quản trị viên: run ledger và archive campaign dùng chung giữa các phiên trình duyệt; phiên không phải tài khoản riêng."
    )
    st.warning(
        "Chỉ dự đoán khi Submit được xác nhận. Xác suất LLM chưa hiệu chuẩn; kinh tế/ROI chỉ là mô phỏng. Chế độ của kết quả đã lưu được ghi riêng."
    )
    try:
        form = resources()
        pin = reference_artifact_sha256()
    except Exception:  # noqa: BLE001 -- No artifact contents/paths in UI.
        st.error(
            "Reference Artifact v2 không hợp lệ hoặc không đáng tin cậy. Không fallback về model bundle."
        )
        return None
    ledger = DashboardRunStore(runtime_path())
    if REQUEST_KEY not in st.session_state:
        st.session_state[REQUEST_KEY] = str(uuid4())
    if ACTIVE_RUN_KEY not in st.session_state and os.environ.get(
        "CHURN_DASHBOARD_DEMO_RUN"
    ):
        st.session_state[ACTIVE_RUN_KEY] = os.environ["CHURN_DASHBOARD_DEMO_RUN"]
    with st.sidebar:
        st.subheader("Dự đoán hỗ trợ: LLM + RAG v2")
        st.caption(
            "Chấm đơn hoặc batch tối đa 5 ID; không tự chấm toàn CSV. Live chỉ xuất hiện khi được bật trên server."
        )
        uploaded = st.file_uploader(
            "CSV: 19 feature + customerID + Churn đã quan sát",
            type=["csv"],
            key="cuong_llm.csv",
        )
        st.download_button(
            "Tải CSV tổng hợp mẫu (DỮ LIỆU GIẢ)",
            synthetic_frame().to_csv(index=False),
            file_name="synthetic_customers.csv",
            mime="text/csv",
        )
        try:
            frame = (
                read_csv(uploaded.getvalue(), form)
                if uploaded is not None
                else validate_frame(synthetic_frame(), form)
            )
        except ValueError as error:
            st.error(str(error))
            return None
        source_name = (
            safe_source_name(uploaded.name)
            if uploaded is not None
            else "synthetic_customers.csv (DỮ LIỆU GIẢ)"
        )
        st.caption(
            f"Đã nạp {len(frame)} dòng; đây là dữ liệu đầu vào, chưa tự tạo dự đoán."
        )
        mode = st.radio(
            "Phạm vi gửi",
            ["Một khách hàng", "Batch nhỏ (tối đa 5)"],
            key="cuong_llm.scope",
        )
        if mode == "Một khách hàng":
            selected = [
                st.selectbox(
                    "ID cần dự đoán",
                    frame["customerID"].tolist(),
                    key="cuong_llm.single_id",
                )
            ]
        else:
            selected = st.multiselect(
                "Chọn rõ các ID sẽ gửi",
                frame["customerID"].tolist(),
                max_selections=5,
                key="cuong_llm.selected_ids",
            )
        allow_live = (
            os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") == "1"
            and os.environ.get("CHURN_DASHBOARD_PREVIEW") != "1"
        )
        provider_mode: Literal["mock", "live"] = "mock"
        if allow_live:
            provider_choice = st.radio(
                "Chế độ Provider",
                ["Mock provider (offline demo)", "Live API (Groq/Gemini)"],
                key="cuong_llm.provider_mode",
            )
            if provider_choice == "Live API (Groq/Gemini)":
                provider_mode = "live"
        if provider_mode == "mock":
            st.info("MOCK provider: đầu ra mô phỏng; không gửi API thật.")
        else:
            st.warning(
                "LIVE API: gửi dữ liệu đã chọn đến Groq/Gemini khi Submit; có thể phát sinh phí, gồm hedge/fallback."
            )
        selected_frame = (
            frame.set_index("customerID", drop=False)
            .loc[selected]
            .reset_index(drop=True)
        )
        confirmation_context = (
            provider_mode,
            pin,
            hashlib.sha256(
                canonical(selected_frame.to_dict(orient="records")).encode()
            ).hexdigest(),
        )
        if st.session_state.get("cuong_llm.confirm_context") != confirmation_context:
            st.session_state["cuong_llm.confirm"] = False
            st.session_state["cuong_llm.confirm_context"] = confirmation_context
        confirm_label = (
            "Xác nhận chỉ chấm các dòng đã chọn bằng Live API (Groq/Gemini)"
            if provider_mode == "live"
            else "Xác nhận chỉ chấm các dòng đã chọn bằng mock provider"
        )
        confirmed = st.checkbox(
            confirm_label,
            key="cuong_llm.confirm",
        )
        st.caption(
            "Live service hiện có: tối đa 120s mỗi dòng (gồm hedge/fallback); batch 5 dòng tối đa khoảng 600s. Không retry ngoài Router."
        )
        if st.button(
            "Dự đoán và lưu qua Queue/Worker",
            key="cuong_llm.submit",
            disabled=not (selected and confirmed),
        ):
            try:
                selected_frame = (
                    frame.set_index("customerID", drop=False)
                    .loc[selected]
                    .reset_index(drop=True)
                )
                adapter = DashboardLLMAdapter(form, pin, ledger, mode=provider_mode)
                run = adapter.prepare(
                    selected_frame, source_name, st.session_state[REQUEST_KEY]
                )
                st.session_state[LAST_ATTEMPT_KEY] = run.run_id
                with st.spinner(f"Đang chạy LLM/RAG {provider_mode} và lưu kết quả…"):
                    job = execute(adapter, run)
                current = ledger.get(run.run_id)
                customers, _ = ledger.frames(run.run_id)
                if not customers.empty:
                    st.session_state[ACTIVE_RUN_KEY] = run.run_id
                if current.status != "COMPLETE":
                    st.warning(
                        f"Run {current.status}: chỉ các dòng thành công được dùng; chưa chấm các dòng còn lại. Không tự retry."
                    )
                else:
                    st.success(
                        f"Đã lưu {len(customers)} dự đoán {provider_mode.upper()}; job {job.status.value if job else 'không nhận'}."
                    )
            except Exception:  # noqa: BLE001 -- No provider/DB/CSV contents in errors.
                st.error(
                    "Không thể hoàn tất lượt dự đoán; không tự gọi lại LLM. Kết quả đã lưu trước đó được giữ."
                )
        if st.button(
            "Tạo lượt mới để chủ động dự đoán lại",
            key="cuong_llm.new_request",
            on_click=_new_request,
        ):
            st.info("Đã cấp request ID mới; chưa gọi LLM. Bấm Dự đoán khi đã sẵn sàng.")
        jobs = SqliteTaskQueue(ledger.path).list_jobs(limit=5)
        if jobs:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Job": j.job_id[:8],
                            "Status": j.status.value,
                            "Retry": j.retry_count,
                        }
                        for j in jobs
                    ]
                ),
                hide_index=True,
            )
        running = SqliteTaskQueue(ledger.path).list_running_jobs()
        if running:
            st.caption(
                "Job bị ngắt không tự retry. Chỉ đóng khi chắc chắn worker đã dừng."
            )
            selected_job = st.selectbox(
                "Job cần khôi phục",
                [job.job_id for job in running],
                key="cuong_llm.interrupted_job",
            )
            context = (
                selected_job,
                next(job.updated_at for job in running if job.job_id == selected_job),
            )
            if st.session_state.get("cuong_llm.recovery_context") != context:
                st.session_state["cuong_llm.worker_stopped"] = False
                st.session_state["cuong_llm.recovery_context"] = context
            safe = st.checkbox(
                "Tôi xác nhận worker của job này đã dừng",
                key="cuong_llm.worker_stopped",
            )
            if st.button(
                "Đóng job bị ngắt, không gọi lại LLM",
                disabled=not safe,
                key="cuong_llm.interrupt",
            ):
                try:
                    job = next(job for job in running if job.job_id == selected_job)
                    if ledger.recover_interrupted_job(job.job_id):
                        interrupted_id = str(job.payload["run_id"])
                        if not ledger.frames(interrupted_id)[0].empty:
                            st.session_state[ACTIVE_RUN_KEY] = interrupted_id
                        st.success(
                            "Đã khôi phục trạng thái từ dữ liệu đã lưu; không gọi lại LLM."
                        )
                except Exception:  # noqa: BLE001 -- Do not expose database payloads.
                    st.error("Không thể khôi phục job; dữ liệu được giữ nguyên.")
    last_attempt = st.session_state.get(LAST_ATTEMPT_KEY)
    if last_attempt:
        attempted = ledger.get(last_attempt)
        if attempted.status in {"PARTIAL", "FAILED"}:
            st.error(
                f"Lần Submit gần nhất: {attempted.status}. {failure_message(ledger.error_code(last_attempt))} Không tự retry; các dòng chưa chạy không có kết quả thành công."
            )
            try:
                st.dataframe(
                    safe_export_frame(ledger.progress(last_attempt)), hide_index=True
                )
            except (ValueError, TypeError, KeyError):
                st.info(failure_message("snapshot"))
    run_id = st.session_state.get(ACTIVE_RUN_KEY)
    if not run_id:
        st.info(
            "Chọn khách hàng và bấm Dự đoán để mở Dashboard. Không có inference khi rerun."
        )
        return None
    run = ledger.get(run_id)
    if run.execution_mode not in ("mock", "live") or run.reference_sha256 != pin:
        st.error("Context kết quả không khớp chế độ preview/reference pin hiện tại.")
        return None
    customers, predictions = ledger.frames(run_id)
    if customers.empty:
        st.info("Run chưa có kết quả thành công để hiển thị.")
        return None
    st.caption(
        f"Đang hiển thị snapshot đã lưu: {len(customers)}/{len(json.loads(run.cohort_json))} dòng · {run.status} · run {run.run_id}. Bản nháp CSV không thay kết quả này."
    )
    provider_label = (
        f"LIVE ({', '.join(sorted(set(predictions['provider'])))})"
        if run.execution_mode == "live"
        and "provider" in predictions.columns
        and not predictions.empty
        else run.execution_mode.upper()
    )
    st.text(
        f"Nguồn: {run.source_name} | Reference: {run.reference_version} | {provider_label} provider"
    )
    return DashboardView(run, customers, predictions)


def _repo() -> SqliteCampaignRepository:
    return SqliteCampaignRepository(
        runtime_path().with_name("campaigns-preview.sqlite3")
    )


def create_campaign(**kwargs) -> Campaign:
    ledger = DashboardRunStore(runtime_path())
    run = ledger.get(kwargs["scoring_run_id"])
    if run.status not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("Campaign requires accepted predictions")
    customers, predictions = ledger.frames(run.run_id)
    assumptions = kwargs["assumptions"]
    dataset_id = hashlib.sha256(
        customers.to_json(orient="records").encode()
    ).hexdigest()
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
            given = (
                actual[expected.columns]
                .sort_values("customerID")
                .reset_index(drop=True)
            )
            if len(wanted) != len(given):
                raise ValueError("Campaign allocation count differs")
            for left, right in zip(
                wanted.to_dict("records"), given.to_dict("records"), strict=True
            ):
                for column in expected.columns:
                    want, got = left[column], right[column]
                    if pd.isna(want) or (
                        isinstance(want, Real) and not math.isfinite(want)
                    ):
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
            "Không thể ghi nhận xuất CSV. Trạng thái chiến dịch chưa được xác nhận; dữ liệu được giữ nguyên."
        )
