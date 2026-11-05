"""Saved campaign snapshots, explicit exports and technical job recovery."""

import pandas as pd
import streamlit as st

from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state
from churn_retention.dashboard.i18n import t
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.modules.campaigns import CampaignStatus
from churn_retention.modules.campaigns.use_cases import safe_export_frame

STATUSES = {"DRAFT": "Bản nháp", "SIMULATED": "Đã lưu mô phỏng", "EXPORTED": "Đã yêu cầu xuất CSV"}
JOB_STATUSES = {
    "PENDING": "Đang chờ",
    "RUNNING": "Đang xử lý",
    "SUCCESS": "Hoàn tất",
    "FAILED": "Không hoàn tất",
    "RETRYING": "Đang chờ lại",
}


def exported(campaign_id: str) -> None:
    try:
        services.mark_exported(campaign_id)
        if services.get_campaign(campaign_id).status != CampaignStatus.EXPORTED:
            raise ValueError("Export status not confirmed")
        st.session_state.pop("campaign.export_error", None)
    except Exception:  # noqa: BLE001 -- Durable error, no paths/raw DB details.
        st.session_state["campaign.export_error"] = (
            "Không thể ghi nhận yêu cầu xuất CSV. Dữ liệu chiến dịch vẫn được giữ nguyên."
        )


ctx = state.context()
ui.header(
    t("ca_eyebrow"),
    t("ca_title"),
    t("ca_desc"),
)
archive_tab, queue_tab = st.tabs([t("ca_tab_archive"), t("ca_tab_queue")])

with archive_tab:
    campaigns = services.list_campaigns()
    if not campaigns:
        ui.empty(
            "Chưa có chiến dịch",
            "Sau khi phân tích, lập phương án ngân sách và lưu tại Kế hoạch giữ chân.",
        )
        st.page_link("views/plan.py", label="Đến kế hoạch giữ chân →")
    else:
        controls = st.columns([2, 1, 1])
        search = ui.persist(
            controls[0].text_input, "Tìm theo tên chiến dịch", key="campaign.search"
        )
        status = ui.persist(
            controls[1].selectbox,
            "Trạng thái",
            ["Tất cả", *STATUSES],
            format_func=lambda v: STATUSES.get(v, v),
            key="campaign.status",
        )
        mode = ui.persist(
            controls[2].selectbox,
            "Nguồn kết quả",
            ["Tất cả", "mock", "live"],
            format_func=lambda v: ui.mode_label(v) if v != "Tất cả" else v,
            key="campaign.mode",
        )
        filtered = [
            c
            for c in campaigns
            if search.casefold() in c.name.casefold()
            and (status == "Tất cả" or c.status.value == status)
            and (mode == "Tất cả" or c.assumptions.get("execution_mode") == mode)
        ]
        filtered.sort(key=lambda c: (c.created_at, c.campaign_id), reverse=True)
        frame = pd.DataFrame(
            [
                {
                    "Chiến dịch": c.name,
                    "Trạng thái": STATUSES[c.status.value],
                    "Ngày tạo": ui.date_label(c.created_at),
                    "Ngân sách (USD)": c.budget,
                    "Khách được chọn": c.results.get("summary", {}).get("n_intervened", 0),
                    "Nguồn": ui.mode_label(c.assumptions.get("execution_mode", "")),
                }
                for c in filtered
            ]
        )
        ui.table(
            frame,
            "campaign.list",
            column_config={"Ngân sách (USD)": st.column_config.NumberColumn(format="$%.2f")},
        )
        if filtered:
            ids = [c.campaign_id for c in filtered]
            selected = st.session_state.get(state.CAMPAIGN)
            if selected in ids and st.session_state.get("campaign.opened") != selected:
                st.session_state["campaign.selected"] = selected
            elif st.session_state.get("campaign.selected") not in ids:
                st.session_state["campaign.selected"] = ids[0]
            cid = st.selectbox(
                "Mở chiến dịch",
                ids,
                format_func=lambda v: next(c.name for c in filtered if c.campaign_id == v),
                key="campaign.selected",
            )
            st.session_state[state.CAMPAIGN] = cid
            st.session_state["campaign.opened"] = cid
            campaign = services.get_campaign(cid)
            with ui.panel("campaign-detail"):
                ui.section(campaign.name, f"Tạo lúc {ui.date_label(campaign.created_at)}")
                ui.badge(STATUSES[campaign.status.value], "success")
                st.caption(ui.mode_label(campaign.assumptions.get("execution_mode", "")))
                summary = campaign.results.get("summary", {})
                ui.summary(
                    [
                        ("Ngân sách", ui.money(campaign.budget), "USD"),
                        ("Chi phí dự kiến", ui.money(summary.get("total_cost")), "Mô phỏng"),
                        ("Lợi ích ròng", ui.money(summary.get("total_net_value")), "Mô phỏng"),
                        (
                            "Khách được chọn",
                            str(summary.get("n_intervened", 0)),
                            "Chưa thực thi can thiệp",
                        ),
                    ]
                )
                scope = st.radio(
                    "Phạm vi xem và xuất CSV",
                    ["Khách được can thiệp", "Toàn bộ phương án đã lưu"],
                    horizontal=True,
                    key="campaign.export_scope",
                )
                only = scope == "Khách được can thiệp"
                allocations = services.export_campaign_dataframe(cid)
                if only and not allocations.empty:
                    allocations = allocations[allocations.action_code != "A0"].copy()
                ui.table(
                    allocations.rename(
                        columns={
                            "customer_id": "Mã khách hàng",
                            "action_code": "Gói",
                            "action_name": "Phương án",
                            "cost": "Chi phí (USD)",
                            "benefit": "Lợi ích (USD)",
                            "net_value": "Lợi ích ròng (USD)",
                        }
                    ),
                    "campaign.customers",
                )
                st.caption(
                    f"Xuất toàn bộ {len(allocations)} dòng thuộc phạm vi trên; "
                    "không chỉ trang đang xem."
                )
                st.download_button(
                    "Tải danh sách chiến dịch (.csv)",
                    services.export_campaign_csv(cid, only_interventions=only),
                    file_name=(
                        f"{campaign.assumptions.get('execution_mode', 'unknown')}"
                        f"_chien_dich_{cid[:8]}.csv"
                    ),
                    mime="text/csv",
                    key="campaign.download",
                    on_click=exported,
                    args=(cid,),
                )
                if st.session_state.get("campaign.export_error"):
                    st.error(st.session_state["campaign.export_error"])
                st.caption(
                    (
                        "Đã yêu cầu xuất CSV chỉ ghi nhận thao tác tải, không có nghĩa đã "
                        "triển khai chiến dịch."
                    )
                )
                with st.expander("Nguồn kết quả & giả định đã lưu"):
                    source = ctx.ledger.get(campaign.scoring_run_id)
                    st.write(f"Nguồn dữ liệu: {source.source_name}")
                    st.write(
                        f"Biên lợi nhuận: {campaign.assumptions['margin']:.0%} · "
                        f"Kỳ hạn: {campaign.assumptions['horizon_months']} tháng"
                    )
                    st.text(f"Reference: {campaign.assumptions['reference_version']}")
                    if st.button("Mở lần phân tích nguồn", key="campaign.source"):
                        state.open_run(campaign.scoring_run_id)
                        st.session_state["_analysis_next_tab"] = "Kết quả & lý do"
                        st.switch_page("views/analysis.py")

with queue_tab:
    ui.section(
        "Công việc phân tích",
        (
            "Hàng đợi xử lý kỹ thuật hiện có. Không phải danh sách phân công "
            "nhân viên; không tự chạy lại API."
        ),
    )
    queue = SqliteTaskQueue(ctx.ledger.path)
    jobs = queue.list_jobs(50)
    if not jobs:
        ui.empty(
            "Chưa có công việc xử lý", "Công việc xuất hiện sau khi bạn chủ động chạy phân tích."
        )
    else:
        ui.table(
            pd.DataFrame(
                [
                    {
                        "Mã công việc": j.job_id[:8],
                        "Trạng thái": JOB_STATUSES.get(j.status.value, j.status.value),
                        "Tạo lúc": ui.date_label(j.created_at),
                        "Số lần thử lại": j.retry_count,
                    }
                    for j in jobs
                ]
            ),
            "queue.jobs",
        )
        job_id = st.selectbox(
            "Xem tiến độ công việc",
            [j.job_id for j in jobs],
            format_func=lambda v: next(
                f"{JOB_STATUSES.get(j.status.value, j.status.value)} · {j.job_id[:8]}"
                for j in jobs
                if j.job_id == v
            ),
            key="queue.detail",
        )
        job = next(j for j in jobs if j.job_id == job_id)
        if "run_id" in job.payload:
            progress = ctx.ledger.progress(str(job.payload["run_id"]))
            progress = progress.drop(columns="error_code").rename(
                columns={
                    "customer_id": "Mã khách hàng",
                    "state": "Trạng thái",
                    "message": "Thông báo",
                }
            )
            progress["Trạng thái"] = progress["Trạng thái"].map(ui.processing_label)
            st.dataframe(safe_export_frame(progress), hide_index=True, width="stretch")
    running = queue.list_running_jobs()
    if running:
        with st.expander("Khôi phục công việc bị gián đoạn"):
            st.warning(
                (
                    "Chỉ đóng khi chắc chắn tiến trình xử lý đã dừng. Không thực hiện "
                    "lại các lời gọi API chưa rõ kết quả."
                )
            )
            selected = st.selectbox(
                "Công việc cần kiểm tra",
                [j.job_id for j in running],
                key="cuong_llm.interrupted_job",
            )
            job = next(j for j in running if j.job_id == selected)
            binding = (selected, job.updated_at)
            if st.session_state.get("cuong_llm.recovery_context") != binding:
                st.session_state["cuong_llm.worker_stopped"] = False
                st.session_state["cuong_llm.recovery_context"] = binding
            confirmed = st.checkbox(
                "Tôi xác nhận tiến trình xử lý công việc này đã dừng",
                key="cuong_llm.worker_stopped",
            )
            if st.button(
                "Đóng công việc bị gián đoạn", disabled=not confirmed, key="cuong_llm.interrupt"
            ):
                try:
                    if ctx.ledger.recover_interrupted_job(job.job_id):
                        if not ctx.ledger.frames(str(job.payload["run_id"]))[0].empty:
                            state.open_run(str(job.payload["run_id"]))
                        st.success("Đã khôi phục trạng thái từ dữ liệu đã lưu. Không gọi lại LLM.")
                except Exception:  # noqa: BLE001 -- Preserve state and hide raw DB errors.
                    st.error("Không thể khôi phục công việc. Dữ liệu đã lưu vẫn được giữ nguyên.")
