"""Explicit single/batch prediction, accepted results and saved run history."""

import hashlib
import os
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn_retention.adapters.database.dashboard_runs import canonical
from churn_retention.contracts.llm_failure import failure_message
from churn_retention.dashboard import components as ui
from churn_retention.dashboard import customer_detail, state
from churn_retention.dashboard.planning import recommendations
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.entrypoints.streamlit.dashboard_llm_service import DashboardLLMAdapter
from churn_retention.modules.campaigns.use_cases import safe_export_frame

STATUS = {
    "COMPLETE": "Hoàn tất",
    "PARTIAL": "Hoàn tất một phần",
    "FAILED": "Không hoàn tất",
    "RUNNING": "Đang xử lý",
    "PENDING": "Đang chờ",
}

ctx = state.context()
ui.header(
    "Hiểu nguy cơ, xem lý do",
    "Phân tích rời bỏ",
    "Phân tích khách hàng đã chọn, xem kết quả và tiếp tục lập kế hoạch.",
)
if st.session_state.get("_analysis_next_tab"):
    st.session_state["analysis.tabs"] = st.session_state.pop("_analysis_next_tab")
run_tab, result_tab, history_tab = st.tabs(
    ["Chạy phân tích", "Kết quả & lý do", "Lịch sử"], key="analysis.tabs"
)

with run_tab:
    if ctx.data is None:
        ui.empty(
            "Chưa chọn dữ liệu để phân tích",
            "Nhập tệp hoặc chọn dữ liệu mẫu tại Khách hàng & dữ liệu.",
        )
        st.page_link("views/customers.py", label="Chọn dữ liệu →")
    elif ctx.form is None or st.session_state.get(state.INPUT_ERROR):
        st.error(st.session_state.get(state.INPUT_ERROR) or ctx.error)
        st.page_link("views/customers.py", label="Kiểm tra lại dữ liệu →")
    else:
        data = ctx.data
        ui.badge(
            "Dữ liệu mẫu · hồ sơ giả" if data.sample else "Dữ liệu đã kiểm tra",
            "warning" if data.sample else "success",
        )
        st.caption(f"Nguồn: {data.source} · {len(data.frame)} khách hàng")
        with ui.panel("analysis-run"):
            ui.section("Chọn phạm vi phân tích")
            scope = st.radio(
                "Số khách trong lần chạy",
                ["Một khách hàng", "Batch nhỏ (tối đa 5)"],
                index=1 if st.session_state.get("analysis.scope") == "Batch nhỏ (tối đa 5)" else 0,
                key="cuong_llm.scope",
                horizontal=True,
                format_func=lambda x: "Theo lô · tối đa 5 khách" if x.startswith("Batch") else x,
            )
            st.session_state["analysis.scope"] = scope
            ids = data.frame.customerID.tolist()
            if scope == "Một khách hàng":
                prior = st.session_state.get("analysis.single")
                if (
                    "cuong_llm.single_id" not in st.session_state
                    or st.session_state["cuong_llm.single_id"] not in ids
                ):
                    st.session_state["cuong_llm.single_id"] = prior if prior in ids else ids[0]
                chosen = st.selectbox("Khách hàng cần phân tích", ids, key="cuong_llm.single_id")
                st.session_state["analysis.single"] = chosen
                selected = [chosen]
            else:
                prior_ids = st.session_state.get(
                    "cuong_llm.selected_ids", st.session_state.get("analysis.selection", [])
                )
                valid_ids = [cid for cid in prior_ids if cid in ids]
                if "cuong_llm.selected_ids" not in st.session_state or valid_ids != prior_ids:
                    st.session_state["cuong_llm.selected_ids"] = valid_ids
                selected = st.multiselect(
                    "Chọn khách hàng (tối đa 5)",
                    ids,
                    max_selections=5,
                    key="cuong_llm.selected_ids",
                )
                st.session_state["analysis.selection"] = selected
            selected_frame = (
                data.frame.set_index("customerID", drop=False).loc[selected].reset_index(drop=True)
            )
            if selected:
                st.dataframe(
                    safe_export_frame(
                        selected_frame[["customerID", "Contract", "MonthlyCharges"]]
                    ).rename(
                        columns={
                            "customerID": "Mã khách hàng",
                            "Contract": "Hợp đồng",
                            "MonthlyCharges": "Cước tháng (USD)",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            mode: Literal["mock", "live"] = "mock"
            allow_live = (
                os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") == "1"
                and os.environ.get("CHURN_DASHBOARD_PREVIEW") != "1"
            )
            if allow_live:
                choice = st.radio(
                    "Cách phân tích",
                    ["Mock provider (offline demo)", "Live API (Groq/Gemini)"],
                    format_func=lambda x: (
                        "Mô phỏng · không gọi API"
                        if x.startswith("Mock")
                        else "API thật · có thể phát sinh phí"
                    ),
                    key="cuong_llm.provider_mode",
                )
                if choice.startswith("Live"):
                    mode = "live"
            if mode == "mock":
                st.info(
                    (
                        "Chế độ mô phỏng: không gửi dữ liệu đến API. Kết quả chỉ để trải "
                        "nghiệm hệ thống."
                    )
                )
            else:
                st.warning(
                    (
                        "API thật gửi 19 thuộc tính của khách đã chọn và dữ liệu tham "
                        "chiếu đến Groq/Gemini. Có thể phát sinh phí, gồm lời gọi dự "
                        "phòng."
                    )
                )
            binding = (
                mode,
                ctx.pin,
                hashlib.sha256(
                    canonical(selected_frame.to_dict(orient="records")).encode()
                ).hexdigest(),
            )
            if st.session_state.get("cuong_llm.confirm_context") != binding:
                st.session_state["cuong_llm.confirm"] = False
                st.session_state["cuong_llm.confirm_context"] = binding
            confirmed = st.checkbox(
                "Tôi xác nhận phạm vi và chế độ phân tích đã chọn", key="cuong_llm.confirm"
            )
            st.caption(
                (
                    "Chỉ chạy khi bấm nút. Tối đa 120 giây/khách; lô 5 khách có thể "
                    "cần 10 phút. Không tự gọi lại khi đổi trang."
                )
            )
            if st.button(
                "Chạy phân tích",
                type="primary",
                key="cuong_llm.submit",
                disabled=not (selected and confirmed),
            ):
                try:
                    adapter = DashboardLLMAdapter(ctx.form, ctx.pin, ctx.ledger, mode=mode)
                    run = adapter.prepare(
                        selected_frame, data.source, st.session_state[services.REQUEST_KEY]
                    )
                    st.session_state[services.LAST_ATTEMPT_KEY] = run.run_id
                    with st.spinner(f"Đang phân tích {len(selected)} khách hàng và lưu kết quả…"):
                        services.execute(adapter, run)
                    customers, _ = ctx.ledger.frames(run.run_id)
                    if not customers.empty:
                        state.open_run(run.run_id)
                    st.session_state["_analysis_next_tab"] = "Kết quả & lý do"
                    st.rerun()
                except Exception:  # noqa: BLE001 -- Sanitized boundary; do not replay or discard saved results.
                    st.error(
                        (
                            "Không thể hoàn tất phân tích. Không tự gọi lại LLM; kết quả đã "
                            "lưu được giữ nguyên."
                        )
                    )
            st.button(
                "Bắt đầu một lượt phân tích mới",
                key="cuong_llm.new_request",
                on_click=services._new_request,
                help="Tạo yêu cầu mới để chủ động phân tích lại. Thao tác này chưa gọi API.",
            )

with result_tab:
    attempt = st.session_state.get(services.LAST_ATTEMPT_KEY)
    if attempt:
        attempted = ctx.ledger.get(attempt)
        if attempted.status in {"PARTIAL", "FAILED"}:
            st.error(
                f"Lần phân tích gần nhất: {STATUS[attempted.status]}. "
                f"{failure_message(ctx.ledger.error_code(attempt))} Không tự thử lại."
            )
            progress = (
                ctx.ledger.progress(attempt)
                .drop(columns="error_code")
                .rename(
                    columns={
                        "customer_id": "Mã khách hàng",
                        "state": "Trạng thái",
                        "message": "Thông báo",
                    }
                )
            )
            progress["Trạng thái"] = progress["Trạng thái"].map(ui.processing_label)
            st.dataframe(safe_export_frame(progress), hide_index=True, width="stretch")
    if ctx.view is None:
        ui.empty(
            "Chưa có kết quả phân tích",
            (
                "Chọn khách hàng và xác nhận chạy tại mục Chạy phân tích. Kết quả "
                "chỉ xuất hiện khi đã lưu thành công."
            ),
        )
    else:
        view = ctx.view
        ui.badge(
            ui.mode_label(view.run.execution_mode),
            "warning" if view.run.execution_mode == "mock" else "success",
        )
        st.caption(
            f"{view.run.source_name} · {STATUS.get(view.run.status, view.run.status)} · "
            f"{len(view.customers)} khách có kết quả"
        )
        st.caption("Điểm LLM chưa hiệu chuẩn. Phân nhóm: cao >70%, trung bình 40–70%, thấp <40%.")
        if ctx.data is not None and not state.matches_input(view.run, ctx.data):
            st.warning(
                (
                    "Dữ liệu đang chọn khác snapshot này. Kết quả dưới đây và kế hoạch"
                    " vẫn dùng dữ liệu đã lưu của lần phân tích, không dùng bản mới."
                )
            )
        frame = view.customers.merge(
            view.predictions, left_on="customerID", right_on="customer_id", validate="one_to_one"
        )
        frame = frame.merge(
            recommendations(view, st.session_state[state.SETTINGS])[
                ["customerID", "action_code", "action_name"]
            ],
            on="customerID",
            how="left",
            validate="one_to_one",
        )
        frame["Nguy cơ"] = frame.churn_prob.map(ui.risk_label)
        controls = st.columns([2, 1, 1])
        search = ui.persist(controls[0].text_input, "Tìm mã khách hàng", key="analysis.search")
        risk = ui.persist(
            controls[1].selectbox,
            "Nguy cơ",
            ["Tất cả", "Cao", "Trung bình", "Thấp"],
            key="analysis.risk",
        )
        action = ui.persist(
            controls[2].selectbox,
            "Đề xuất",
            ["Tất cả", "Có can thiệp", "Không can thiệp"],
            key="analysis.action",
        )
        filtered = frame[frame.customerID.str.contains(search, case=False, regex=False)].copy()
        if risk != "Tất cả":
            filtered = filtered[filtered["Nguy cơ"] == risk]
        if action != "Tất cả":
            filtered = filtered[
                filtered.action_code.notna()
                & (
                    (filtered.action_code != "A0")
                    if action == "Có can thiệp"
                    else (filtered.action_code == "A0")
                )
            ]
        filtered = filtered.sort_values(
            ["churn_prob", "customerID"], ascending=[False, True], kind="stable"
        )
        shown = filtered[["customerID", "churn_prob", "Nguy cơ", "action_name"]].rename(
            columns={
                "customerID": "Mã khách hàng",
                "churn_prob": "Điểm rời bỏ",
                "action_name": "Đề xuất",
            }
        )
        shown["Điểm rời bỏ"] = shown["Điểm rời bỏ"].map(ui.percent)
        shown["Đề xuất"] = shown["Đề xuất"].fillna("Không thuộc diện can thiệp")
        ui.table(shown, "analysis.results")
        st.download_button(
            "Xuất danh sách kết quả đã lọc (.csv)",
            safe_export_frame(filtered).to_csv(index=False),
            file_name=f"{view.run.execution_mode}_ket_qua_da_loc.csv",
            mime="text/csv",
        )
        customer_detail.render(filtered, "analysis", view)
        st.page_link(
            "views/plan.py",
            label="Lập kế hoạch từ lần phân tích này →",
            icon=":material/arrow_forward:",
        )
        with st.expander("Quan sát theo nhóm khách hàng"):
            st.caption(
                (
                    "Tỷ lệ Churn dưới đây là dữ liệu đã quan sát trong snapshot; không"
                    " phải kết quả LLM hay tác động nhân quả."
                )
            )
            for field, title in [
                ("Contract", "Hợp đồng"),
                ("InternetService", "Internet"),
                ("TechSupport", "Hỗ trợ kỹ thuật"),
                ("PaymentMethod", "Thanh toán"),
            ]:
                observed = view.customers.assign(
                    observed_churn=view.customers.Churn.eq("Yes").astype(int)
                )
                grouped = (
                    observed.groupby(field).observed_churn.agg(["mean", "count"]).reset_index()
                )
                grouped[field] = grouped[field].map(customer_detail.translated)
                st.markdown(f"**{title}**")
                if len(grouped) > 1 and len(observed) >= 5:
                    fig = go.Figure(
                        go.Bar(x=grouped[field], y=grouped["mean"], marker_color=ui.INDIGO)
                    )
                    fig.update_yaxes(title="Churn quan sát", tickformat=".0%")
                    ui.chart(fig, key=f"eda.{field}")
                else:
                    grouped["mean"] = grouped["mean"].map(ui.percent)
                    st.dataframe(
                        grouped.rename(
                            columns={field: title, "mean": "Churn quan sát", "count": "Số khách"}
                        ),
                        hide_index=True,
                        width="stretch",
                    )

with history_tab:
    runs = state.recent_runs(ctx)
    ui.section("Các lần phân tích đã lưu", "Tối đa 50 công việc gần nhất. Mở lại không gọi LLM.")
    if not runs:
        ui.empty("Chưa có lịch sử", "Lần phân tích sẽ được ghi lại khi bạn chủ động chạy.")
    else:
        ui.table(
            pd.DataFrame(
                [
                    {
                        "Lần phân tích": r.run_id[:8],
                        "Nguồn": r.source_name,
                        "Chế độ": ui.mode_label(r.execution_mode),
                        "Trạng thái": STATUS.get(r.status, r.status),
                    }
                    for r in runs
                ]
            ),
            "analysis.history",
        )
        run_id = st.selectbox(
            "Chọn lần phân tích",
            [r.run_id for r in runs],
            format_func=lambda value: next(
                f"{r.source_name} · {STATUS.get(r.status, r.status)} · {r.run_id[:8]}"
                for r in runs
                if r.run_id == value
            ),
            key="history.selected",
        )
        if st.button("Mở kết quả đã lưu", key="history.open"):
            state.open_run(run_id)
            st.session_state[services.LAST_ATTEMPT_KEY] = run_id
            st.session_state["_analysis_next_tab"] = "Kết quả & lý do"
            st.rerun()
