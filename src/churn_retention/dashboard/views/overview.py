"""Executive overview; rendering never performs inference."""

import plotly.graph_objects as go
import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state, visual_improvements
from churn_retention.dashboard.i18n import t
from churn_retention.dashboard.planning import allocation, executive_report, recommendations, totals
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services

ctx = state.context()
# Clear visual hierarchy với storytelling
st.markdown(f"## 📊 {t('ov_title')}")
st.markdown(f"### {t('ov_subtitle')}")
st.caption(t("ov_caption"))

# Navigation breadcrumb
st.markdown(t("ov_breadcrumb"))

# Add clear focal point
st.markdown("---")

if ctx.view is None:
    ui.empty(
        t("ov_empty_title"),
        t("ov_empty_desc"),
    )
    st.page_link(
        "views/customers.py", label=t("ov_link_import"), icon=":material/upload_file:"
    )
    if ctx.data is not None:
        st.success(f"Đã chọn {len(ctx.data.frame)} khách hàng · {ctx.data.source}")
        st.page_link("views/analysis.py", label=t("ov_link_analysis"))
else:
    view = ctx.view
    ui.badge(
        ui.mode_label(view.run.execution_mode),
        "warning" if view.run.execution_mode == "mock" else "success",
    )
    st.caption(
        f"Nguồn đang xem: {view.run.source_name} · {len(view.customers)} khách có kết quả đã lưu"
    )
    high = int((view.predictions.churn_prob > 0.7).sum())
    campaigns = services.list_campaigns()
    # Tổ chức layout với visual hierarchy rõ ràng
    st.markdown(f"### {t('ov_kpi_title')}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            t("ov_metric_analyzed"),
            f"{len(view.customers):,}",
            f"{len(view.customers)} khách",
            delta_color="off",
        )

    with col2:
        st.metric(
            t("ov_metric_high_risk"),
            str(high),
            f"{high / len(view.customers) * 100:.1f}% tổng số",
            delta_color="inverse" if high > 0 else "off",
        )

    with col3:
        active_customers = int(((view.customers.Churn == "No") & (view.customers.tenure > 0)).sum())
        st.metric(
            t("ov_metric_active"),
            str(active_customers),
            f"{active_customers / len(view.customers) * 100:.1f}% retention",
            delta_color="normal",
        )

    with col4:
        st.metric(
            t("ov_metric_campaigns"),
            str(len(campaigns)),
            f"{len(campaigns)} kế hoạch",
            delta_color="off",
        )

    # Add whitespace
    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left, ui.panel("overview-risk"):
        st.markdown(f"### {t('ov_risk_title')}")
        st.caption(t("ov_risk_caption"))

        # Thay vì tabs, dùng progressive disclosure
        show_pie = st.checkbox(t("ov_show_pie"), value=False, key="show_pie_chart")

        risk_cats = [t("ov_risk_low"), t("ov_risk_medium"), t("ov_risk_high")]
        counts = (
            view.predictions.churn_prob.map(ui.risk_label)
            .value_counts()
            .reindex(["Thấp", "Trung bình", "Cao"], fill_value=0)
        )
        if st.session_state.get("ui_lang", "vi") == "en":
            counts.index = ["Low", "Medium", "High"]

        # Biểu đồ chính - luôn hiển thị
        if len(view.predictions) >= 3:
            dark = st.session_state.get("ui_dark_mode", False)
            text_color = "#f1f5f9" if dark else "#1a202c"
            axis_color = "#cbd5e1" if dark else "#4a5568"
            grid_color = "#334155" if dark else "#e2e8f0"

            fig = go.Figure(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=["#10b981", "#f59e0b", "#ef4444"],  # Green, Amber, Red
                    marker_line_width=0,
                    text=counts.values,
                    textposition="outside",
                    textfont=dict(size=13, color=text_color),
                    hovertemplate="<b>%{x}</b><br>%{y} khách hàng<br>%{customdata}% tổng số<extra></extra>",
                    customdata=[
                        (counts.values[i] / counts.values.sum() * 100).round(1)
                        for i in range(len(counts))
                    ],
                )
            )

            fig.update_layout(
                title=None,
                xaxis=dict(
                    title=None,
                    color=axis_color,
                    gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(size=12),
                ),
                yaxis=dict(
                    title="Số khách hàng" if st.session_state.get("ui_lang", "vi") == "vi" else "Customers",
                    title_font=dict(size=12, color=axis_color),
                    color=axis_color,
                    gridcolor=grid_color,
                    dtick=max(1, counts.values.max() // 5),
                ),
                margin=dict(l=10, r=10, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="#1e293b" if dark else "white",
                    font_size=12,
                    font_family="Inter",
                ),
            )

            # Thêm insight từ data
            high_count = counts.iloc[2] if len(counts) > 2 else 0
            if high_count > 0:
                high_pct = (high_count / counts.values.sum() * 100).round(1)
                st.info(f"{t('ov_insight_prefix')} {high_pct}% khách hàng có nguy cơ cao cần can thiệp ngay")

            st.plotly_chart(fig, use_container_width=True)

            # Progressive disclosure: Pie chart
            if show_pie:
                st.markdown("---")
                st.markdown(t("ov_pie_title"))
                pie_fig = visual_improvements.create_risk_pie_chart(view.predictions)
                st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.dataframe(
                counts.rename_axis("Mức nguy cơ").reset_index(name="Số khách hàng"),
                hide_index=True,
                use_container_width=True,
            )

        st.page_link("views/analysis.py", label=t("ov_risk_link"))
    with right, ui.panel("overview-plan"):
        st.markdown(f"### {t('ov_plan_title')}")
        st.caption(t("ov_plan_caption"))

        plan = totals(allocation(view, st.session_state[state.SETTINGS]))

        # Tổ chức metrics với visual hierarchy
        st.markdown(t("ov_plan_expected"))

        if plan["total_net_value"] > 0:
            roi_color = "normal"
            roi_icon = "📈"
        else:
            roi_color = "inverse"
            roi_icon = "📉"

        # Primary metric - Net Value
        st.metric(
            f"{roi_icon} {t('ov_metric_net')}",
            ui.money(plan["total_net_value"]),
            f"ROI: {(plan['total_net_value'] / plan['total_cost'] * 100):.0f}%"
            if plan["total_cost"] > 0
            else "N/A",
            delta_color=roi_color,
        )

        # Secondary metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                t("ov_metric_intervene"),
                str(plan["n_intervened"]),
                f"{plan['n_intervened']}/{plan['n_total']} khách",
                delta_color="off",
            )

        with col2:
            st.metric(
                t("ov_metric_cost"),
                ui.money(plan["total_cost"]),
                f"{plan['total_cost'] / plan['total_benefit'] * 100:.0f}% lợi ích"
                if plan["total_benefit"] > 0
                else "N/A",
                delta_color="off",
            )

        # Clear call-to-action
        st.markdown("---")

        cta_col1, cta_col2 = st.columns(2)
        with cta_col1:
            if st.button(t("ov_btn_plan"), use_container_width=True, type="primary"):
                st.switch_page("views/plan.py")

        with cta_col2:
            if st.button(t("ov_btn_campaigns"), use_container_width=True):
                st.switch_page("views/campaigns.py")

        # Quick insight
        if plan["n_intervened"] == 0 and plan["total_cost"] == 0:
            st.info(t("ov_no_plan_info"))
    # Executive summary với progressive disclosure
    st.markdown("---")

    summary_expanded = st.checkbox(t("ov_exec_toggle"), value=False, key="exec_summary")

    if summary_expanded:
        st.markdown(t("ov_exec_title"))
        st.caption(t("ov_exec_caption"))

        recs = recommendations(view, st.session_state[state.SETTINGS])
        grouped = (
            recs.groupby("action_code", sort=True)
            .agg(
                n=("customerID", "size"),
                cost=("cost", "sum"),
                benefit=("benefit", "sum"),
                net=("net_value", "sum"),
            )
            .reset_index()
            .rename(
                columns={
                    "action_code": "Gói",
                    "n": "Số khách",
                    "cost": "Chi phí (USD)",
                    "benefit": "Lợi ích (USD)",
                    "net": "Lợi ích ròng (USD)",
                }
            )
        )
        grouped["Chi phí (USD)"] = grouped["Chi phí (USD)"].map(ui.money)
        grouped["Lợi ích (USD)"] = grouped["Lợi ích (USD)"].map(ui.money)
        grouped["Lợi ích ròng (USD)"] = grouped["Lợi ích ròng (USD)"].map(ui.money)
        st.dataframe(grouped, hide_index=True, width="stretch")

        report = executive_report(view, st.session_state[state.SETTINGS])
        st.download_button(
            t("ov_exec_download"),
            report,
            file_name=f"{view.run.execution_mode}_retention_summary.md",
            mime="text/markdown",
        )

# Workflow guidance với visual hierarchy rõ ràng
st.markdown("---")
st.markdown(t("ov_workflow_title"))

st.markdown(t("ov_step1_title"))
col1a, col1b = st.columns([3, 1])
with col1a:
    st.markdown(t("ov_step1_body"))
    st.caption(t("ov_step1_caption"))
with col1b:
    if st.button(t("ov_step1_btn"), use_container_width=True):
        st.switch_page("views/customers.py")

st.markdown(t("ov_step2_title"))
col2a, col2b = st.columns([3, 1])
with col2a:
    st.markdown(t("ov_step2_body"))
    st.caption(t("ov_step2_caption"))
with col2b:
    if st.button(t("ov_step2_btn"), use_container_width=True):
        st.switch_page("views/analysis.py")

st.markdown(t("ov_step3_title"))
col3a, col3b = st.columns([3, 1])
with col3a:
    st.markdown(t("ov_step3_body"))
    st.caption(t("ov_step3_caption"))
with col3b:
    if st.button(t("ov_step3_btn"), use_container_width=True):
        st.switch_page("views/plan.py")

# Quick tip
st.info(t("ov_tip"))
