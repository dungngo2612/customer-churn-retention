"""Executive overview; rendering never performs inference."""

import plotly.graph_objects as go
import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state, visual_improvements
from churn_retention.dashboard.i18n import t
from churn_retention.dashboard.planning import allocation, executive_report, recommendations, totals
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services

ctx = state.context()
dark = st.session_state.get("ui_dark_mode", False)
lang = st.session_state.get("ui_lang", "vi")

card_bg = "rgba(30, 41, 59, 0.7)" if dark else "rgba(248, 250, 252, 0.95)"
border_color = "rgba(255, 255, 255, 0.12)" if dark else "rgba(226, 232, 240, 0.9)"
text_title = "#f8fafc" if dark else "#0f172a"
text_body = "#cbd5e1" if dark else "#475569"
text_secondary = "#94a3b8" if dark else "#64748b"

# ── Hero Introduction Banner ──────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.12) 0%, rgba(147, 51, 234, 0.08) 100%);
                border: 1px solid {border_color}; border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;">
        <div style="display: inline-block; background-color: rgba(79, 70, 229, 0.2); color: {'#a5b4fc' if dark else '#4f46e5'}; 
                    font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 20px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">
            🚀 AI Customer Churn Retention Platform
        </div>
        <h1 style="color: {text_title}; font-size: 26px; font-weight: 700; margin: 0 0 10px 0; font-family: Inter, sans-serif;">
            {'Trung tâm Điều hành & Giữ chân Khách hàng Viễn thông' if lang == 'vi' else 'Customer Churn Retention Operations Hub'}
        </h1>
        <p style="color: {text_body}; font-size: 14px; line-height: 1.6; margin: 0; max-width: 950px;">
            {'Nền tảng ứng dụng trí tuệ nhân tạo (AI/LLM) và tối ưu hóa kinh tế học chuyên sâu cho ngành viễn thông. Hệ thống giúp phát hiện sớm nguy cơ khách hàng rời bỏ, giải thích nguyên nhân gốc rễ theo góc nhìn nghiệp vụ thực tế và tự động phân bổ ngân sách can thiệp (hỗ trợ kỹ thuật, hợp đồng, cước phí) để tối đa hóa doanh thu giữ chân (MRR/ARR).' if lang == 'vi' else 'AI/LLM & economic optimization platform specialized for telecom retention. Detect early subscriber churn risk, uncover root business causes, and automatically allocate intervention budgets to maximize recurring revenue preservation (MRR/ARR).'}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Core Capabilities 4 Cards ─────────────────────────────────────────────
st.markdown(f"### 💡 {'Năng lực Cốt lõi của Nền tảng' if lang == 'vi' else 'Core Platform Capabilities'}")
cap_cols = st.columns(4)

capabilities = [
    {
        "icon": "🔍",
        "title_vi": "Dự đoán Rời bỏ Đa chiều",
        "title_en": "Multi-Factor Churn Scoring",
        "desc_vi": "Phân tích 19 trường dữ liệu khách hàng (hợp đồng, cước phí, dịch vụ gia tăng) để tính toán chính xác xác suất rời bỏ.",
        "desc_en": "Evaluate 19 customer attributes (contracts, monthly bills, add-ons) to accurately compute churn probability.",
        "color": "#3b82f6",
    },
    {
        "icon": "💡",
        "title_vi": "Giải thích Nghiệp vụ Thực tế",
        "title_en": "Actionable Business Insights",
        "desc_vi": "Chỉ rõ lý do vì sao thuê bao muốn rời mạng (cước cao, thiếu hỗ trợ kỹ thuật hay hợp đồng ngắn hạn) thay vì các con số vô hồn.",
        "desc_en": "Pinpoint concrete business drivers (high charges, lack of tech support, month-to-month terms) rather than black-box scores.",
        "color": "#10b981",
    },
    {
        "icon": "🎯",
        "title_vi": "Tối ưu Hóa Ngân sách Can thiệp",
        "title_en": "Economic Budget Optimization",
        "desc_vi": "Thuật toán kinh tế học phân bổ ngân sách vào các gói can thiệp (Kỹ thuật A1, Hợp đồng A2, Giảm cước A3) nhằm đạt ROI cao nhất.",
        "desc_en": "Algorithmic resource allocation across 3 interventions (Support A1, Contract A2, Discount A3) for maximal ROI.",
        "color": "#f59e0b",
    },
    {
        "icon": "📊",
        "title_vi": "Khám phá & Phân khúc Khách hàng",
        "title_en": "EDA & Customer Personas",
        "desc_vi": "Trực quan hóa dữ liệu tức thì (EDA) và tự động phân nhóm VIP, Khách mới, Khách nhạy cảm giá để chăm sóc đúng đối tượng.",
        "desc_en": "Instant EDA visualization and automatic segmentation into VIPs, New Friction, and Price-Sensitive personas.",
        "color": "#8b5cf6",
    },
]

for i, cap in enumerate(capabilities):
    with cap_cols[i]:
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; border: 1px solid {border_color}; border-top: 4px solid {cap['color']};
                        border-radius: 10px; padding: 16px; height: 100%; min-height: 180px;">
                <div style="font-size: 24px; margin-bottom: 8px;">{cap['icon']}</div>
                <div style="font-size: 15px; font-weight: 600; color: {text_title}; margin-bottom: 6px;">
                    {cap['title_vi'] if lang == 'vi' else cap['title_en']}
                </div>
                <div style="font-size: 13px; color: {text_secondary}; line-height: 1.5;">
                    {cap['desc_vi'] if lang == 'vi' else cap['desc_en']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# ── 4-Step Retention Lifecycle Workflow ────────────────────────────────────
st.markdown(f"### 🔄 {'Quy trình Vận hành Giữ chân 4 Bước' if lang == 'vi' else '4-Step Retention Lifecycle Workflow'}")

wf_cols = st.columns(4)
steps = [
    {
        "step": "Bước 1" if lang == "vi" else "Step 1",
        "icon": "📁",
        "name_vi": "Khách hàng & Dữ liệu",
        "name_en": "Customers & Data",
        "desc_vi": "Nạp file CSV, khám phá tương quan EDA và xem phân khúc khách hàng.",
        "desc_en": "Import CSV, explore EDA charts, and inspect customer personas.",
        "link": "views/customers.py",
        "link_text": "Xem Khách hàng →" if lang == "vi" else "Go to Customers →",
    },
    {
        "step": "Bước 2" if lang == "vi" else "Step 2",
        "icon": "🔍",
        "name_vi": "Phân tích Rời bỏ AI",
        "name_en": "AI Churn Analysis",
        "desc_vi": "Chạy mô hình dự đoán xác suất churn và phân tích nguyên nhân nghiệp vụ.",
        "desc_en": "Run AI scoring and generate feature-based churn explanations.",
        "link": "views/analysis.py",
        "link_text": "Xem Phân tích →" if lang == "vi" else "Go to Analysis →",
    },
    {
        "step": "Bước 3" if lang == "vi" else "Step 3",
        "icon": "🎯",
        "name_vi": "Kế hoạch Giữ chân",
        "name_en": "Retention Planning",
        "desc_vi": "Thiết lập ngân sách, chọn chính sách can thiệp A1–A3 tối ưu ROI.",
        "desc_en": "Configure budget limits, optimize actions A1–A3 for maximum net benefit.",
        "link": "views/plan.py",
        "link_text": "Xem Kế hoạch →" if lang == "vi" else "Go to Plan →",
    },
    {
        "step": "Bước 4" if lang == "vi" else "Step 4",
        "icon": "🚀",
        "name_vi": "Quản lý Chiến dịch",
        "name_en": "Campaign Execution",
        "desc_vi": "Lưu snapshot chiến dịch và xuất file CSV chuyển giao cho Telesales/CSKH.",
        "desc_en": "Save campaign snapshots and export CSV for outbound CSKH teams.",
        "link": "views/campaigns.py",
        "link_text": "Xem Chiến dịch →" if lang == "vi" else "Go to Campaigns →",
    },
]

for i, s in enumerate(steps):
    with wf_cols[i]:
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; border: 1px solid {border_color}; border-radius: 10px; padding: 14px; height: 100%;">
                <div style="font-size: 11px; font-weight: 700; color: #6366f1; text-transform: uppercase; margin-bottom: 4px;">{s['step']}</div>
                <div style="font-size: 14px; font-weight: 600; color: {text_title}; margin-bottom: 4px;">
                    {s['icon']} {s['name_vi'] if lang == 'vi' else s['name_en']}
                </div>
                <div style="font-size: 12px; color: {text_secondary}; line-height: 1.4; margin-bottom: 10px;">
                    {s['desc_vi'] if lang == 'vi' else s['desc_en']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(s["link"], label=s["link_text"])

st.markdown("---")

if ctx.view is None:
    st.markdown(f"### 🚀 {'Trạng thái Hệ thống & Khởi động Nhanh' if lang == 'vi' else 'System Status & Quick Launch'}")
    if ctx.data is not None:
        st.success(
            f"✅ **{'Tập dữ liệu đang chọn' if lang == 'vi' else 'Active Dataset'}:** {ctx.data.source} "
            f"({len(ctx.data.frame)} {'khách hàng' if lang == 'vi' else 'customers'}). "
            f"{'Bạn có thể tiếp tục sang trang Phân tích để chạy dự đoán hoặc xem phân khúc tại trang Khách hàng.' if lang == 'vi' else 'You can now proceed to Churn Analysis or explore segments in Customers.'}"
        )
        c_link1, c_link2 = st.columns(2)
        with c_link1:
            st.page_link("views/analysis.py", label="🔍 " + ("Bắt đầu Phân tích Rời bỏ AI →" if lang == "vi" else "Start AI Churn Analysis →"), icon=":material/insights:")
        with c_link2:
            st.page_link("views/customers.py", label="📊 " + ("Xem Khám phá dữ liệu EDA & Phân khúc →" if lang == "vi" else "View EDA & Segmentation →"), icon=":material/group:")
    else:
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; border: 1px dashed {border_color}; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 32px; margin-bottom: 8px;">📂</div>
                <div style="font-size: 17px; font-weight: 600; color: {text_title}; margin-bottom: 6px;">
                    {'Hệ thống đã sẵn sàng phân tích' if lang == 'vi' else 'System Ready for Analysis'}
                </div>
                <div style="font-size: 14px; color: {text_secondary}; max-width: 650px; margin: 0 auto;">
                    {'Để bắt đầu, bạn hãy chuyển sang trang <b>Khách hàng & dữ liệu</b> để tải lên file CSV của bạn hoặc chọn trải nghiệm ngay với dữ liệu mẫu có sẵn.' if lang == 'vi' else 'To get started, switch to the <b>Customers & data</b> page to upload your CSV file or explore with built-in sample data.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            st.page_link("views/customers.py", label="📁 " + ("Chuyển tới trang Quản lý Khách hàng & Tải dữ liệu →" if lang == "vi" else "Go to Customers & Import Data →"), icon=":material/group:")
        with c_act2:
            st.page_link("views/analysis.py", label="🔍 " + ("Xem trang Phân tích Rời bỏ AI →" if lang == "vi" else "Go to Churn Analysis →"), icon=":material/insights:")
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
