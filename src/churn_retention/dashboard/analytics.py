"""Exploratory Data Analysis (EDA) and Customer Segmentation for raw & uploaded datasets."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churn_retention.dashboard.i18n import t


def _chart_theme() -> tuple[str, str, str, str]:
    dark = st.session_state.get("ui_dark_mode", False)
    text_color = "#f1f5f9" if dark else "#1e293b"
    muted_color = "#94a3b8" if dark else "#64748b"
    grid_color = "rgba(255, 255, 255, 0.08)" if dark else "rgba(0, 0, 0, 0.06)"
    bg_color = "rgba(0, 0, 0, 0)"
    return text_color, muted_color, grid_color, bg_color


def assign_personas(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize customers into distinct business personas based on contract, charges, and tenure."""
    result = df.copy()

    def classify(row: pd.Series) -> str:
        charges = float(row.get("MonthlyCharges", 0))
        contract = str(row.get("Contract", ""))
        tenure = int(row.get("tenure", 0))
        tech_support = str(row.get("TechSupport", "No"))
        payment = str(row.get("PaymentMethod", ""))

        # 1. VIP nguy cơ cao: Cước cao (>= $70) nhưng hợp đồng theo tháng (Month-to-month)
        if charges >= 70.0 and contract == "Month-to-month":
            return "vip_at_risk"

        # 2. Khách hàng mới gặp trở ngại: Dưới 12 tháng, không có hỗ trợ kỹ thuật
        if tenure <= 12 and tech_support == "No":
            return "new_friction"

        # 3. Khách hàng trung thành: Gắn bó trên 24 tháng, hợp đồng 1 hoặc 2 năm
        if tenure >= 24 and contract in ["One year", "Two year"]:
            return "loyal_core"

        # 4. Khách nhạy cảm về giá: Trả qua Electronic check hoặc cước >= $50 với hợp đồng tháng
        if payment == "Electronic check" or (charges >= 50.0 and contract == "Month-to-month"):
            return "price_sensitive"

        return "standard"

    result["persona_code"] = [classify(row) for _, row in result.iterrows()]
    return result


PERSONA_CONFIG = {
    "vip_at_risk": {
        "icon": "💎",
        "name_vi": "Khách VIP nguy cơ cao",
        "name_en": "High-Value VIP at Risk",
        "color": "#ef4444",
        "bg_color": "rgba(239, 68, 68, 0.12)",
        "risk_level": "Rất cao (Nguy cơ mất doanh thu lớn nhất)",
        "risk_level_en": "Critical (Highest MRR exposure)",
        "strategy_vi": "Can thiệp ưu tiên cấp 1: Telesales gọi riêng, chiết khấu 15% kèm cam kết hợp đồng 1 năm để 'khóa' chân khách hàng.",
        "strategy_en": "Priority 1 intervention: Dedicated account manager call, 15% discount for 1-year contract lock.",
    },
    "new_friction": {
        "icon": "⚠️",
        "name_vi": "Khách mới gặp trở ngại",
        "name_en": "New Customer Friction",
        "color": "#f59e0b",
        "bg_color": "rgba(245, 158, 11, 0.12)",
        "risk_level": "Cao (Rời bỏ sớm do trải nghiệm ban đầu)",
        "risk_level_en": "High (Early churn due to onboarding friction)",
        "strategy_vi": "Tặng gói Hỗ trợ kỹ thuật 24/7 miễn phí 3 tháng + gửi hướng dẫn thiết lập mạng tối ưu.",
        "strategy_en": "Free 3-month TechSupport 24/7 package + proactive onboarding network checkup.",
    },
    "price_sensitive": {
        "icon": "💰",
        "name_vi": "Khách nhạy cảm về giá",
        "name_en": "Price-Sensitive Seeker",
        "color": "#3b82f6",
        "bg_color": "rgba(59, 130, 246, 0.12)",
        "risk_level": "Trung bình (Dễ bị hút bởi khuyến mãi đối thủ)",
        "risk_level_en": "Moderate (Vulnerable to competitor price promotions)",
        "strategy_vi": "Tái cơ cấu gói cước: bỏ bớt dịch vụ không dùng để giảm hóa đơn hoặc chuyển sang thanh toán tự động để nhận ưu đãi.",
        "strategy_en": "Bundle optimization: remove unused add-ons or offer autopay discount incentives.",
    },
    "loyal_core": {
        "icon": "🛡️",
        "name_vi": "Khách hàng trung thành",
        "name_en": "Loyal Core Base",
        "color": "#10b981",
        "bg_color": "rgba(16, 185, 129, 0.12)",
        "risk_level": "Thấp (Nền tảng doanh thu bền vững)",
        "risk_level_en": "Low (Predictable, recurring revenue foundation)",
        "strategy_vi": "Chương trình tri ân (Loyalty Rewards): Tích điểm đổi quà, ưu tiên nâng cấp băng thông khi ra mắt gói mới.",
        "strategy_en": "Loyalty appreciation: Points reward program, priority bandwidth upgrades.",
    },
    "standard": {
        "icon": "👥",
        "name_vi": "Khách hàng tiêu chuẩn",
        "name_en": "Standard Customer Base",
        "color": "#64748b",
        "bg_color": "rgba(100, 116, 139, 0.12)",
        "risk_level": "Trung bình - Thấp",
        "risk_level_en": "Moderate to Low",
        "strategy_vi": "Duy trì chất lượng dịch vụ ổn định và khảo sát mức độ hài lòng định kỳ.",
        "strategy_en": "Maintain consistent service quality and periodic satisfaction surveys.",
    },
}


def render_eda_tab(df: pd.DataFrame) -> None:
    """Render Exploratory Data Analysis charts and metrics for the active dataset."""
    lang = st.session_state.get("ui_lang", "vi")
    text_color, muted_color, grid_color, bg_color = _chart_theme()

    st.markdown(
        f"### 📊 {'Khám phá & Chẩn đoán tập dữ liệu' if lang == 'vi' else 'Exploratory Data Analysis (EDA)'}"
    )
    st.caption(
        "Tổng quan phân bố cước phí, loại hợp đồng và tỷ lệ sử dụng dịch vụ của tập khách hàng trước khi chạy dự đoán AI."
        if lang == "vi"
        else "Overview of charges distribution, contract types, and service adoption before running AI inference."
    )

    # ── Summary KPI Cards ──────────────────────────────────────────────────────────
    n_total = len(df)
    avg_charges = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 0.0
    fiber_pct = (
        (df["InternetService"] == "Fiber optic").mean() * 100
        if "InternetService" in df.columns
        else 0.0
    )
    no_tech_pct = (
        (df["TechSupport"] == "No").mean() * 100 if "TechSupport" in df.columns else 0.0
    )
    m2m_pct = (
        (df["Contract"] == "Month-to-month").mean() * 100 if "Contract" in df.columns else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="👥 " + ("Tổng khách hàng" if lang == "vi" else "Total Customers"),
            value=f"{n_total:,}",
        )
    with c2:
        st.metric(
            label="💵 " + ("Cước tháng TB (MRR)" if lang == "vi" else "Avg Monthly Bill"),
            value=f"${avg_charges:.2f}",
        )
    with c3:
        st.metric(
            label="⚡ " + ("Dùng Cáp quang" if lang == "vi" else "Fiber Optic Users"),
            value=f"{fiber_pct:.1f}%",
        )
    with c4:
        st.metric(
            label="⚠️ " + ("Hợp đồng theo tháng" if lang == "vi" else "Month-to-Month"),
            value=f"{m2m_pct:.1f}%",
        )

    st.divider()

    # ── Row 1: Contract Distribution vs Churn & Monthly Charges Histogram ──────
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            f"#### 📄 {'Cơ cấu Hợp đồng & Trạng thái Rời bỏ' if lang == 'vi' else 'Contract Structure & Churn Status'}"
        )
        if "Contract" in df.columns and "Churn" in df.columns:
            contract_order = ["Month-to-month", "One year", "Two year"]
            contract_labels = {
                "Month-to-month": "Từng tháng" if lang == "vi" else "Month-to-month",
                "One year": "1 năm" if lang == "vi" else "1 Year",
                "Two year": "2 năm" if lang == "vi" else "2 Years",
            }
            grouped = (
                df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0).reset_index()
            )
            # Ensure columns exist
            for col in ["No", "Yes"]:
                if col not in grouped.columns:
                    grouped[col] = 0

            # Sort by canonical contract order
            grouped["order"] = grouped["Contract"].map(
                lambda x: contract_order.index(x) if x in contract_order else 99
            )
            grouped = grouped.sort_values("order")

            fig_contract = go.Figure()
            fig_contract.add_trace(
                go.Bar(
                    x=[contract_labels.get(c, c) for c in grouped["Contract"]],
                    y=grouped["No"],
                    name="Đang hoạt động" if lang == "vi" else "Active",
                    marker_color="#10b981",
                )
            )
            fig_contract.add_trace(
                go.Bar(
                    x=[contract_labels.get(c, c) for c in grouped["Contract"]],
                    y=grouped["Yes"],
                    name="Đã rời bỏ" if lang == "vi" else "Churned",
                    marker_color="#ef4444",
                )
            )
            fig_contract.update_layout(
                barmode="stack",
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=text_color, family="Inter, sans-serif"),
                margin=dict(l=20, r=20, t=20, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                xaxis=dict(gridcolor=grid_color, color=muted_color),
                yaxis=dict(gridcolor=grid_color, color=muted_color),
                height=320,
            )
            st.plotly_chart(fig_contract, use_container_width=True)
            st.caption(
                "💡 **Nhận định:** Khách hàng ký hợp đồng từng tháng (Month-to-month) luôn có tỷ lệ rời bỏ cao nhất. Chuyển đổi sang hợp đồng 1–2 năm là chìa khóa giữ chân cốt lõi."
                if lang == "vi"
                else "💡 **Insight:** Month-to-month contracts exhibit the highest churn rate. Migrating users to 1-2 year contracts is a primary retention lever."
            )

    with col_chart2:
        st.markdown(
            f"#### 💳 {'Phân bố Tiền cước hàng tháng (USD)' if lang == 'vi' else 'Monthly Charges Distribution (USD)'}"
        )
        if "MonthlyCharges" in df.columns:
            fig_charges = px.histogram(
                df,
                x="MonthlyCharges",
                color="Churn" if "Churn" in df.columns else None,
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                labels={
                    "MonthlyCharges": "Cước tháng ($)"
                    if lang == "vi"
                    else "Monthly Charges ($)",
                    "count": "Số khách" if lang == "vi" else "Customers",
                    "Churn": "Rời bỏ" if lang == "vi" else "Churn",
                },
                nbins=20,
            )
            fig_charges.add_vline(
                x=avg_charges,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"TB: ${avg_charges:.1f}",
                annotation_position="top right",
                annotation_font_color=text_color,
            )
            fig_charges.update_layout(
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=text_color, family="Inter, sans-serif"),
                margin=dict(l=20, r=20, t=20, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                xaxis=dict(gridcolor=grid_color, color=muted_color),
                yaxis=dict(gridcolor=grid_color, color=muted_color),
                height=320,
            )
            st.plotly_chart(fig_charges, use_container_width=True)
            st.caption(
                "💡 **Nhận định:** Nhóm cước cao ($70 - $105/tháng) thường dùng Cáp quang và nhiều gói giải trí, là nhóm tạo doanh thu cao nhưng cũng rất nhạy cảm với chất lượng dịch vụ."
                if lang == "vi"
                else "💡 **Insight:** High-charge tier ($70 - $105/mo) generates high MRR but exhibits vulnerability if service quality drops."
            )

    # ── Row 2: Add-on Services Adoption ──────────────────────────────────────────
    st.markdown(
        f"#### 🛡️ {'Mức độ sử dụng các Dịch vụ Viễn thông & Gia tăng' if lang == 'vi' else 'Telecommunication & Add-on Services Adoption'}"
    )
    service_cols = [
        ("OnlineSecurity", "Bảo mật trực tuyến", "Online Security"),
        ("TechSupport", "Hỗ trợ kỹ thuật 24/7", "Tech Support"),
        ("OnlineBackup", "Sao lưu đám mây", "Online Backup"),
        ("DeviceProtection", "Bảo hiểm thiết bị", "Device Protection"),
        ("StreamingTV", "Truyền hình trực tuyến", "Streaming TV"),
        ("StreamingMovies", "Phim trực tuyến", "Streaming Movies"),
    ]
    rates = []
    for col_name, label_vi, label_en in service_cols:
        if col_name in df.columns:
            pct_has = (df[col_name] == "Yes").mean() * 100
            rates.append(
                {
                    "Dịch vụ": label_vi if lang == "vi" else label_en,
                    "Tỷ lệ sử dụng (%)": pct_has,
                }
            )

    if rates:
        rates_df = pd.DataFrame(rates).sort_values(
            "Tỷ lệ sử dụng (%)", ascending=True
        )
        fig_services = go.Figure(
            go.Bar(
                x=rates_df["Tỷ lệ sử dụng (%)"],
                y=rates_df["Dịch vụ"],
                orientation="h",
                marker=dict(
                    color=rates_df["Tỷ lệ sử dụng (%)"],
                    colorscale="Blues",
                    showscale=False,
                ),
                text=[f"{v:.1f}%" for v in rates_df["Tỷ lệ sử dụng (%)"]],
                textposition="outside",
            )
        )
        fig_services.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, family="Inter, sans-serif"),
            margin=dict(l=20, r=40, t=10, b=20),
            xaxis=dict(
                gridcolor=grid_color, color=muted_color, range=[0, 105], title="%"
            ),
            yaxis=dict(gridcolor=grid_color, color=muted_color),
            height=280,
        )
        st.plotly_chart(fig_services, use_container_width=True)


def render_segmentation_tab(df: pd.DataFrame) -> None:
    """Render Customer Segmentation (Personas) with action playbooks and interactive filtering."""
    lang = st.session_state.get("ui_lang", "vi")
    classified = assign_personas(df)
    n_total = len(classified)

    st.markdown(
        f"### 👥 {'Phân khúc Khách hàng & Chân dung Rủi ro' if lang == 'vi' else 'Customer Segmentation & Risk Personas'}"
    )
    st.caption(
        "Tự động nhóm khách hàng thành 4 chân dung kinh doanh cốt lõi dựa trên hành vi hợp đồng, chi tiêu và thời gian gắn bó."
        if lang == "vi"
        else "Automatically segment customers into 4 core business personas based on contract behavior, spending, and tenure."
    )

    # ── Cards summary ────────────────────────────────────────────────────────────
    counts = classified["persona_code"].value_counts()
    cols = st.columns(4)
    persona_keys = ["vip_at_risk", "new_friction", "price_sensitive", "loyal_core"]

    for i, p_key in enumerate(persona_keys):
        cfg = PERSONA_CONFIG[p_key]
        p_count = counts.get(p_key, 0)
        p_pct = (p_count / n_total * 100) if n_total > 0 else 0.0
        p_mrr = (
            classified[classified["persona_code"] == p_key]["MonthlyCharges"].sum()
            if p_count > 0
            else 0.0
        )

        with cols[i]:
            st.markdown(
                f"""
                <div style="background-color: {cfg['bg_color']}; border-left: 4px solid {cfg['color']}; padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="font-size: 20px;">{cfg['icon']} <b>{cfg['name_vi'] if lang == 'vi' else cfg['name_en']}</b></div>
                    <div style="font-size: 24px; font-weight: bold; margin: 6px 0; color: {cfg['color']};">{p_count:,} <span style="font-size: 14px; font-weight: normal; color: inherit;">({p_pct:.1f}%)</span></div>
                    <div style="font-size: 12px; opacity: 0.85;">MRR: <b>${p_mrr:,.0f}/tháng</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Interactive Persona Selector & Action Playbook ───────────────────────────
    st.markdown(
        f"#### 🎯 {'Chi tiết phân khúc & Chiến lược can thiệp' if lang == 'vi' else 'Persona Breakdown & Action Playbook'}"
    )

    options = [
        ("vip_at_risk", f"💎 {PERSONA_CONFIG['vip_at_risk']['name_vi'] if lang == 'vi' else PERSONA_CONFIG['vip_at_risk']['name_en']} ({counts.get('vip_at_risk', 0)})"),
        ("new_friction", f"⚠️ {PERSONA_CONFIG['new_friction']['name_vi'] if lang == 'vi' else PERSONA_CONFIG['new_friction']['name_en']} ({counts.get('new_friction', 0)})"),
        ("price_sensitive", f"💰 {PERSONA_CONFIG['price_sensitive']['name_vi'] if lang == 'vi' else PERSONA_CONFIG['price_sensitive']['name_en']} ({counts.get('price_sensitive', 0)})"),
        ("loyal_core", f"🛡️ {PERSONA_CONFIG['loyal_core']['name_vi'] if lang == 'vi' else PERSONA_CONFIG['loyal_core']['name_en']} ({counts.get('loyal_core', 0)})"),
    ]

    selected_persona_code = st.radio(
        "Chọn nhóm phân khúc để xem chi tiết:" if lang == "vi" else "Select persona to inspect:",
        [opt[0] for opt in options],
        format_func=lambda code: next(opt[1] for opt in options if opt[0] == code),
        horizontal=True,
        key="seg_persona_radio",
    )

    selected_cfg = PERSONA_CONFIG[selected_persona_code]
    p_subset = classified[classified["persona_code"] == selected_persona_code]

    # Action Playbook Banner
    st.info(
        f"**{selected_cfg['icon']} Mức độ rủi ro:** {selected_cfg['risk_level'] if lang == 'vi' else selected_cfg['risk_level_en']}\n\n"
        f"**📋 Chiến lược can thiệp:** {selected_cfg['strategy_vi'] if lang == 'vi' else selected_cfg['strategy_en']}"
    )

    if p_subset.empty:
        st.caption(
            "Tập dữ liệu hiện tại không có khách hàng nào thuộc phân khúc này."
            if lang == "vi"
            else "No customers in the current dataset match this persona."
        )
    else:
        st.markdown(
            f"**Danh sách {len(p_subset)} khách hàng thuộc phân khúc này:**"
            if lang == "vi"
            else f"**List of {len(p_subset)} customers in this persona:**"
        )
        display_cols = [
            "customerID",
            "Contract",
            "MonthlyCharges",
            "tenure",
            "InternetService",
            "TechSupport",
            "Churn",
        ]
        available_cols = [col for col in display_cols if col in p_subset.columns]
        sub_table = p_subset[available_cols].copy().rename(
            columns={
                "customerID": "Mã KH" if lang == "vi" else "Customer ID",
                "Contract": "Hợp đồng" if lang == "vi" else "Contract",
                "MonthlyCharges": "Cước tháng ($)" if lang == "vi" else "Monthly ($)",
                "tenure": "Thâm niên (tháng)" if lang == "vi" else "Tenure (mo)",
                "InternetService": "Internet",
                "TechSupport": "Hỗ trợ KT" if lang == "vi" else "TechSupport",
                "Churn": "Rời bỏ" if lang == "vi" else "Churn",
            }
        )
        st.dataframe(sub_table, use_container_width=True, hide_index=True)

