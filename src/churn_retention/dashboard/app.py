"""Streamlit Dashboard for Customer Retention & Economic Optimization (Person 3).

Chạy dashboard bằng lệnh:
    streamlit run src/churn_retention/dashboard/app.py
"""

from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Đảm bảo Python nhận diện được package src/
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_retention.adapters.predictor.fake import FakePredictor
from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    Condition,
    MANUAL_PAYMENTS,
    default_actions,
    evaluate_actions,
    recommend,
    sensitivity_by_margin,
)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ thống Tối ưu Hóa Giữ Chân Khách Hàng | Executive DSS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL CUSTOM CSS ĐỒNG BỘ THEME HIỆN ĐẠI (EXECUTIVE SAAS VIBE) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Bỏ khoảng trống thừa phía trên */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }

    /* Hero Banner hiện đại */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-desc {
        color: #cbd5e1;
        font-size: 0.95rem;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .pill-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .status-pill {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8fafc;
    }

    /* KPI Card Containers */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .modern-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .card-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .card-footer {
        font-size: 0.84rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .badge-emerald {
        background: #ecfdf5;
        color: #047857;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-indigo {
        background: #eef2ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-rose {
        background: #ffe4e6;
        color: #e11d48;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-amber {
        background: #fef3c7;
        color: #d97706;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* Container bao bọc các biểu đồ và panel */
    .chart-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .chart-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Hồ sơ 360 styling */
    .profile-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .profile-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #e2e8f0;
        font-size: 0.88rem;
    }
    .profile-item:last-child {
        border-bottom: none;
    }
    .profile-key {
        color: #64748b;
        font-weight: 500;
    }
    .profile-val {
        color: #0f172a;
        font-weight: 700;
    }

    .recommend-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .recommend-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #15803d;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .recommend-sub {
        font-size: 0.88rem;
        color: #166534;
        line-height: 1.4;
    }

    /* Tùy biến thanh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Tùy biến Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #475569;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #4338ca !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR: CẤU HÌNH THAM SỐ KINH TẾ (GIỮ NGUYÊN 100% LOGIC) ---
with st.sidebar:
    st.markdown("### ⚙️ Thiết Lập Kịch Bản")
    st.caption("Cấu hình giả định cho mô hình Proxy CLV")

    base_margin = st.slider(
        "Biên lợi nhuận giả định (m)",
        min_value=0.10,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Dùng để tính Proxy CLV: V = MonthlyCharges * H * m",
    )
    horizon = st.selectbox(
        "Kỳ hạn đánh giá H (tháng)",
        options=[6, 12, 24],
        index=1,
        help="Kỳ hạn ước lượng dòng tiền giữ chân",
    )

    st.markdown("---")
    st.markdown("### 🛠️ Chi Phí & Hiệu Quả Gói Can Thiệp")
    st.caption("Tùy biến tham số chi phí ($c, f$), tỷ lệ nhận ($u$) & mức giảm ($e$)")

    with st.expander("A1: Hỗ trợ Kỹ thuật & Bảo mật", expanded=False):
        c1_contact = st.number_input("A1: Chi phí liên hệ ($)", value=2.0, step=0.5, key="c1_c")
        c1_offer = st.number_input("A1: Chi phí ưu đãi ($)", value=15.0, step=1.0, key="c1_o")
        c1_u = st.slider("A1: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05, key="c1_u")
        c1_e = st.slider("A1: Mức giảm churn (e)", 0.1, 1.0, 0.25, 0.05, key="c1_e")

    with st.expander("A2: Tư vấn Hợp đồng & Thanh toán", expanded=False):
        c2_contact = st.number_input("A2: Chi phí liên hệ ($)", value=2.0, step=0.5, key="c2_c")
        c2_offer = st.number_input("A2: Chi phí ưu đãi ($)", value=20.0, step=1.0, key="c2_o")
        c2_u = st.slider("A2: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.35, 0.05, key="c2_u")
        c2_e = st.slider("A2: Mức giảm churn (e)", 0.1, 1.0, 0.40, 0.05, key="c2_e")

    with st.expander("A3: Ưu đãi Giảm Cước Dài Hạn", expanded=False):
        c3_contact = st.number_input("A3: Chi phí liên hệ ($)", value=3.0, step=0.5, key="c3_c")
        c3_offer = st.number_input("A3: Chi phí voucher ($)", value=40.0, step=2.0, key="c3_o")
        c3_u = st.slider("A3: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05, key="c3_u")
        c3_e = st.slider("A3: Mức giảm churn (e)", 0.1, 1.0, 0.50, 0.05, key="c3_e")

# Dựng lại danh mục hành động theo tham số người dùng nhập
current_actions = (
    ActionConfig("A0", "Không can thiệp", 0, 0, 0, 0, is_no_action=True),
    ActionConfig(
        "A1", "Hỗ trợ kỹ thuật & bảo mật chủ động",
        c1_contact, c1_offer, c1_u, c1_e,
        all_of=(Condition("InternetService", "ne", "No"),),
        any_of=(Condition("TechSupport", "eq", "No"), Condition("OnlineSecurity", "eq", "No")),
    ),
    ActionConfig(
        "A2", "Tư vấn hợp đồng & thanh toán tự động",
        c2_contact, c2_offer, c2_u, c2_e,
        any_of=(Condition("Contract", "eq", "Month-to-month"), Condition("PaymentMethod", "in", MANUAL_PAYMENTS)),
    ),
    ActionConfig(
        "A3", "Ưu đãi giảm cước cam kết dài hạn",
        c3_contact, c3_offer, c3_u, c3_e,
        all_of=(Condition("Contract", "eq", "Month-to-month"), Condition("MonthlyCharges", "ge", 50.0)),
    ),
)


# --- NẠP DỮ LIỆU & CHẠY DỰ ĐOÁN (FAKE PREDICTOR CỦA NGƯỜI 1) ---
@st.cache_data
def load_data_and_predict():
    possible_paths = [
        ROOT_DIR / "data" / "raw" / "telco_churn.csv",
        ROOT_DIR / "data" / "telco_churn.csv",
        ROOT_DIR / "telco_churn.csv",
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)

    if data_path is None:
        st.warning("⚠️ Chưa tìm thấy file `telco_churn.csv`. Đang sử dụng dữ liệu mô phỏng.")
        customers = pd.DataFrame({
            "customerID": ["CUST_A", "CUST_B", "CUST_C"],
            "MonthlyCharges": [80.0, 25.0, 65.0],
            "Contract": ["Month-to-month", "Month-to-month", "One year"],
            "InternetService": ["Fiber optic", "DSL", "DSL"],
            "TechSupport": ["No", "No", "Yes"],
            "OnlineSecurity": ["No", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "Churn": ["No", "No", "No"],
            "tenure": [12, 5, 24],
        })
    else:
        customers = pd.read_csv(data_path)

    predictor = FakePredictor(seed=42)
    probs = predictor.predict_batch(
        features=customers,
        customer_ids=customers["customerID"].tolist()
    )
    predictions = pd.DataFrame({
        "customer_id": customers["customerID"].tolist(),
        "churn_prob": probs,
    })

    return customers, predictions


customers_df, predictions_df = load_data_and_predict()

# --- CHẠY KHUYẾN NGHỊ VÀ TỔNG HỢP ---
recommendations = recommend(
    customers=customers_df,
    predictions=predictions_df,
    actions=current_actions,
    margin=base_margin,
    horizon_months=horizon,
)

# Lọc các khách hàng có can thiệp thực sự (loại A0)
interventions = recommendations[recommendations["action_code"] != "A0"]

# --- HERO BANNER EXECUTIVE ---
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🎯</span> Hệ Thống Tối Ưu Hóa Chiến Lược Giữ Chân Khách Hàng
        </div>
        <div class="hero-desc">
            Khoa học Dịch vụ & Hỗ trợ Ra quyết định: Cá nhân hóa gói can thiệp giữ chân dựa trên Proxy CLV và tối đa hóa Lợi nhuận Ròng Kỳ vọng (ΔEV).
        </div>
        <div class="pill-group">
            <span class="status-pill">📊 Biên lợi nhuận m = {base_margin:.0%}</span>
            <span class="status-pill">⏱️ Kỳ hạn đánh giá H = {horizon} tháng</span>
            <span class="status-pill">👥 Tệp khách hàng nạp: {len(customers_df):,}</span>
            <span class="status-pill" style="background: rgba(16, 185, 129, 0.25); border-color: rgba(16, 185, 129, 0.4);">
                ✅ Hệ thống tối ưu hoạt động
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- TABS ĐIỀU HƯỚNG ---
tab1, tab2, tab3 = st.tabs([
    "📊 Tổng quan Kinh tế (Overview)",
    "👤 Hồ Sơ Khách Hàng 360° & Đề Xuất Can Thiệp",
    "📈 Phân tích Độ nhạy (Sensitivity)",
])

# ==============================================================================
# TAB 1: TỔNG QUAN KINH TẾ (OVERVIEW)
# ==============================================================================
with tab1:
    # 1. Tính toán các chỉ số KPI
    n_total = len(recommendations)
    n_intervened = len(interventions)
    intervention_rate = (n_intervened / n_total) if n_total > 0 else 0
    total_cost = interventions["cost"].sum()
    total_net_value = interventions["net_value"].sum()
    total_benefit = interventions["benefit"].sum()
    avg_cost_per_user = (total_cost / n_intervened) if n_intervened > 0 else 0
    roi = ((total_net_value / total_cost) * 100) if total_cost > 0 else 0

    # 4 KPI Cards
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="modern-card">
                <div class="card-label">
                    <span>Khách Hàng Hoạt Động</span>
                    <span>👥</span>
                </div>
                <div class="card-value">{n_total:,}</div>
                <div class="card-footer">
                    <span>Đủ điều kiện xét duyệt chính sách</span>
                </div>
            </div>
            <div class="modern-card">
                <div class="card-label">
                    <span>Quyết Định Can Thiệp</span>
                    <span>🎯</span>
                </div>
                <div class="card-value">{n_intervened:,}</div>
                <div class="card-footer">
                    Tỷ lệ tiếp cận: <span class="badge-indigo">{intervention_rate:.1%} tổng tệp</span>
                </div>
            </div>
            <div class="modern-card">
                <div class="card-label">
                    <span>Tổng Ngân Sách Dự Kiến</span>
                    <span>💳</span>
                </div>
                <div class="card-value">${total_cost:,.2f}</div>
                <div class="card-footer">
                    Trung bình: <strong>${avg_cost_per_user:,.2f}</strong> / khách
                </div>
            </div>
            <div class="modern-card">
                <div class="card-label">
                    <span>Lợi Nhuận Ròng (ΔEV)</span>
                    <span>🚀</span>
                </div>
                <div class="card-value" style="color: #059669;">+${total_net_value:,.2f}</div>
                <div class="card-footer">
                    Gross: <strong>${total_benefit:,.2f}</strong> | <span class="badge-emerald">ROI: +{roi:.1f}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. KHU VỰC BIỂU ĐỒ CAO CẤP VỚI PLOTLY (DONUT + WATERFALL)
    col_chart_left, col_chart_right = st.columns([1, 1.2])

    action_counts = recommendations["action_code"].value_counts().reset_index()
    action_counts.columns = ["action_code", "n_customers"]

    action_metrics = recommendations.groupby("action_code").agg(
        cost_sum=("cost", "sum"),
        benefit_sum=("benefit", "sum"),
        net_value_sum=("net_value", "sum"),
    ).reset_index()

    summary_df = action_counts.merge(action_metrics, on="action_code", how="left")
    action_names = {a.code: a.name for a in current_actions}
    summary_df["action_name"] = summary_df["action_code"].map(action_names)
    summary_df["percentage"] = (summary_df["n_customers"] / n_total) * 100 if n_total > 0 else 0

    with col_chart_left:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">🍩 Cơ Cấu Phân Bổ Hành Động Can Thiệp</div>', unsafe_allow_html=True)

        colors_map = {
            "A0": "#94a3b8",
            "A1": "#3b82f6",
            "A2": "#06b6d4",
            "A3": "#10b981",
        }
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=[f"{row.action_code}: {row.action_name}" for _, row in summary_df.iterrows()],
                    values=summary_df["n_customers"],
                    hole=0.58,
                    marker=dict(colors=[colors_map.get(code, "#6366f1") for code in summary_df["action_code"]]),
                    textinfo="percent+label",
                    textposition="inside",
                    hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,} khách<br>Tỷ lệ: %{percent}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=False,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>{n_total:,}</b><br><span style='font-size:12px;color:#64748b;'>Tổng khách</span>",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_right:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">🌊 Phân Rã Dòng Tiền Kinh Tế (Waterfall Analysis)</div>', unsafe_allow_html=True)

        total_contact_cost = sum(
            len(recommendations[recommendations["action_code"] == a.code]) * a.contact_cost
            for a in current_actions if not a.is_no_action
        )
        total_offer_cost = total_cost - total_contact_cost

        fig_waterfall = go.Figure(
            go.Waterfall(
                name="Tài chính",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Lợi Ích Gộp (Benefit)", "Chi Phí Liên Hệ", "Chi Phí Ưu Đãi", "Lợi Nhuận Ròng (ΔEV)"],
                textposition="outside",
                text=[f"+${total_benefit:,.0f}", f"-${total_contact_cost:,.0f}", f"-${total_offer_cost:,.0f}", f"+${total_net_value:,.0f}"],
                y=[total_benefit, -total_contact_cost, -total_offer_cost, total_net_value],
                connector=dict(line=dict(color="#94a3b8", width=1, dash="dot")),
                decreasing=dict(marker=dict(color="#f43f5e")),
                increasing=dict(marker=dict(color="#0284c7")),
                totals=dict(marker=dict(color="#10b981")),
            )
        )
        fig_waterfall.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Giá trị ($)", gridcolor="#f1f5f9"),
            xaxis=dict(tickfont=dict(size=12, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. SO SÁNH HIỆU QUẢ CÁC GÓI CAN THIỆP & BẢNG TỔNG HỢP CHI TIẾT
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chart-header">⚖️ So Sánh Hiệu Quả Tài Chính Từng Gói Can Thiệp (A1, A2, A3)</div>', unsafe_allow_html=True)

    interv_actions = summary_df[summary_df["action_code"] != "A0"].copy()
    col_bar, col_table = st.columns([1, 1.2])

    with col_bar:
        fig_grouped = go.Figure(
            data=[
                go.Bar(
                    name="Tổng Chi Phí",
                    x=interv_actions["action_code"],
                    y=interv_actions["cost_sum"],
                    marker_color="#f97316",
                    text=[f"${v:,.0f}" for v in interv_actions["cost_sum"]],
                    textposition="auto",
                ),
                go.Bar(
                    name="Lợi Nhuận Ròng (ΔEV)",
                    x=interv_actions["action_code"],
                    y=interv_actions["net_value_sum"],
                    marker_color="#10b981",
                    text=[f"${v:,.0f}" for v in interv_actions["net_value_sum"]],
                    textposition="auto",
                ),
            ]
        )
        fig_grouped.update_layout(
            barmode="group",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    with col_table:
        table_display = summary_df[[
            "action_code", "action_name", "n_customers", "percentage",
            "cost_sum", "benefit_sum", "net_value_sum"
        ]].copy()
        table_display.columns = [
            "Mã", "Tên Gói Hành Động", "Số Khách", "Tỷ Lệ %",
            "Tổng Chi Phí", "Lợi Ích Gộp", "Lợi Nhuận Ròng (ΔEV)"
        ]
        st.dataframe(
            table_display.style.format({
                "Số Khách": "{:,}",
                "Tỷ Lệ %": "{:.1f}%",
                "Tổng Chi Phí": "${:,.2f}",
                "Lợi Ích Gộp": "${:,.2f}",
                "Lợi Nhuận Ròng (ΔEV)": "${:,.2f}",
            }),
            use_container_width=True,
            height=280,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: HỒ SƠ KHÁCH HÀNG 360° & ĐỀ XUẤT CÁ NHÂN HÓA (CUSTOMER 360)
# ==============================================================================
with tab2:
    st.markdown("### 📋 Danh Sách Đề Xuất & Hồ Sơ Khách Hàng 360°")
    st.caption("Khám phá chi tiết từng khách hàng, thấu hiểu nguyên nhân rời bỏ và xem căn cứ kinh tế của gói can thiệp tối ưu.")

    # 1. BỘ LỌC ĐA CHIỀU (SMART FILTERS)
    with st.expander("🔍 Bộ Lọc Danh Sách Khách Hàng", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1, 1, 1])
        
        with f_col1:
            search_query = st.text_input("Tìm kiếm Mã Khách Hàng (customerID):", placeholder="Ví dụ: 7590-VHVEG...")
        with f_col2:
            action_filter = st.selectbox(
                "Gói can thiệp:",
                options=["Tất cả", "Chỉ can thiệp (A1, A2, A3)", "A0: Không can thiệp", "A1: Hỗ trợ kỹ thuật", "A2: Hợp đồng & TT", "A3: Giảm cước"],
                index=1,
            )
        with f_col3:
            risk_filter = st.selectbox(
                "Mức rủi ro Churn:",
                options=["Tất cả", "Rất cao (> 70%)", "Trung bình (40% - 70%)", "Thấp (< 40%)"],
                index=0,
            )
        with f_col4:
            contract_filter = st.selectbox(
                "Loại hợp đồng:",
                options=["Tất cả", "Month-to-month", "One year", "Two year"],
                index=0,
            )

    # Ghép thông tin hợp đồng vào recommendations để lọc
    rec_with_features = recommendations.merge(
        customers_df[["customerID", "Contract", "MonthlyCharges", "tenure", "InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod"]],
        on="customerID",
        how="left"
    )

    # Áp dụng bộ lọc
    filtered_df = rec_with_features.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["customerID"].str.contains(search_query.strip(), case=False)]
    if action_filter == "Chỉ can thiệp (A1, A2, A3)":
        filtered_df = filtered_df[filtered_df["action_code"] != "A0"]
    elif action_filter.startswith("A"):
        code_sel = action_filter.split(":")[0]
        filtered_df = filtered_df[filtered_df["action_code"] == code_sel]
    
    if risk_filter == "Rất cao (> 70%)":
        filtered_df = filtered_df[filtered_df["churn_prob"] > 0.70]
    elif risk_filter == "Trung bình (40% - 70%)":
        filtered_df = filtered_df[(filtered_df["churn_prob"] >= 0.40) & (filtered_df["churn_prob"] <= 0.70)]
    elif risk_filter == "Thấp (< 40%)":
        filtered_df = filtered_df[filtered_df["churn_prob"] < 0.40]

    if contract_filter != "Tất cả":
        filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]

    st.markdown(f"**Kết quả:** Tìm thấy **{len(filtered_df):,}** khách hàng thỏa mãn tiêu chí.")

    # Hiển thị bảng tổng hợp
    st.dataframe(
        filtered_df[[
            "rank", "customerID", "action_code", "action_name", "churn_prob",
            "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
        ]].style.format({
            "churn_prob": "{:.1%}",
            "churn_prob_after": "{:.1%}",
            "breakeven_prob": "{:.1%}",
            "proxy_value": "${:.2f}",
            "cost": "${:.2f}",
            "benefit": "${:.2f}",
            "net_value": "${:+.2f}",
        }),
        use_container_width=True,
        height=260,
    )

    # Nút Export dữ liệu CSV
    csv_data = filtered_df[[
        "rank", "customerID", "action_code", "action_name", "churn_prob",
        "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
    ]].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Tải Danh Sách Can Thiệp Này (.CSV)",
        data=csv_data,
        file_name="retention_recommendations_filtered.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # 2. KHU VỰC KHÁM PHÁ HỒ SƠ 360° CHO 1 KHÁCH HÀNG CỤ THỂ
    st.markdown("#### 👤 Khám Phá Chuyên Sâu Hồ Sơ Khách Hàng (Customer 360 Deep-Dive)")
    
    # Cho phép chọn khách hàng từ kết quả lọc
    if not filtered_df.empty:
        cust_list = filtered_df["customerID"].tolist()
        cust_options = [f"{cid} | Gói: {row.action_code} | ΔEV: ${row.net_value:+.2f} | Churn: {row.churn_prob:.1%}" for cid, row in zip(cust_list, filtered_df.itertuples())]
        
        selected_idx = st.selectbox(
            "👉 Chọn một khách hàng để xem phân tích chi tiết & căn cứ ra quyết định:",
            options=range(len(cust_list)),
            format_func=lambda i: cust_options[i],
            index=0,
        )
        selected_cust_id = cust_list[selected_idx]
    else:
        st.info("Không có khách hàng nào phù hợp với bộ lọc hiện tại.")
        selected_cust_id = None

    if selected_cust_id:
        cust_info = customers_df[customers_df["customerID"] == selected_cust_id].iloc[0]
        cust_rec = recommendations[recommendations["customerID"] == selected_cust_id].iloc[0]

        # Đánh giá tất cả các hành động hợp lệ của khách hàng này (tái sử dụng evaluate_actions)
        cust_eval = evaluate_actions(
            customers=customers_df[customers_df["customerID"] == selected_cust_id],
            predictions=predictions_df[predictions_df["customer_id"] == selected_cust_id],
            actions=current_actions,
            margin=base_margin,
            horizon_months=horizon,
        )

        col_c360_left, col_c360_mid, col_c360_right = st.columns([1, 1.3, 1.1])

        # CỘT 1: PROFILE & GAUGE RỦI RO
        with col_c360_left:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-header">📌 Profile: {selected_cust_id}</div>', unsafe_allow_html=True)
            
            # Gauge Chart Plotly đo Churn Risk
            churn_val = float(cust_rec["churn_prob"]) * 100
            churn_after_val = float(cust_rec["churn_prob_after"]) * 100
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_val,
                number={'suffix': "%", 'valueformat': ".1f"},
                delta={'reference': churn_after_val, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}, 'position': "bottom"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#ef4444" if churn_val > 65 else ("#f59e0b" if churn_val > 35 else "#10b981")},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 35], 'color': '#dcfce7'},
                        {'range': [35, 65], 'color': '#fef3c7'},
                        {'range': [65, 100], 'color': '#fee2e2'},
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 3},
                        'thickness': 0.75,
                        'value': churn_after_val
                    }
                }
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=15, r=15, t=25, b=15),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(text="Vạch xanh: Sau can thiệp", x=0.5, y=-0.15, showarrow=False, font=dict(size=11, color="#3b82f6"))
                ]
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Chi tiết thông tin dịch vụ
            st.markdown(
                f"""
                <div class="profile-box">
                    <div class="profile-item"><span class="profile-key">Loại Hợp Đồng:</span> <span class="profile-val">{cust_info['Contract']}</span></div>
                    <div class="profile-item"><span class="profile-key">Thời Gian Gắn Bó:</span> <span class="profile-val">{cust_info['tenure']} tháng</span></div>
                    <div class="profile-item"><span class="profile-key">Cước Phí Tháng:</span> <span class="profile-val">${cust_info['MonthlyCharges']:.2f} / tháng</span></div>
                    <div class="profile-item"><span class="profile-key">Dịch Vụ Internet:</span> <span class="profile-val">{cust_info['InternetService']}</span></div>
                    <div class="profile-item"><span class="profile-key">Hỗ Trợ Kỹ Thuật:</span> <span class="profile-val">{cust_info['TechSupport']}</span></div>
                    <div class="profile-item"><span class="profile-key">Bảo Mật Online:</span> <span class="profile-val">{cust_info['OnlineSecurity']}</span></div>
                    <div class="profile-item"><span class="profile-key">Thanh Toán:</span> <span class="profile-val">{cust_info['PaymentMethod']}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # CỘT 2: CĂN CỨ KINH TẾ & SO SÁNH CÁC PHƯƠNG ÁN HỢP LỆ
        with col_c360_mid:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">💡 Quyết Định Tối Ưu & Căn Cứ Kinh Tế</div>', unsafe_allow_html=True)

            best_code = cust_rec["action_code"]
            best_name = cust_rec["action_name"]
            best_ev = cust_rec["net_value"]
            proxy_val = cust_rec["proxy_value"]
            breakeven = cust_rec["breakeven_prob"]

            if best_code != "A0":
                st.markdown(
                    f"""
                    <div class="recommend-box">
                        <div class="recommend-title">
                            <span>⭐</span> Đề Xuất: Gói {best_code} — {best_name}
                        </div>
                        <div class="recommend-sub">
                            Lợi nhuận ròng tối đa: <strong>+${best_ev:.2f}</strong> | Ngưỡng hòa vốn p*: <strong>{breakeven:.1%}</strong> (Rủi ro hiện tại {churn_val:.1%} > p* nên can thiệp có lãi).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="recommend-box" style="background:#f1f5f9; border-color:#cbd5e1;">
                        <div class="recommend-title" style="color:#475569;">
                            <span>⚪</span> Đề Xuất: Gói A0 — Không Can Thiệp
                        </div>
                        <div class="recommend-sub" style="color:#64748b;">
                            Các gói can thiệp có chi phí vượt quá lợi ích kỳ vọng (ΔEV ≤ 0) hoặc khách hàng không đủ điều kiện.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("##### So Sánh Giá Trị Ròng (ΔEV) Từng Phương Án Hợp Lệ:")
            
            # Biểu đồ cột so sánh ΔEV các action của riêng khách hàng này
            fig_action_comp = go.Figure()
            colors_comp = ["#10b981" if code == best_code and best_code != "A0" else ("#94a3b8" if code == "A0" else "#3b82f6") for code in cust_eval["action_code"]]
            
            fig_action_comp.add_trace(go.Bar(
                x=cust_eval["action_code"],
                y=cust_eval["net_value"],
                text=[f"${v:+.2f}" for v in cust_eval["net_value"]],
                textposition="auto",
                marker_color=colors_comp,
                hovertemplate="<b>%{x}</b><br>Giá trị ròng (ΔEV): %{y:$.2f}<extra></extra>",
            ))
            fig_action_comp.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Lợi nhuận ròng ΔEV ($)", gridcolor="#f1f5f9"),
                xaxis=dict(title="Mã hành động"),
            )
            st.plotly_chart(fig_action_comp, use_container_width=True)

            # Bảng con so sánh các phương án
            cust_eval_display = cust_eval[[
                "action_code", "action_name", "cost", "benefit", "net_value", "breakeven_prob"
            ]].rename(columns={
                "action_code": "Mã",
                "action_name": "Tên gói",
                "cost": "Chi phí C",
                "benefit": "Lợi ích B",
                "net_value": "Giá trị ròng ΔEV",
                "breakeven_prob": "Hòa vốn p*",
            })
            st.dataframe(
                cust_eval_display.style.format({
                    "Chi phí C": "${:.2f}",
                    "Lợi ích B": "${:.2f}",
                    "Giá trị ròng ΔEV": "${:+.2f}",
                    "Hòa vốn p*": "{:.1%}",
                }),
                use_container_width=True,
                height=160,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # CỘT 3: YẾU TỐ TRẢI NGHIỆM GÂY CHURN & KỊCH BẢN CSKH
        with col_c360_right:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">🔍 Yếu Tố Trải Nghiệm Gây Rời Bỏ</div>', unsafe_allow_html=True)

            # Đánh giá các drivers dựa trên dữ liệu thực tế của khách hàng (Cầu nối logic SHAP)
            driver_factors = []
            if cust_info["Contract"] == "Month-to-month":
                driver_factors.append(("Hợp đồng từng tháng (Month-to-month)", 0.38, "Nguy cơ cao: Thiếu tính ràng buộc dài hạn"))
            if cust_info["MonthlyCharges"] >= 65:
                driver_factors.append((f"Cước phí cao (${cust_info['MonthlyCharges']:.0f}/th)", 0.28, "Nguy cơ: Nhạy cảm về giá"))
            if cust_info["TechSupport"] == "No":
                driver_factors.append(("Thiếu dịch vụ Tech Support", 0.22, "Nguy cơ: Khó khăn khi gặp sự cố kỹ thuật"))
            if cust_info["PaymentMethod"] in MANUAL_PAYMENTS:
                driver_factors.append(("Thanh toán thủ công (Check)", 0.18, "Nguy cơ: Trải nghiệm thanh toán phiền hà"))
            if cust_info["tenure"] <= 6:
                driver_factors.append(("Khách hàng mới (< 6 tháng)", 0.15, "Nguy cơ: Chưa hình thành thói quen gắn bó"))

            if not driver_factors:
                driver_factors.append(("Khách hàng có độ ổn định tốt", 0.05, "Tích cực"))

            df_drivers = pd.DataFrame(driver_factors, columns=["factor", "impact", "desc"])
            
            fig_drivers = px.bar(
                df_drivers,
                x="impact",
                y="factor",
                orientation="h",
                text=[f"+{v:.2f}" for v in df_drivers["impact"]],
                labels={"impact": "Mức độ tác động", "factor": ""},
                color_discrete_sequence=["#f43f5e"]
            )
            fig_drivers.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_drivers, use_container_width=True)

            st.markdown("##### 💬 Kịch Bản Tư Vấn Cho Nhân Viên CSKH:")
            if best_code == "A3":
                script_text = f"Chào anh/chị {selected_cust_id}, cảm ơn anh/chị đã đồng hành cùng nhà mạng. Để tri ân khách hàng thân thiết, chúng em xin gửi tặng voucher giảm giá cước và gói cước ưu đãi đặc biệt khi anh/chị nâng cấp hợp đồng 1 năm..."
            elif best_code == "A1":
                script_text = f"Chào anh/chị {selected_cust_id}, kỹ thuật viên của nhà mạng liên hệ hỗ trợ kiểm tra định kỳ đường truyền Fiber và kích hoạt miễn phí gói bảo mật Online Security cho gia đình mình..."
            elif best_code == "A2":
                script_text = f"Chào anh/chị {selected_cust_id}, nhà mạng xin giới thiệu tính năng thanh toán tự động tiện lợi kèm ưu đãi hoàn tiền 10% cước phí kỳ đầu tiên..."
            else:
                script_text = "Khách hàng có nguy cơ rời mạng thấp hoặc phương án can thiệp không mang lại hiệu quả kinh tế. Không cần chủ động gọi điện ưu đãi."

            st.info(script_text)

            # Nút hành động tương tác
            if best_code != "A0":
                if st.button("🚀 Kích Hoạt Gói Can Thiệp Cho Khách Hàng Này", key="btn_apply", use_container_width=True):
                    st.success(f"✅ Đã ghi nhận áp dụng gói {best_code} cho khách hàng {selected_cust_id}. Chi phí dự kiến: ${cust_rec['cost']:.2f}, Lợi nhuận ròng kỳ vọng: +${best_ev:.2f}!")

            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: PHÂN TÍCH ĐỘ NHẠY & MÔ PHỎNG NGÂN SÁCH (SENSITIVITY & BUDGET SIMULATION)
# ==============================================================================
with tab3:
    st.markdown("### 📈 Phân Tích Độ Nhạy & Mô Phỏng Giới Hạn Ngân Sách")
    st.caption("Khảo sát sự thay đổi của lợi nhuận ròng khi biên lợi nhuận (m) thay đổi, và mô phỏng tối ưu hóa danh mục can thiệp khi bị giới hạn ngân sách.")

    # 1. PHÂN TÍCH ĐỘ NHẠY THEO MARGIN (m)
    sens_df = sensitivity_by_margin(
        customers=customers_df,
        predictions=predictions_df,
        margins=(0.20, 0.30, 0.40, 0.50),
        actions=current_actions,
        horizon_months=horizon,
    )

    st.markdown("#### 1. Độ Nhạy Khi Biên Lợi Nhuận Giả Định (m) Biến Động")
    
    sens_col1, sens_col2 = st.columns([1.2, 1])

    with sens_col1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">📊 Xu Hướng Dòng Tiền (ΔEV, Chi Phí, Lợi Ích)</div>', unsafe_allow_html=True)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=[f"{m:.0%}" for m in sens_df["margin"]],
            y=sens_df["total_net_value"],
            name="Lợi Nhuận Ròng (ΔEV)",
            line=dict(color="#10b981", width=3),
            mode="lines+markers+text",
            text=[f"${v:,.0f}" for v in sens_df["total_net_value"]],
            textposition="top center",
        ))
        fig_sens.add_trace(go.Scatter(
            x=[f"{m:.0%}" for m in sens_df["margin"]],
            y=sens_df["total_cost"],
            name="Tổng Chi Phí",
            line=dict(color="#f97316", width=2, dash="dash"),
            mode="lines+markers",
        ))
        fig_sens.add_trace(go.Scatter(
            x=[f"{m:.0%}" for m in sens_df["margin"]],
            y=sens_df["total_benefit"],
            name="Lợi Ích Gộp",
            line=dict(color="#3b82f6", width=2),
            mode="lines+markers",
        ))
        fig_sens.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Giá trị ($)", gridcolor="#f1f5f9"),
            xaxis=dict(title="Biên Lợi Nhuận (m)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sens_col2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">📦 Cơ Cấu Gói Can Thiệp Khi Margin Tăng</div>', unsafe_allow_html=True)
        
        # Biểu đồ Stacked Bar số lượng A1, A2, A3 theo từng margin
        action_cols = [c for c in sens_df.columns if c.startswith("n_A")]
        fig_stacked = go.Figure()
        colors_dict = {"n_A1": "#3b82f6", "n_A2": "#06b6d4", "n_A3": "#10b981"}
        names_dict = {"n_A1": "A1 (Kỹ thuật)", "n_A2": "A2 (Hợp đồng)", "n_A3": "A3 (Giảm cước)"}

        for ac in action_cols:
            fig_stacked.add_trace(go.Bar(
                name=names_dict.get(ac, ac),
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df[ac],
                marker_color=colors_dict.get(ac, "#6366f1"),
                text=sens_df[ac],
                textposition="auto",
            ))

        fig_stacked.update_layout(
            barmode="stack",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Số lượng khách can thiệp", gridcolor="#f1f5f9"),
            xaxis=dict(title="Biên Lợi Nhuận (m)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bảng chi tiết độ nhạy
    sens_df_display = sens_df.rename(columns={
        "margin": "Biên Lợi Nhuận (m)",
        "n_customers": "Tổng Khách Đủ ĐK",
        "n_intervened": "Số Khách Can Thiệp",
        "total_cost": "Tổng Chi Phí",
        "total_benefit": "Lợi Ích Gộp",
        "total_net_value": "Lợi Nhuận Ròng (ΔEV)",
    })
    st.dataframe(
        sens_df_display.style.format({
            "Biên Lợi Nhuận (m)": "{:.0%}",
            "Tổng Khách Đủ ĐK": "{:,}",
            "Số Khách Can Thiệp": "{:,}",
            "Tổng Chi Phí": "${:,.2f}",
            "Lợi Ích Gộp": "${:,.2f}",
            "Lợi Nhuận Ròng (ΔEV)": "${:,.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # 2. MÔ PHỎNG GIỚI HẠN NGÂN SÁCH (BUDGET CONSTRAINT SANDBOX)
    st.markdown("#### 2. 🎛️ Mô Phỏng Tối Ưu Khi Giới Hạn Ngân Sách (Budget Constraint Sandbox)")
    st.caption("Trong thực tế, doanh nghiệp không thể giải ngân không giới hạn. Công cụ này cho phép bạn thử nghiệm hạn mức ngân sách và tự động lọc danh sách khách hàng mang lại ROI cao nhất.")

    current_total_cost = float(interventions["cost"].sum())
    
    sim_col_left, sim_col_right = st.columns([1, 1.4])

    with sim_col_left:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">⚙️ Thiết Lập Hạn Mức Ngân Sách</div>', unsafe_allow_html=True)

        budget_cap = st.slider(
            "Hạn mức ngân sách tối đa ($):",
            min_value=2000,
            max_value=max(int(current_total_cost) + 5000, 40000),
            value=min(15000, int(current_total_cost)),
            step=1000,
            help="Hệ thống sẽ ưu tiên khách hàng có ROI = ΔEV / Cost cao nhất cho đến khi hết ngân sách này.",
        )

        # Thuật toán phân bổ ngân sách: Sắp xếp theo ROI = ΔEV / Cost
        interv_sim = interventions.copy()
        interv_sim["roi_ratio"] = interv_sim["net_value"] / interv_sim["cost"]
        interv_sim = interv_sim.sort_values(by=["roi_ratio", "net_value"], ascending=[False, False]).reset_index(drop=True)
        interv_sim["cum_cost"] = interv_sim["cost"].cumsum()
        
        approved_sim = interv_sim[interv_sim["cum_cost"] <= budget_cap]
        
        n_approved = len(approved_sim)
        actual_spent = float(approved_sim["cost"].sum()) if not approved_sim.empty else 0
        actual_earned = float(approved_sim["net_value"].sum()) if not approved_sim.empty else 0
        budget_utilization = (actual_spent / budget_cap * 100) if budget_cap > 0 else 0
        sim_roi = (actual_earned / actual_spent * 100) if actual_spent > 0 else 0
        retention_efficiency = (actual_earned / total_net_value * 100) if total_net_value > 0 else 0

        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-top: 10px;">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; margin-bottom: 4px;">TỔNG KẾT PHÂN BỔ:</div>
                <div style="font-size: 0.92rem; color: #0f172a; padding: 4px 0;">• Khách hàng được duyệt: <strong>{n_approved:,} / {len(interventions):,}</strong> ({n_approved/len(interventions):.1%})</div>
                <div style="font-size: 0.92rem; color: #0f172a; padding: 4px 0;">• Ngân sách thực tiêu: <strong>${actual_spent:,.2f}</strong> / ${budget_cap:,} ({budget_utilization:.1%})</div>
                <div style="font-size: 0.92rem; color: #059669; padding: 4px 0; font-weight: 700;">• Lợi nhuận ròng thu về: +${actual_earned:,.2f}</div>
                <div style="font-size: 0.92rem; color: #4338ca; padding: 4px 0;">• Đạt: <strong>{retention_efficiency:.1%}</strong> hiệu quả kinh tế so với ngân sách vô hạn.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with sim_col_right:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">📊 Phân Bổ Ngân Sách Theo Gói Can Thiệp Trong Hạn Mức</div>', unsafe_allow_html=True)

        if not approved_sim.empty:
            sim_action_breakdown = approved_sim.groupby("action_code").agg(
                n_cust=("customerID", "count"),
                cost_sum=("cost", "sum"),
                net_sum=("net_value", "sum"),
            ).reset_index()

            fig_sim = go.Figure(data=[
                go.Bar(
                    name="Ngân Sách Tiêu ($)",
                    x=sim_action_breakdown["action_code"],
                    y=sim_action_breakdown["cost_sum"],
                    marker_color="#f97316",
                    text=[f"${v:,.0f}" for v in sim_action_breakdown["cost_sum"]],
                    textposition="auto",
                ),
                go.Bar(
                    name="Lợi Nhuận Ròng Thu Về ($)",
                    x=sim_action_breakdown["action_code"],
                    y=sim_action_breakdown["net_sum"],
                    marker_color="#10b981",
                    text=[f"${v:,.0f}" for v in sim_action_breakdown["net_sum"]],
                    textposition="auto",
                ),
            ])
            fig_sim.update_layout(
                barmode="group",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Giá trị ($)", gridcolor="#f1f5f9"),
                xaxis=dict(title="Gói can thiệp"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Hạn mức ngân sách quá nhỏ để thực hiện bất kỳ hành động can thiệp nào.")

        st.markdown('</div>', unsafe_allow_html=True)