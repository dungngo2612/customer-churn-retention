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
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.datasets.validate import validate_dataset
from churn_retention.ml.explainability.shap_explainer import (
    get_global_feature_importances,
    explain_single_customer_shap,
)
from churn_retention.modules.campaigns import (
    Campaign,
    CampaignStatus,
    create_campaign,
    get_campaign,
    list_campaigns,
    export_campaign_csv,
    export_campaign_dataframe,
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

    /* Hero Banner chuẩn Bento Executive */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #2e1065 100%);
        border-radius: 18px;
        padding: 26px 32px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -8px rgba(15, 23, 42, 0.25), 0 4px 6px -4px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.35) 0%, rgba(99, 102, 241, 0) 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-desc {
        color: #cbd5e1;
        font-size: 0.96rem;
        max-width: 880px;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .pill-group {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
    }
    .status-pill {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8fafc;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* KPI Grid chuẩn Bento Card */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-bottom: 26px;
    }
    .modern-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 4px 6px -2px rgba(15, 23, 42, 0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .modern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08), 0 4px 8px -2px rgba(15, 23, 42, 0.03);
        border-color: #cbd5e1;
    }
    .modern-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: transparent;
        transition: background 0.2s ease;
    }
    .card-accent-blue::before { background: #3b82f6; }
    .card-accent-indigo::before { background: #6366f1; }
    .card-accent-amber::before { background: #f59e0b; }
    .card-accent-emerald::before { background: #10b981; }

    .card-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-icon-wrap {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
    }
    .card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
        font-feature-settings: "tnum" on, "lnum" on;
        white-space: nowrap;
    }
    .card-footer {
        font-size: 0.84rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges ngữ nghĩa */
    .badge-emerald {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-indigo {
        background: #eef2ff;
        color: #4f46e5;
        border: 1px solid #c7d2fe;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-rose {
        background: #fff1f2;
        color: #e11d48;
        border: 1px solid #fecdd3;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-amber {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    /* Container bao bọc các biểu đồ chuẩn Bento Panel */
    .chart-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        margin-bottom: 22px;
        transition: border-color 0.2s ease;
    }
    .chart-panel:hover {
        border-color: #cbd5e1;
    }
    .chart-header {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }

    /* Hồ sơ Customer 360 styling */
    .profile-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .profile-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
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

    /* Hộp Đề Xuất Tối Ưu chuẩn Executive Decision Ticket */
    .recommend-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -2px rgba(22, 101, 52, 0.05);
    }
    .recommend-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #15803d;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .recommend-sub {
        font-size: 0.9rem;
        color: #166534;
        line-height: 1.5;
    }

    /* Tùy biến thanh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Tùy biến Tabs dạng Segmented Control */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 9px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748b;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #4338ca !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- NẠP DỮ LIỆU & CHẠY DỰ ĐOÁN (KHÔNG DÙNG DỮ LIỆU CỨNG, BẮT LỖI RÕ NẾU THIẾU BUNDLE) ---
def get_bundle_path():
    bundle_dirs = [
        ROOT_DIR / "artifacts" / "bundles",
        ROOT_DIR / "artifacts" / "bundle",
    ]
    for b_dir in bundle_dirs:
        if b_dir.exists():
            for p in b_dir.glob("**/model.pkl"):
                return p.parent
    return None


@st.cache_data
def load_data_and_predict(is_demo_mode: bool = False):
    possible_paths = [
        ROOT_DIR / "data" / "raw" / "telco_churn.csv",
        ROOT_DIR / "data" / "telco_churn.csv",
        ROOT_DIR / "telco_churn.csv",
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)

    if data_path is None:
        if not is_demo_mode:
            st.error("❌ Không tìm thấy file dữ liệu thật `telco_churn.csv`! Vui lòng tải dữ liệu vào data/ hoặc bật 'Chế độ Demo'.")
            st.stop()
        else:
            customers = pd.DataFrame({
                "customerID": ["DEMO_01", "DEMO_02", "DEMO_03"],
                "MonthlyCharges": [85.0, 25.0, 65.0],
                "Contract": ["Month-to-month", "Two year", "Month-to-month"],
                "InternetService": ["Fiber optic", "DSL", "DSL"],
                "TechSupport": ["No", "Yes", "No"],
                "OnlineSecurity": ["No", "Yes", "No"],
                "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Electronic check"],
                "Churn": ["No", "No", "No"],
                "tenure": [6, 48, 12],
                "TotalCharges": [510.0, 1200.0, 780.0],
            })
            dataset_name = "demo_sample.csv"
    else:
        customers = pd.read_csv(data_path)
        dataset_name = data_path.name
        if "TotalCharges" in customers.columns:
            customers["TotalCharges"] = pd.to_numeric(
                customers["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)

    bundle_path = get_bundle_path()
    model_version = "fake-v1"
    predictor_info = "FakePredictor (Mô phỏng)"
    probs = None

    if not is_demo_mode:
        if bundle_path is None or not bundle_path.exists():
            st.error(
                "❌ Không tìm thấy model bundle thật trong `artifacts/bundles/`! "
                "Luồng chạy thật bắt buộc phải có bundle CatBoost. "
                "Vui lòng kiểm tra model bundle hoặc bật 'Chế độ Demo' ở thanh bên nếu muốn thử nghiệm."
            )
            st.stop()

        try:
            from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
            bp = BundlePredictor()
            bp.load(str(bundle_path))
            probs = bp.predict_batch(
                features=customers.drop(columns=[c for c in ("customerID", "Churn") if c in customers.columns]),
                customer_ids=customers["customerID"].tolist()
            )
            model_version = bp.model_version
            predictor_info = f"BundlePredictor (CatBoost - {model_version})"
        except Exception as exc:
            st.error(f"❌ Lỗi khi nạp hoặc chạy model bundle: {exc}")
            st.stop()
    else:
        fake_pred = FakePredictor(seed=42)
        probs = fake_pred.predict_batch(
            features=customers,
            customer_ids=customers["customerID"].tolist()
        )
        model_version = fake_pred.model_version
        predictor_info = f"FakePredictor (Chế độ Demo - {model_version})"

    predictions = pd.DataFrame({
        "customer_id": customers["customerID"].tolist(),
        "churn_prob": probs,
    })

    return customers, predictions, predictor_info, dataset_name, model_version


# Quản lý chế độ Demo qua session_state
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False

customers_df, predictions_df, predictor_info, active_dataset_name, active_model_version = load_data_and_predict(
    is_demo_mode=st.session_state["demo_mode"]
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

    st.markdown("---")
    st.markdown("### 🤖 Nguồn Dự Đoán Churn")
    st.caption(f"Trạng thái: **{predictor_info}**")

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

# --- TABS ĐIỀU HƯỚNG (6 TẦNG HỆ THỐNG) ---
tab_e2e, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Luồng Trọn Vẹn (E2E Pipeline)",
    "📊 Tổng quan Kinh tế (Overview)",
    "🔍 Khám Phá Dữ Liệu & Hành Vi Churn (EDA)",
    "👤 Hồ Sơ Khách Hàng 360° & Đề Xuất Can Thiệp",
    "🎛️ Phân Tích Độ Nhạy & Mô Phỏng Ngân Sách",
    "📋 Quản Lý Chiến Dịch (Campaigns)",
])

# ==============================================================================
# TAB E2E: LUỒNG THAO TÁC TRỌN VẸN (TASK 3: GHÉP LUỒNG DỮ LIỆU THẬT)
# ==============================================================================
with tab_e2e:
    st.markdown("### 🚀 Luồng Ứng Dụng Khép Kín Bằng Dữ Liệu Thật (End-to-End Pipeline)")
    st.caption("Trải nghiệm đầy đủ chuỗi giá trị: **Upload Dataset ➔ Scoring Job ➔ Phân Tích Xác Suất & SHAP ➔ Nhập Ngân Sách ➔ Mô Phỏng ➔ Lưu & Xuất Chiến Dịch** mà không cần sửa file thủ công.")

    # Status Pills
    bundle_ready = "CatBoost" in predictor_info
    model_status_badge = '<span class="status-pill status-ready">● Model Sẵn Sàng (CatBoost Bundle)</span>' if bundle_ready else '<span class="status-pill" style="color:#f59e0b; border-color:#f59e0b;">● Chế độ Demo</span>'
    st.markdown(
        f"""
        <div style="background: #0f172a; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 20px; margin-bottom: 20px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
            <div><span style="color:#94a3b8; font-size:0.8rem;">DATASET HIỆN HÀNH:</span><br><strong style="color:#fff;">{active_dataset_name} ({len(customers_df):,} khách)</strong></div>
            <div><span style="color:#94a3b8; font-size:0.8rem;">MODEL VERSION:</span><br><strong style="color:#fff;">{active_model_version}</strong></div>
            <div><span style="color:#94a3b8; font-size:0.8rem;">TRẠNG THÁI SCORING:</span><br>{model_status_badge}</div>
            <div><span style="color:#94a3b8; font-size:0.8rem;">QUY ƯỚC KINH TẾ:</span><br><span style="background:rgba(99,102,241,0.2); color:#a5b4fc; border:1px solid #6366f1; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700;">Giả Định Kinh Tế (m={base_margin:.0%}, H={horizon}m)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 BƯỚC THAO TÁC LIÊN HOÀN
    step1_exp, step2_exp, step3_exp, step4_exp, step5_exp = st.tabs([
        "1️⃣ Upload & Kiểm Tra Dataset",
        "2️⃣ Chạy Scoring Job (Inference)",
        "3️⃣ Xem Xác Suất & TreeSHAP",
        "4️⃣ Nhập Ngân Sách & Mô Phỏng",
        "5️⃣ Lưu & Xuất Chiến Dịch",
    ])

    with step1_exp:
        st.markdown("#### 1️⃣ Quản Lý & Tải Lên Tập Dữ Liệu Khách Hàng")
        st.caption("Kiểm tra toàn diện dữ liệu theo hợp đồng schema trước khi đưa vào mô hình học máy.")

        uploaded_file = st.file_uploader(
            "Tải lên file CSV khách hàng mới (hoặc dùng mặc định `telco_churn.csv`):",
            type=["csv"],
            key="e2e_file_uploader",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                val_report = validate_dataset(uploaded_df)
                if val_report.is_valid:
                    st.success(f"✅ File `{uploaded_file.name}` hợp lệ 100%! Tổng số: **{val_report.total_rows:,}** dòng dữ liệu.")
                    st.dataframe(uploaded_df.head(5), width="stretch")
                else:
                    st.error(f"❌ File `{uploaded_file.name}` có {len(val_report.errors)} lỗi dữ liệu:")
                    err_df = pd.DataFrame([{"Dòng": e.row or "Toàn bảng", "Cột": e.column, "Chi tiết lỗi": e.reason} for e in val_report.errors])
                    st.dataframe(err_df, width="stretch")
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")
        else:
            st.info(f"📂 Đang dùng tập dữ liệu mặc định: **`{active_dataset_name}`** ({len(customers_df):,} bản ghi).")
            try:
                _default_val = validate_dataset(customers_df)
                if _default_val.is_valid:
                    st.success(f"✅ Kiểm tra schema: {_default_val.total_rows:,} dòng, 0 lỗi (ô `TotalCharges` trống đã được chuẩn hoá về 0).")
                else:
                    st.warning(f"⚠️ Kiểm tra schema phát hiện {len(_default_val.errors)} vấn đề trong dữ liệu mặc định.")
            except Exception as _exc:
                st.warning(f"Không thể chạy kiểm tra schema cho dữ liệu mặc định: {_exc}")
            st.dataframe(customers_df.head(4), width="stretch")

    with step2_exp:
        st.markdown("#### 2️⃣ Chạy Scoring Job Chấm Điểm Rời Mạng")
        st.caption("Áp dụng mô hình CatBoost đã đóng gói (Bundle) để dự báo xác suất rời mạng cho từng khách hàng.")

        sc_col1, sc_col2 = st.columns([1.5, 1])
        with sc_col1:
            st.markdown(f"• **Mô hình triển khai:** `{active_model_version}`")
            st.markdown(f"• **Tập khách hàng đầu vào:** `{active_dataset_name}` ({len(customers_df):,} khách)")
            st.markdown(f"• **Cơ chế Inference:** `{predictor_info}`")

            if st.button("🚀 Chạy Scoring Job Trên Toàn Bộ Dữ Liệu", type="primary", key="btn_run_scoring", disabled=not bundle_ready):
                from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
                from churn_retention.contracts.jobs.models import JobType, TaskJob
                from churn_retention.entrypoints.worker.worker import JobWorker

                _ds_path = next(
                    (p for p in [
                        ROOT_DIR / "data" / "raw" / "telco_churn.csv",
                        ROOT_DIR / "data" / "telco_churn.csv",
                        ROOT_DIR / "telco_churn.csv",
                    ] if p.exists()),
                    None,
                )
                _run_id = f"run-batch-{active_model_version[:10]}"
                _idem_key = f"scoring:{active_dataset_name}:{active_model_version}"
                _queue = SqliteTaskQueue()
                _existing_job = _queue.get_by_idempotency_key(_idem_key)
                if _existing_job is not None:
                    st.info(
                        f"♻️ Job chấm điểm này đã tồn tại (trạng thái **{_existing_job.status.value}**). "
                        "Khóa chống trùng (idempotency key) chặn việc chạy và ghi trùng kết quả."
                    )
                else:
                    with st.spinner("Đang đưa job vào hàng đợi và xử lý bằng CatBoost..."):
                        _job = TaskJob.create(
                            job_type=JobType.SCORING,
                            idempotency_key=_idem_key,
                            payload={
                                "dataset_path": str(_ds_path) if _ds_path else None,
                                "bundle_path": str(get_bundle_path()),
                                "scoring_run_id": _run_id,
                            },
                        )
                        _queue.enqueue(_job)
                        _done = JobWorker(_queue).process_one()
                    if _done is not None and _done.status.value == "SUCCESS":
                        st.success(
                            f"✅ Scoring Job hoàn tất: đã lưu {_done.result.get('total_stored_in_db', 0):,} dự đoán "
                            f"(mới: {_done.result.get('inserted_new', 0):,})."
                        )
                    elif _done is not None:
                        st.error(f"❌ Scoring Job {_done.status.value}: {_done.error_reason}")
                st.session_state["scoring_run_id"] = _run_id

        with sc_col2:
            st.markdown(
                f"""
                <div style="background:rgba(148,163,184,0.12); border:1px solid rgba(148,163,184,0.35); border-radius:10px; padding:12px;">
                    <div style="font-size:0.8rem; color:#64748b; font-weight:700;">CHẤT LƯỢNG MÔ HÌNH:</div>
                    <div style="font-size:1.1rem; color:inherit; font-weight:800; margin-top:4px;">ROC-AUC: 84.8%</div>
                    <div style="font-size:0.85rem; color:#059669; font-weight:600;">PR-AUC (AP): 66.5%</div>
                    <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Đo trên tập kiểm thử khi huấn luyện (seed 42)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### 📋 Mẫu Xác Suất Dự Báo (Probabilities Batch):")
        st.dataframe(predictions_df.head(6), width="stretch")

    with step3_exp:
        st.markdown("#### 3️⃣ Phân Phối Xác Suất Rời Mạng & Giải Thích TreeSHAP")
        st.caption("Minh bạch hóa mô hình hộp đen: Nhận diện các yếu tố thúc đẩy khách hàng rời bỏ theo thuật toán SHAP.")

        shap_col1, shap_col2 = st.columns([1.2, 1])

        # Trích xuất Global SHAP Importance nếu dùng model bundle
        bp_bundle_path = get_bundle_path()
        df_importances = pd.DataFrame()
        test_bp = None
        if bp_bundle_path and bp_bundle_path.exists():
            try:
                from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
                test_bp = BundlePredictor()
                test_bp.load(str(bp_bundle_path))
                df_importances = get_global_feature_importances(test_bp)
            except Exception:
                pass

        with shap_col1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">🌟 Top 10 Đặc Trưng Quyết Định Rủi Ro Churn (Global SHAP Importance)</div>', unsafe_allow_html=True)
            if not df_importances.empty:
                top10_imp = df_importances.head(10).sort_values(by="importance", ascending=True)
                fig_shap_bar = go.Figure(go.Bar(
                    x=top10_imp["importance"],
                    y=top10_imp["feature"],
                    orientation="h",
                    marker_color="#6366f1",
                    text=[f"{v:.1f}%" for v in top10_imp["importance"]],
                    textposition="auto",
                ))
                fig_shap_bar.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Tầm Quan Trọng Tương Đối (%)", gridcolor="rgba(148,163,184,0.25)"),
                )
                st.plotly_chart(fig_shap_bar, width="stretch")
            else:
                st.info("Biểu đồ SHAP yêu cầu Model Bundle CatBoost thật để phân tích trọng số.")
            st.markdown('</div>', unsafe_allow_html=True)

        with shap_col2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">🔍 Giải Thích SHAP Cho Từng Khách Hàng (Local TreeSHAP)</div>', unsafe_allow_html=True)
            sample_cid = st.selectbox("Chọn khách hàng để soi nguyên nhân rủi ro:", options=customers_df["customerID"].head(10).tolist(), key="e2e_shap_cid")
            selected_row = customers_df[customers_df["customerID"] == sample_cid]
            selected_prob = float(predictions_df.loc[predictions_df["customer_id"] == sample_cid, "churn_prob"].values[0]) if not predictions_df.empty else 0.5
            st.metric("Xác suất rời mạng (p):", f"{selected_prob:.1%}")

            if test_bp is not None:
                try:
                    local_shap_df = explain_single_customer_shap(test_bp, selected_row)
                    if local_shap_df is not None and not local_shap_df.empty:
                        st.dataframe(
                            local_shap_df.head(6)[["feature", "value", "impact", "shap_value"]].style.format({"shap_value": "{:+.3f}"}),
                            width="stretch",
                            height=220,
                        )
                except Exception:
                    st.info("Chưa thể tính Local SHAP cho khách hàng này.")
            st.markdown('</div>', unsafe_allow_html=True)

    with step4_exp:
        st.markdown("#### 4️⃣ Nhập Ngân Sách & Mô Phỏng Can Thiệp")
        st.caption("Thuật toán Tham lam tỷ số sinh lời chi phí phân bổ tối ưu từng USD ngân sách.")
        st.info(
            "ℹ️ Gói được chọn ở đây có thể **khác** gói ΔEV cao nhất ở tab *Hồ sơ Khách hàng 360°*. "
            "Khi ngân sách bị giới hạn, thuật toán ưu tiên gói có **ΔEV trên mỗi USD** cao nhất (thường là gói rẻ hơn như A2) "
            "để phục vụ được nhiều khách hơn. Đây là heuristic tham lam (xấp xỉ), không đảm bảo tối ưu tuyệt đối."
        )

        st.markdown(
            f"""
            <div style="background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size:0.86rem; color:#312e81;">
                <strong>🏷️ NHÃN GIẢ ĐỊNH KINH TẾ (Economic Assumptions):</strong>
                Biên lợi nhuận $m = {base_margin:.0%}$, Kỳ hạn $H = {horizon}$ tháng, Giá trị thay thế $V = \text{{MonthlyCharges}} \times H \times m$.
                Chi phí can thiệp $C = c_{{\text{{contact}}}} + u \times c_{{\text{{offer}}}}$, mức giảm rủi ro $r = u \times e$.
                Khách chỉ được can thiệp khi lợi ích ròng $\\Delta EV = B - C > 0$.
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2 = st.columns([1, 1.5])
        with b_col1:
            e2e_budget = st.slider("Hạn mức ngân sách B (USD):", min_value=1000, max_value=50000, value=15000, step=1000, key="e2e_budget_slider")
            alloc_e2e = select_actions_under_budget(
                customers=customers_df,
                predictions=predictions_df,
                budget=e2e_budget,
                actions=current_actions,
                margin=base_margin,
                horizon_months=horizon,
            )
            interv_e2e = alloc_e2e[alloc_e2e["action_code"] != "A0"]
            spent_e2e = float(interv_e2e["cost"].sum()) if not interv_e2e.empty else 0.0
            earned_e2e = float(interv_e2e["net_value"].sum()) if not interv_e2e.empty else 0.0
            benefit_e2e = float(interv_e2e["benefit"].sum()) if not interv_e2e.empty else 0.0

            st.metric("Ngân sách thực tiêu:", f"${spent_e2e:,.2f} / ${e2e_budget:,}")
            st.metric("Lợi nhuận ròng thu về (ΔEV):", f"+${earned_e2e:,.2f}")
            st.metric("Số khách được can thiệp:", f"{len(interv_e2e):,} khách")

        with b_col2:
            st.markdown("##### 👥 Danh Sách Phân Bổ Tối Ưu Hàng Đầu (Xếp theo Tỷ Số Sinh Lời ΔEV / C):")
            st.dataframe(
                interv_e2e[["customerID", "action_code", "action_name", "churn_prob", "cost", "benefit", "net_value", "efficiency_ratio"]].head(8).style.format({
                    "churn_prob": "{:.1%}",
                    "cost": "${:,.2f}",
                    "benefit": "${:,.2f}",
                    "net_value": "${:,.2f}",
                    "efficiency_ratio": "{:.2f}",
                }),
                width="stretch",
                height=260,
            )

    with step5_exp:
        st.markdown("#### 5️⃣ Lưu & Xuất Chiến Dịch Mô Phỏng")
        st.caption("Đóng băng kết quả vào cơ sở dữ liệu bất biến và tải về danh sách thực thi CSV.")

        camp_save_col1, camp_save_col2 = st.columns(2)
        with camp_save_col1:
            e2e_camp_name = st.text_input("Tên chiến dịch mô phỏng:", value=f"Chiến dịch ngân sách ${e2e_budget:,} - {pd.Timestamp.now().strftime('%d/%m/%Y')}", key="e2e_camp_name_input")
        with camp_save_col2:
            e2e_camp_ds_ref = st.text_input("Dataset Reference (Linh hoạt):", value=active_dataset_name, key="e2e_camp_ds_ref")

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("💾 Lưu Chiến Dịch Bất Biến Vào DB", type="primary", key="btn_save_e2e_camp"):
                camp_results = {
                    "summary": {
                        "total_cost": spent_e2e,
                        "total_benefit": benefit_e2e,
                        "total_net_value": earned_e2e,
                        "n_intervened": len(interv_e2e),
                        "n_total": len(alloc_e2e),
                    },
                    "allocations": alloc_e2e.to_dict("records"),
                }
                new_saved_camp = create_campaign(
                    name=e2e_camp_name,
                    dataset_id=e2e_camp_ds_ref,
                    model_version=active_model_version,
                    scoring_run_id=f"scoring-run-{active_model_version}",
                    budget=float(e2e_budget),
                    assumptions={"margin": base_margin, "horizon_months": horizon, "actions": [a.code for a in current_actions]},
                    results=camp_results,
                    status=CampaignStatus.SIMULATED,
                )
                st.session_state["last_saved_camp_id"] = new_saved_camp.campaign_id
                st.success(f"🎉 Chiến dịch **'{new_saved_camp.name}'** đã được lưu bất biến vào DB! (Mã ID: `{new_saved_camp.campaign_id}`)")

        with col_act2:
            last_id = st.session_state.get("last_saved_camp_id")
            if last_id:
                csv_bytes = export_campaign_csv(last_id, only_interventions=True)
                st.download_button(
                    label="📥 Xuất File CSV Danh Sách Được Chọn (customer_id, hành động, chi phí, lợi ích)",
                    data=csv_bytes,
                    file_name=f"campaign_{last_id[:8]}_interventions.csv",
                    mime="text/csv",
                    type="secondary",
                    help="Tải file CSV danh sách khách hàng được chọn can thiệp và tự động cập nhật trạng thái sang EXPORTED.",
                    key="e2e_csv_dl_btn",
                )
            else:
                st.info("👉 Vui lòng nhấn **'💾 Lưu Chiến Dịch'** trước khi xuất file CSV.")


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
            <div class="modern-card card-accent-blue">
                <div class="card-label">
                    <span>Khách Hàng Hoạt Động</span>
                    <div class="card-icon-wrap" style="background:#eff6ff; color:#2563eb;">👥</div>
                </div>
                <div class="card-value">{n_total:,}</div>
                <div class="card-footer">
                    <span>Đủ điều kiện xét duyệt chính sách</span>
                </div>
            </div>
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Quyết Định Can Thiệp</span>
                    <div class="card-icon-wrap" style="background:#eef2ff; color:#4f46e5;">🎯</div>
                </div>
                <div class="card-value">{n_intervened:,}</div>
                <div class="card-footer">
                    Tỷ lệ tiếp cận: <span class="badge-indigo">{intervention_rate:.1%} tổng tệp</span>
                </div>
            </div>
            <div class="modern-card card-accent-amber">
                <div class="card-label">
                    <span>Tổng Ngân Sách Dự Kiến</span>
                    <div class="card-icon-wrap" style="background:#fffbeb; color:#d97706;">💳</div>
                </div>
                <div class="card-value">${total_cost:,.0f}</div>
                <div class="card-footer">
                    TB: <strong>${avg_cost_per_user:,.1f}</strong> / khách
                </div>
            </div>
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Lợi Nhuận Ròng (ΔEV)</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">🚀</div>
                </div>
                <div class="card-value" style="color: #059669;">+${total_net_value:,.0f}</div>
                <div class="card-footer">
                    Gross: <strong>${total_benefit:,.0f}</strong> | <span class="badge-emerald">ROI: +{roi:.1f}%</span>
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
                    textinfo="percent",
                    textposition="inside",
                    hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,} khách<br>Tỷ lệ: %{percent}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            height=350,
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>{n_total:,}</b><br><span style='font-size:12px;color:#64748b;'>Tổng khách</span>",
                    x=0.5,
                    y=0.5,
                    font_size=15,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_right:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">💰 Cân Đối Dòng Tiền & Hiệu Quả Can Thiệp</div>', unsafe_allow_html=True)

        total_contact_cost = sum(
            len(recommendations[recommendations["action_code"] == a.code]) * a.contact_cost
            for a in current_actions if not a.is_no_action
        )
        total_offer_cost = total_cost - total_contact_cost

        flow_labels = [
            "Lợi ích gộp (Gross Benefit)",
            "Chi phí liên hệ CSKH",
            "Chi phí gói ưu đãi/voucher",
            "Lợi nhuận ròng (Net ΔEV)"
        ]
        flow_values = [total_benefit, total_contact_cost, total_offer_cost, total_net_value]
        flow_colors = ["#0284c7", "#f43f5e", "#f97316", "#10b981"]
        flow_texts = [
            f"+${total_benefit:,.0f}",
            f"-${total_contact_cost:,.0f}",
            f"-${total_offer_cost:,.0f}",
            f"+${total_net_value:,.0f}"
        ]

        fig_cashflow = go.Figure(go.Bar(
            y=flow_labels,
            x=flow_values,
            orientation="h",
            text=flow_texts,
            textposition="auto",
            marker=dict(color=flow_colors),
            hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
        ))
        fig_cashflow.update_layout(
            height=320,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Giá trị tài chính ($)", gridcolor="rgba(148,163,184,0.25)"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12, family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig_cashflow, width="stretch")
        st.caption("💡 Lợi ích gộp bảo vệ được sau khi trừ chi phí CSKH và voucher sẽ đem lại khoản **Lợi nhuận ròng (ΔEV) dương**.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. SO SÁNH HIỆU QUẢ CÁC GÓI CAN THIỆP & BẢNG TỔNG HỢP CHI TIẾT
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chart-header">⚖️ So Sánh Hiệu Quả Tài Chính Từng Gói Can Thiệp (A1, A2, A3)</div>', unsafe_allow_html=True)

    interv_actions = summary_df[summary_df["action_code"] != "A0"].copy()
    col_bar, col_table = st.columns([0.7, 1.8])

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
            yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(tickangle=0),
        )
        st.plotly_chart(fig_grouped, width="stretch")

    with col_table:
        table_display = summary_df[[
            "action_code", "action_name", "n_customers", "percentage",
            "cost_sum", "benefit_sum", "net_value_sum"
        ]].copy().reset_index(drop=True)
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
            width="stretch",
            height=280,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # 4. XUẤT BÁO CÁO TÓM TẮT ĐIỀU HÀNH (1-CLICK EXECUTIVE SUMMARY)
    with st.expander("📄 Xem & Tải Báo Cáo Tóm Tắt Điều Hành (Executive Summary Report)", expanded=False):
        top_action = summary_df.loc[summary_df[summary_df["action_code"] != "A0"]["net_value_sum"].idxmax()]
        a0_row = summary_df[summary_df['action_code']=='A0']
        a0_count = int(a0_row['n_customers'].values[0]) if not a0_row.empty else 0
        report_text = f"""# BÁO CÁO TÓM TẮT CHIẾN LƯỢC GIỮ CHÂN KHÁCH HÀNG (EXECUTIVE SUMMARY)
**Dự án:** Customer Churn & Retention Decision Support System
**Kịch bản cơ sở:** Biên lợi nhuận m = {base_margin:.0%} | Kỳ hạn H = {horizon} tháng

---

### 1. KẾT QUẢ ĐIỀU HÀNH TỔNG THỂ
- **Quy mô tệp khách hàng hoạt động:** {n_total:,} khách hàng
- **Quy mô can thiệp được duyệt (ΔEV > 0):** {n_intervened:,} khách hàng ({intervention_rate:.1%} tổng tệp)
- **Tổng ngân sách đề xuất đầu tư:** ${total_cost:,.2f} (Chi phí trung bình: ${avg_cost_per_user:,.2f}/khách)
- **Tổng doanh thu giữ lại dự kiến (Gross Benefit):** ${total_benefit:,.2f}
- **Lợi nhuận ròng kỳ vọng (Net Economic Value - ΔEV):** +${total_net_value:,.2f}
- **Tỷ suất sinh lời trên chi phí (ROI):** +{roi:.1f}%

### 2. PHÂN BỔ THEO GÓI HÀNH ĐỘNG
- **Gói hiệu quả tài chính cao nhất:** {top_action['action_code']} ({top_action['action_name']}) mang lại +${top_action['net_value_sum']:,.2f} lợi nhuận ròng.
- **Phương án không can thiệp (A0):** {a0_count:,} khách hàng do chi phí can thiệp lớn hơn lợi ích bảo vệ được (ΔEV ≤ 0).

### 3. KHUYẾN NGHỊ CHÍNH SÁCH
1. Ưu tiên giải ngân trước cho nhóm khách hàng hợp đồng từng tháng và cước cao bằng gói A3.
2. Kiểm soát chặt chi phí tiếp cận qua điện thoại, chuyển dịch dần sang kênh số hóa để mở rộng điểm hòa vốn p*.
"""
        st.markdown(report_text)
        st.download_button(
            label="📥 Tải Văn Bản Báo Cáo (.MD)",
            data=report_text.encode("utf-8"),
            file_name="executive_retention_summary.md",
            mime="text/markdown",
        )

# ==============================================================================
# TAB 2: KHÁM PHÁ DỮ LIỆU & HÀNH VI CHURN (EDA PATTERNS)
# ==============================================================================
with tab2:
    st.markdown("### 🔍 Khám Phá Dữ Liệu & Quy Luật Hành Vi Khách Hàng Rời Bỏ")
    st.caption("Bức tranh dữ liệu toàn tệp viễn thông: Phát hiện các điểm nghẽn trải nghiệm và xác định chân dung nhóm khách hàng có nguy cơ rời mạng cao nhất.")

    # 1. KPI tổng quan tập dữ liệu
    eda_total = len(customers_df)
    churn_count = int((customers_df["Churn"] == "Yes").sum())
    churn_pct = (churn_count / eda_total) * 100 if eda_total > 0 else 0
    retained_count = eda_total - churn_count
    avg_mc = float(customers_df["MonthlyCharges"].mean())
    avg_tenure = float(customers_df["tenure"].mean())

    eda_kpi1, eda_kpi2, eda_kpi3, eda_kpi4 = st.columns(4)
    with eda_kpi1:
        st.markdown(
            f"""
            <div class="modern-card card-accent-blue">
                <div class="card-label">
                    <span>Tổng Tệp Khảo Sát</span>
                    <div class="card-icon-wrap" style="background:#eff6ff; color:#2563eb;">📁</div>
                </div>
                <div class="card-value">{eda_total:,}</div>
                <div class="card-footer"><span>Toàn bộ mẫu dữ liệu Telco</span></div>
            </div>
            """, unsafe_allow_html=True
        )
    with eda_kpi2:
        st.markdown(
            f"""
            <div class="modern-card card-accent-amber">
                <div class="card-label">
                    <span>Tỷ Lệ Churn Toàn Mạng</span>
                    <div class="card-icon-wrap" style="background:#fff1f2; color:#e11d48;">⚠️</div>
                </div>
                <div class="card-value" style="color: #e11d48;">{churn_pct:.1f}%</div>
                <div class="card-footer"><span>{churn_count:,} khách đã rời mạng</span></div>
            </div>
            """, unsafe_allow_html=True
        )
    with eda_kpi3:
        st.markdown(
            f"""
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Cước TB Toàn Mạng</span>
                    <div class="card-icon-wrap" style="background:#eef2ff; color:#4f46e5;">💵</div>
                </div>
                <div class="card-value">${avg_mc:.1f}</div>
                <div class="card-footer"><span>Cước trung bình / tháng</span></div>
            </div>
            """, unsafe_allow_html=True
        )
    with eda_kpi4:
        st.markdown(
            f"""
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Thời Gian Gắn Bó TB</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">⏳</div>
                </div>
                <div class="card-value">{avg_tenure:.1f} <span style="font-size:1.1rem;font-weight:600;">th</span></div>
                <div class="card-footer"><span>Khoảng ~{avg_tenure / 12:.1f} năm trung bình</span></div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Hai biểu đồ cột đối sánh: Hợp đồng & Internet Service
    eda_row1_c1, eda_row1_c2 = st.columns([1, 1])

    with eda_row1_c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">📑 Tỷ Lệ Churn Theo Loại Hợp Đồng (Contract)</div>', unsafe_allow_html=True)
        
        contract_churn = customers_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        contract_churn.columns = ["Contract", "ChurnRate"]
        contract_order = ["Month-to-month", "One year", "Two year"]
        contract_churn = contract_churn.sort_values(by="Contract", key=lambda x: x.map({v: i for i, v in enumerate(contract_order)}))

        fig_contract = go.Figure(go.Bar(
            x=contract_churn["Contract"],
            y=contract_churn["ChurnRate"],
            marker_color=["#f43f5e", "#f59e0b", "#10b981"],
            text=[f"{v:.1f}%" for v in contract_churn["ChurnRate"]],
            textposition="auto",
        ))
        fig_contract.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="Loại hợp đồng", tickangle=0),
        )
        st.plotly_chart(fig_contract, width="stretch")
        _cr = dict(zip(contract_churn["Contract"], contract_churn["ChurnRate"]))
        _m2m, _two = _cr.get("Month-to-month"), _cr.get("Two year")
        if _m2m is not None and _two:
            st.caption(f"💡 **Insight:** Hợp đồng Month-to-month có tỷ lệ churn {_m2m:.1f}%, cao gấp **{_m2m / _two:.1f} lần** so với hợp đồng 2 năm ({_two:.1f}%). Đây là lý do chính sách A3 nhắm thẳng vào nhóm này.")
        st.markdown('</div>', unsafe_allow_html=True)

    with eda_row1_c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">🌐 Nghịch Lý Dịch Vụ Cáp Quang (Fiber Optic Paradox)</div>', unsafe_allow_html=True)
        
        internet_churn = customers_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        internet_churn.columns = ["InternetService", "ChurnRate"]
        
        fig_internet = go.Figure(go.Bar(
            x=internet_churn["InternetService"],
            y=internet_churn["ChurnRate"],
            marker_color=["#0284c7", "#f43f5e", "#94a3b8"],
            text=[f"{v:.1f}%" for v in internet_churn["ChurnRate"]],
            textposition="auto",
        ))
        fig_internet.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="Dịch vụ Internet", tickangle=0),
        )
        st.plotly_chart(fig_internet, width="stretch")
        _ir = dict(zip(internet_churn["InternetService"], internet_churn["ChurnRate"]))
        _fiber, _dsl = _ir.get("Fiber optic"), _ir.get("DSL")
        if _fiber is not None and _dsl:
            st.caption(f"💡 **Insight:** Mạng Fiber optic có tỷ lệ churn {_fiber:.1f}%, gấp **{_fiber / _dsl:.1f} lần** DSL ({_dsl:.1f}%). Giả thuyết: giá cước cao và thường thiếu gói hỗ trợ kỹ thuật đi kèm (cần kiểm chứng thêm).")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Phân tích Dịch vụ đi kèm & Phương thức thanh toán
    eda_row2_c1, eda_row2_c2 = st.columns([1, 1])

    with eda_row2_c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">🛠️ Tác Động Của Hỗ Trợ Kỹ Thuật (Tech Support)</div>', unsafe_allow_html=True)
        
        tech_churn = customers_df.groupby("TechSupport")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        tech_churn.columns = ["TechSupport", "ChurnRate"]

        fig_tech = go.Figure(go.Bar(
            x=tech_churn["TechSupport"],
            y=tech_churn["ChurnRate"],
            marker_color=["#f43f5e", "#94a3b8", "#10b981"],
            text=[f"{v:.1f}%" for v in tech_churn["ChurnRate"]],
            textposition="auto",
        ))
        fig_tech.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="Có dịch vụ Tech Support không", tickangle=0),
        )
        st.plotly_chart(fig_tech, width="stretch")
        _tr = dict(zip(tech_churn["TechSupport"], tech_churn["ChurnRate"]))
        _no_ts, _yes_ts = _tr.get("No"), _tr.get("Yes")
        if _no_ts is not None and _yes_ts is not None:
            st.caption(f"💡 **Insight:** Khách không có hỗ trợ kỹ thuật rời mạng {_no_ts:.1f}%, trong khi khách có dịch vụ này chỉ {_yes_ts:.1f}%. Đây là căn cứ thiết kế gói A1.")
        st.markdown('</div>', unsafe_allow_html=True)

    with eda_row2_c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-header">💳 Tác Động Của Phương Thức Thanh Toán</div>', unsafe_allow_html=True)
        
        pay_churn = customers_df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        pay_churn.columns = ["PaymentMethod", "ChurnRate"]
        pay_churn["ShortName"] = pay_churn["PaymentMethod"].map({
            "Electronic check": "Electronic check (Thủ công)",
            "Mailed check": "Mailed check (Thủ công)",
            "Bank transfer (automatic)": "Bank transfer (Tự động)",
            "Credit card (automatic)": "Credit card (Tự động)"
        })

        fig_pay = go.Figure(go.Bar(
            y=pay_churn["ShortName"],
            x=pay_churn["ChurnRate"],
            orientation="h",
            marker_color=["#10b981", "#10b981", "#f43f5e", "#f59e0b"],
            text=[f"{v:.1f}%" for v in pay_churn["ChurnRate"]],
            textposition="auto",
        ))
        fig_pay.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.25)"),
        )
        st.plotly_chart(fig_pay, width="stretch")
        _pr = dict(zip(pay_churn["PaymentMethod"], pay_churn["ChurnRate"]))
        _echeck = _pr.get("Electronic check")
        _auto = [v for k, v in _pr.items() if "automatic" in str(k)]
        if _echeck is not None and _auto:
            st.caption(f"💡 **Insight:** Thanh toán bằng Electronic check rời mạng {_echeck:.1f}%, trong khi thanh toán tự động chỉ {min(_auto):.1f}–{max(_auto):.1f}%. Đây là căn cứ thiết kế gói A2.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: HỒ SƠ KHÁCH HÀNG 360° & ĐỀ XUẤT CÁ NHÂN HÓA (CUSTOMER 360)
# ==============================================================================
with tab3:
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

    # Hiển thị bảng tổng hợp (Index và số thứ tự bắt đầu từ 0)
    filtered_display = filtered_df[[
        "rank", "customerID", "action_code", "action_name", "churn_prob",
        "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
    ]].copy().reset_index(drop=True)
    filtered_display["rank"] = range(len(filtered_display))

    st.dataframe(
        filtered_display.style.format({
            "churn_prob": "{:.1%}",
            "churn_prob_after": "{:.1%}",
            "breakeven_prob": "{:.1%}",
            "proxy_value": "${:.2f}",
            "cost": "${:.2f}",
            "benefit": "${:.2f}",
            "net_value": "${:+.2f}",
        }),
        width="stretch",
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

        # HÀNG 1: PROFILE, QUYẾT ĐỊNH TỐI ƯU & YẾU TỐ TRẢI NGHIỆM GÂY RỜI BỎ (3 CỘT THOÁNG ĐÃNG)
        r1_col1, r1_col2, r1_col3 = st.columns([1, 1.15, 1.15])

        # HÀNG 1 - CỘT 1: PROFILE & GAUGE RỦI RO
        with r1_col1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-header">📌 Hồ Sơ Khách Hàng: {selected_cust_id}</div>', unsafe_allow_html=True)
            
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
                height=210,
                margin=dict(l=15, r=15, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(text="Vạch xanh: Sau can thiệp", x=0.5, y=-0.15, showarrow=False, font=dict(size=11, color="#3b82f6"))
                ]
            )
            st.plotly_chart(fig_gauge, width="stretch")

            # Chi tiết thông tin dịch vụ với visual badges
            contract_badge = '<span class="badge-rose">Month-to-month</span>' if cust_info['Contract'] == 'Month-to-month' else f'<span class="badge-emerald">{cust_info["Contract"]}</span>'
            tech_badge = '<span class="badge-rose">Chưa có</span>' if cust_info['TechSupport'] == 'No' else '<span class="badge-emerald">Đã kích hoạt</span>'
            sec_badge = '<span class="badge-rose">Chưa có</span>' if cust_info['OnlineSecurity'] == 'No' else '<span class="badge-emerald">Đã kích hoạt</span>'
            pay_badge = '<span class="badge-amber">Thủ công</span>' if cust_info['PaymentMethod'] in MANUAL_PAYMENTS else '<span class="badge-indigo">Tự động</span>'

            st.markdown(
                f"""
                <div class="profile-box">
                    <div class="profile-item"><span class="profile-key">Loại Hợp Đồng:</span> {contract_badge}</div>
                    <div class="profile-item"><span class="profile-key">Thời Gian Gắn Bó:</span> <span class="profile-val">{cust_info['tenure']} tháng</span></div>
                    <div class="profile-item"><span class="profile-key">Cước Phí Tháng:</span> <span class="profile-val" style="color:#0284c7;">${cust_info['MonthlyCharges']:.2f}</span></div>
                    <div class="profile-item"><span class="profile-key">Dịch Vụ Internet:</span> <span class="profile-val">{cust_info['InternetService']}</span></div>
                    <div class="profile-item"><span class="profile-key">Hỗ Trợ Kỹ Thuật:</span> {tech_badge}</div>
                    <div class="profile-item"><span class="profile-key">Bảo Mật Online:</span> {sec_badge}</div>
                    <div class="profile-item"><span class="profile-key">Thanh Toán:</span> {pay_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # HÀNG 1 - CỘT 2: QUYẾT ĐỊNH TỐI ƯU & CĂN CỨ KINH TẾ (THIẾT KẾ XUỐNG DÒNG RÕ RÀNG)
        with r1_col2:
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
                            <span>⭐</span> Gói Đề Xuất: {best_code} — {best_name}
                        </div>
                        <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                            <div style="background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 8px; border: 1px solid #bbf7d0;">
                                <div style="font-size: 0.78rem; color: #166534; font-weight: 600;">LỢI NHUẬN RÒNG (ΔEV)</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #15803d;">+${best_ev:.2f}</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 8px; border: 1px solid #bbf7d0;">
                                <div style="font-size: 0.78rem; color: #166534; font-weight: 600;">GIÁ TRỊ BẢO VỆ (V)</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #0369a1;">${proxy_val:.2f}</div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; padding: 12px 14px; background: rgba(255,255,255,0.85); border-radius: 10px; border: 1px solid #86efac; line-height: 1.6; font-size: 0.86rem; color: #14532d;">
                            <div>🎯 <b>Căn cứ kinh tế & điều kiện hòa vốn:</b></div>
                            <div style="margin-top: 4px;">• Ngưỡng hòa vốn tối thiểu: <b>p* = {breakeven:.1%}</b></div>
                            <div>• Rủi ro rời bỏ hiện tại: <b>p = {churn_val:.1f}%</b></div>
                            <div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #bbf7d0; color: #15803d; font-weight: 600;">
                                ➔ Vì <b>p ({churn_val:.1f}%) > p* ({breakeven:.1%})</b> nên can thiệp chắc chắn sinh lời kỳ vọng dương (+${best_ev:.2f}).
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="recommend-box" style="background:#f8fafc; border-color:#cbd5e1;">
                        <div class="recommend-title" style="color:#475569;">
                            <span>⚪</span> Đề Xuất: Gói A0 — Không Can Thiệp
                        </div>
                        <div class="recommend-sub" style="color:#64748b; margin-top: 8px; line-height: 1.6;">
                            Tất cả phương án can thiệp có chi phí đều mang lại lợi nhuận ròng âm (ΔEV ≤ 0) hoặc rủi ro rời bỏ thấp hơn ngưỡng hòa vốn (p < p*). Giữ nguyên trạng thái để tối ưu hóa hiệu quả ngân sách.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # HÀNG 1 - CỘT 3: YẾU TỐ TRẢI NGHIỆM GÂY RỜI BỎ (SỐ NẰM NGANG, RÕ RÀNG)
        with r1_col3:
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
                height=260,
                margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)", tickangle=0),
                yaxis=dict(autorange="reversed"),
            )
            fig_drivers.update_traces(
                textposition="inside",
                textangle=0,
                insidetextanchor="middle",
                textfont=dict(size=12, color="white", family="Plus Jakarta Sans", weight="bold"),
            )
            st.plotly_chart(fig_drivers, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # HÀNG 2: SO SÁNH GIÁ TRỊ RÒNG (ΔEV) & LUẬN CỨ RA QUYẾT ĐỊNH (2 CỘT RỘNG RÃI, THOÁNG MẮT)
        r2_col1, r2_col2 = st.columns([1.35, 1])

        # HÀNG 2 - CỘT 1: SO SÁNH CÁC PHƯƠNG ÁN HỢP LỆ (BIỂU ĐỒ & BẢNG CON)
        with r2_col1:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">⚖️ So Sánh Giá Trị Ròng (ΔEV) Từng Phương Án Hợp Lệ</div>', unsafe_allow_html=True)

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
                yaxis=dict(title="Lợi nhuận ròng ΔEV ($)", gridcolor="rgba(148,163,184,0.25)"),
                xaxis=dict(title="Mã hành động", tickangle=0),
            )
            st.plotly_chart(fig_action_comp, width="stretch")

            # Bảng con so sánh các phương án (Index bắt đầu từ 0)
            cust_eval_display = cust_eval[[
                "action_code", "action_name", "cost", "benefit", "net_value", "breakeven_prob"
            ]].rename(columns={
                "action_code": "Mã",
                "action_name": "Tên gói",
                "cost": "Chi phí C",
                "benefit": "Lợi ích B",
                "net_value": "Giá trị ròng ΔEV",
                "breakeven_prob": "Hòa vốn p*",
            }).reset_index(drop=True)

            st.dataframe(
                cust_eval_display.style.format({
                    "Chi phí C": "${:.2f}",
                    "Lợi ích B": "${:.2f}",
                    "Giá trị ròng ΔEV": "${:+.2f}",
                    "Hòa vốn p*": "{:.1%}",
                }),
                width="stretch",
                height=170,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # HÀNG 2 - CỘT 2: LUẬN CỨ RA QUYẾT ĐỊNH CAN THIỆP & PHÊ DUYỆT HÀNH ĐỘNG
        with r2_col2:
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-header">💡 Luận Cứ Ra Quyết Định Can Thiệp</div>', unsafe_allow_html=True)

            if best_code != "A0":
                be_val = cust_rec.get("breakeven_prob", None)
                be_text = f"{be_val:.1%}" if (be_val is not None and pd.notna(be_val)) else "N/A"
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 18px; font-size: 0.88rem; line-height: 1.6; color: #334155;">
                        <div style="margin-bottom: 8px;">• <b>Lý do lựa chọn gói {best_code}:</b> Phương án mang lại thặng dư lợi nhuận kỳ vọng ròng cao nhất (+${best_ev:.2f}) sau khi đã trừ toàn bộ chi phí liên hệ và ưu đãi (${cust_rec['cost']:.1f}).</div>
                        <div style="margin-bottom: 8px;">• <b>Điều kiện hòa vốn (p* = {be_text}):</b> Hiệu lực giữ chân của gói can thiệp cần giảm xác suất rời bỏ tối thiểu {be_text} để đảm bảo không lỗ vốn. Hiện tại rủi ro rời bỏ là {churn_val:.1f}%, vượt xa ngưỡng an toàn.</div>
                        <div>• <b>Khuyến nghị điều hành:</b> Phê duyệt gói can thiệp ngay để giữ chân khách hàng và bảo toàn dòng tiền cước định kỳ <b>${cust_info['MonthlyCharges']:.1f}/tháng</b>.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                if st.button("🚀 Phê Duyệt & Ghi Nhận Gói Can Thiệp", key="btn_apply", width="stretch"):
                    st.success(f"✅ Đã ghi nhận áp dụng gói {best_code} ({cust_rec['action_name']}) cho khách hàng {selected_cust_id}. Chi phí dự kiến: ${cust_rec['cost']:.2f}, Lợi nhuận ròng kỳ vọng: +${best_ev:.2f}!")
            else:
                st.info("Phương án tối ưu là **A0 (Không can thiệp)** vì mọi phương án có chi phí đều mang lại lợi nhuận ròng âm (ΔEV ≤ 0). Giữ nguyên trạng thái để tối ưu hóa ngân sách.")

            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: PHÂN TÍCH ĐỘ NHẠY & MÔ PHỎNG NGÂN SÁCH (SENSITIVITY & BUDGET SIMULATION)
# ==============================================================================
with tab4:
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
            yaxis=dict(title="Giá trị ($)", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="Biên Lợi Nhuận (m)", tickangle=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sens, width="stretch")
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
            yaxis=dict(title="Số lượng khách can thiệp", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="Biên Lợi Nhuận (m)", tickangle=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_stacked, width="stretch")
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
        width="stretch",
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
            value=max(2000, min(int(st.session_state.get("e2e_budget_slider", 15000)), max(int(current_total_cost) + 5000, 40000))),
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
                yaxis=dict(title="Giá trị ($)", gridcolor="rgba(148,163,184,0.25)"),
                xaxis=dict(title="Gói can thiệp", tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sim, width="stretch")
        else:
            st.info("Hạn mức ngân sách quá nhỏ để thực hiện bất kỳ hành động can thiệp nào.")

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 5: QUẢN LÝ CHIẾN DỊCH (CAMPAIGN MANAGEMENT - TASK 2)
# ==============================================================================
with tab5:
    st.markdown("### 📋 Quản Lý Chiến Dịch Mô Phỏng Giữ Chân Khách Hàng")
    st.caption("Lưu trữ, theo dõi và xuất danh sách các chiến dịch can thiệp theo ngân sách với nguyên tắc bất biến (Immutability).")

    # Banner giải thích nguyên tắc bất biến
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 12px; padding: 16px 20px; color: #fff; margin-bottom: 20px; border-left: 4px solid #6366f1;">
            <div style="font-weight: 700; font-size: 1rem; color: #a5b4fc; margin-bottom: 4px;">🔒 NGUYÊN TẮC BẤT BIẾN & TRUY VẾT DỮ LIỆU:</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                Mỗi chiến dịch được gắn chặt với 3 tham chiếu bất biến: <strong>Dataset ID</strong>, <strong>Model Version</strong>, và <strong>Scoring Run ID</strong>. 
                Khi đổi sang mô hình mới hoặc tập dữ liệu mới, hệ thống sẽ tạo một chiến dịch mới độc lập và bảo toàn 100% kết quả lịch sử của chiến dịch cũ, không bao giờ ghi đè.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_campaigns = list_campaigns()

    camp_sub1, camp_sub2 = st.tabs(["📂 Danh Sách Chiến Dịch Đã Lưu", "➕ Tạo Chiến Dịch Mới Từ Mô Phỏng"])

    with camp_sub1:
        if not all_campaigns:
            st.info("ℹ️ Hiện chưa có chiến dịch nào được lưu. Bạn có thể chuyển sang tab **'➕ Tạo Chiến Dịch Mới Từ Mô Phỏng'** để khởi tạo chiến dịch đầu tiên.")
        else:
            camp_options = {c.campaign_id: f"{c.name} [{c.status.value}] — ${c.budget:,.0f} — {c.created_at[:19]}" for c in all_campaigns}
            selected_camp_id = st.selectbox(
                "Chọn chiến dịch để xem chi tiết & xuất báo cáo:",
                options=list(camp_options.keys()),
                format_func=lambda x: camp_options[x],
            )

            if selected_camp_id:
                selected_camp = get_campaign(selected_camp_id)
                summary_data = selected_camp.results.get("summary", {})

                # Metadata Cards
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                with meta_col1:
                    st.metric("Tên chiến dịch", selected_camp.name)
                    st.caption(f"ID: `{selected_camp.campaign_id[:8]}...`")
                with meta_col2:
                    st.metric("Dataset Reference", selected_camp.dataset_id)
                    st.caption(f"Created: {selected_camp.created_at[:10]}")
                with meta_col3:
                    st.metric("Model Version", selected_camp.model_version)
                    st.caption(f"Scoring Run: `{selected_camp.scoring_run_id}`")
                with meta_col4:
                    badge_color = "#10b981" if selected_camp.status == CampaignStatus.EXPORTED else "#3b82f6"
                    st.metric("Trạng Thái", selected_camp.status.value)
                    st.markdown(f'<span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:700;">{selected_camp.status.value}</span>', unsafe_allow_html=True)

                st.markdown("---")

                # 4 KPI cards của chiến dịch
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.metric("Hạn Mức Ngân Sách ($)", f"${selected_camp.budget:,.2f}")
                with kpi_col2:
                    spent_val = summary_data.get("total_cost", 0.0)
                    st.metric("Chi Phí Thực Tiêu ($)", f"${spent_val:,.2f}")
                with kpi_col3:
                    benefit_val = summary_data.get("total_benefit", 0.0)
                    st.metric("Lợi Ích Kỳ Vọng ($)", f"${benefit_val:,.2f}")
                with kpi_col4:
                    net_val = summary_data.get("total_net_value", 0.0)
                    st.metric("Lợi Nhuận Ròng (ΔEV)", f"+${net_val:,.2f}")

                st.markdown("#### 👥 Danh Sách Khách Hàng Được Chọn Can Thiệp")
                camp_df = export_campaign_dataframe(selected_camp.campaign_id)
                interv_df = camp_df[camp_df["action_code"] != "A0"] if not camp_df.empty and "action_code" in camp_df.columns else camp_df

                if not interv_df.empty:
                    st.dataframe(
                        interv_df.style.format({
                            "cost": "${:,.2f}",
                            "benefit": "${:,.2f}",
                            "net_value": "${:,.2f}",
                        }),
                        width="stretch",
                        height=280,
                    )
                else:
                    st.info("Chiến dịch này không có khách hàng nào được phân bổ can thiệp tốn phí (Toàn bộ là A0).")

                # Nút xuất CSV (Step 3 của Task 2)
                csv_data = export_campaign_csv(selected_camp.campaign_id, only_interventions=True)
                st.download_button(
                    label="📥 Xuất File CSV Danh Sách Được Chọn (customer_id, hành động, chi phí, lợi ích)",
                    data=csv_data,
                    file_name=f"campaign_{selected_camp.name.replace(' ', '_')}_{selected_camp.campaign_id[:8]}.csv",
                    mime="text/csv",
                    type="primary",
                    help="Xuất danh sách khách hàng được chọn can thiệp và tự động chuyển trạng thái sang EXPORTED.",
                )

    with camp_sub2:
        st.markdown("#### ➕ Tạo Bản Ghi Chiến Dịch Mới")
        st.caption("Khởi tạo chiến dịch lưu trữ từ mô phỏng ngân sách hiện tại.")

        cur_model_version = "catboost-20261018T130222-seed42" if "CatBoost" in predictor_info else "fake-v1"
        cur_dataset_id = "telco_churn.csv"
        cur_scoring_run_id = f"scoring-run-{cur_model_version}"

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            new_camp_name = st.text_input("Tên chiến dịch:", value=f"Chiến dịch giữ chân - {pd.Timestamp.now().strftime('%d/%m/%Y')}")
            new_camp_budget = st.number_input("Hạn mức ngân sách ($):", min_value=0.0, value=float(st.session_state.get("e2e_budget_slider", 15000)), step=1000.0)
        with c_col2:
            st.text_input("Dataset Reference (Linh hoạt):", value=active_dataset_name, help="Mã định danh hoặc tên tập dữ liệu áp dụng cho chiến dịch này.")
            st.text_input("Model Version (bất biến):", value=cur_model_version, disabled=True)

        if st.button("🚀 Khởi Tạo & Chạy Phân Bổ Chiến Dịch Mới", type="primary"):
            # Chạy thuật toán phân bổ ngân sách
            alloc_df = select_actions_under_budget(
                customers=customers_df,
                predictions=predictions_df,
                budget=new_camp_budget,
                actions=current_actions,
                margin=base_margin,
                horizon_months=horizon,
            )
            interv_alloc = alloc_df[alloc_df["action_code"] != "A0"]

            camp_results = {
                "summary": {
                    "total_cost": float(interv_alloc["cost"].sum()) if not interv_alloc.empty else 0.0,
                    "total_benefit": float(interv_alloc["benefit"].sum()) if not interv_alloc.empty else 0.0,
                    "total_net_value": float(interv_alloc["net_value"].sum()) if not interv_alloc.empty else 0.0,
                    "n_intervened": len(interv_alloc),
                    "n_total": len(alloc_df),
                },
                "allocations": alloc_df.to_dict("records"),
            }

            created_camp = create_campaign(
                name=new_camp_name,
                dataset_id=cur_dataset_id,
                model_version=cur_model_version,
                scoring_run_id=cur_scoring_run_id,
                budget=new_camp_budget,
                assumptions={
                    "margin": base_margin,
                    "horizon_months": horizon,
                    "actions": [a.code for a in current_actions],
                },
                results=camp_results,
                status=CampaignStatus.SIMULATED,
            )
            st.success(f"🎉 Đã tạo thành công chiến dịch **'{created_camp.name}'** (ID: `{created_camp.campaign_id}`)!")
            st.rerun()

    # --- THEO DÕI HÀNG ĐỢI TASK QUEUE & CHỐNG TRÙNG LẶP (TASK 4) ---
    with st.expander("⚙️ Giám Sát Hạ Tầng Task Queue & Khóa Chống Trùng (Retry & Idempotency)"):
        st.caption("Truy vết trạng thái các tác vụ ngầm (Background Jobs), chính sách Retry có giới hạn và cơ chế Idempotent.")
        from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
        queue = SqliteTaskQueue()
        jobs = queue.list_jobs(limit=10)
        if jobs:
            jobs_df = pd.DataFrame([
                {
                    "Job ID": j.job_id[:8] + "...",
                    "Loại Job": j.job_type.value,
                    "Idempotency Key": j.idempotency_key,
                    "Trạng Thái": j.status.value,
                    "Số lần Retry": f"{j.retry_count}/{j.max_retries}",
                    "Lỗi (nếu có)": j.error_reason or "None",
                    "Thời gian": j.created_at[:19],
                }
                for j in jobs
            ])
            st.dataframe(jobs_df, width="stretch")
        else:
            st.info("Hiện chưa có tác vụ nền nào trong hàng đợi.")
