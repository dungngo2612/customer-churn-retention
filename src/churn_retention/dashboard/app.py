"""Streamlit Dashboard for Customer Retention & Economic Optimization (Person 3).

Design System: UI/UX Pro Max (Data-Dense Enterprise Dashboard — Stripe / Linear Minimalist Aesthetic)
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
    page_title="Retention Decision Support System | Telecom Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LUCIDE SVG ICONS (CHUẨN UI/UX PRO MAX - KHÔNG DÙNG EMOJI LÒE LOẸT) ---
SVG_USERS = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
SVG_TARGET = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
SVG_WALLET = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/><path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/></svg>'
SVG_TRENDING = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>'
SVG_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
SVG_CLOCK = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
SVG_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
SVG_LAYERS = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>'

# --- DESIGN SYSTEM CSS (UI/UX PRO MAX + STRIPE / LINEAR MINIMALISM) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }

    /* Header tinh giản chuẩn Linear / Stripe */
    .exec-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1e40af;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .exec-header-left {
        max-width: 760px;
    }
    .exec-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1e40af;
        margin-bottom: 4px;
    }
    .exec-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .exec-subtitle {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }
    .exec-meta-group {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        align-items: center;
    }
    .meta-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
        font-family: 'Fira Code', monospace;
    }
    .meta-chip-active {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* KPI Grid 4 cột chuẩn Enterprise */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .modern-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .modern-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .modern-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .card-accent-blue::before { background: #1e40af; }
    .card-accent-indigo::before { background: #4f46e5; }
    .card-accent-amber::before { background: #d97706; }
    .card-accent-emerald::before { background: #059669; }

    .card-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-icon-wrap {
        width: 30px;
        height: 30px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .card-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }
    .card-footer {
        font-size: 0.8rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges ngữ nghĩa */
    .badge-emerald {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'Fira Code', monospace;
    }
    .badge-indigo {
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'Fira Code', monospace;
    }
    .badge-rose {
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-amber {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* Native Streamlit Bordered Containers làm Card thật bao quanh biểu đồ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 12px !important;
    }

    .panel-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: space-between;
        letter-spacing: -0.01em;
    }
    .panel-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
    }

    /* Hồ sơ Customer 360 styling */
    .profile-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 8px;
    }
    .profile-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.84rem;
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
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }

    /* Hộp Đề Xuất Tối Ưu */
    .recommend-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #059669;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .recommend-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
    }

    /* Sidebar chuẩn sạch */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Segmented Tabs chuẩn Component Gallery */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
        background-color: transparent;
        border: none;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e40af !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- NẠP DỮ LIỆU & CHẠY DỰ ĐOÁN ---
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
            st.error("Không tìm thấy file dữ liệu thật `telco_churn.csv`. Vui lòng kiểm tra thư mục data/.")
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
                "Không tìm thấy model bundle thật trong `artifacts/bundles/`. "
                "Luồng chạy thật bắt buộc phải có bundle CatBoost."
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
            st.error(f"Lỗi khi nạp hoặc chạy model bundle: {exc}")
            st.stop()
    else:
        fake_pred = FakePredictor(seed=42)
        probs = fake_pred.predict_batch(
            features=customers,
            customer_ids=customers["customerID"].tolist()
        )
        model_version = fake_pred.model_version
        predictor_info = f"FakePredictor (Demo - {model_version})"

    predictions = pd.DataFrame({
        "customer_id": customers["customerID"].tolist(),
        "churn_prob": probs,
    })

    return customers, predictions, predictor_info, dataset_name, model_version


if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False

customers_df, predictions_df, predictor_info, active_dataset_name, active_model_version = load_data_and_predict(
    is_demo_mode=st.session_state["demo_mode"]
)

# --- SIDEBAR: CẤU HÌNH THAM SỐ KINH TẾ ---
with st.sidebar:
    st.markdown("### Tham Số Kinh Tế (CLV)")
    st.caption("Giả định tài chính cho mô hình Proxy CLV")

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
    st.markdown("### Cấu Hình Gói Can Thiệp")
    st.caption("Chi phí ($c, f$), tỷ lệ chấp nhận ($u$) & mức giảm ($e$)")

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
    st.markdown("### Dữ Liệu & Mô Hình AI")
    bundle_ready = "CatBoost" in predictor_info
    st.caption(f"• Tập dữ liệu: **`{active_dataset_name}`** ({len(customers_df):,} bản ghi)")
    st.caption(f"• Mô hình: **{'CatBoost Bundle (ROC-AUC 84.8%)' if bundle_ready else predictor_info}**")

    with st.expander("Đổi file CSV / Hạ tầng hệ thống", expanded=False):
        uploaded_file = st.file_uploader(
            "Tải lên file CSV khách hàng mới (cùng cấu trúc Telco):",
            type=["csv"],
            key="sidebar_csv_uploader",
        )
        if uploaded_file is not None:
            try:
                _up_df = pd.read_csv(uploaded_file)
                if "TotalCharges" in _up_df.columns:
                    _up_df["TotalCharges"] = pd.to_numeric(
                        _up_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
                    ).fillna(0.0)
                _val_rep = validate_dataset(_up_df)
                if _val_rep.is_valid:
                    customers_df = _up_df
                    active_dataset_name = uploaded_file.name
                    _bpath = get_bundle_path()
                    if not st.session_state["demo_mode"] and _bpath and _bpath.exists():
                        from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
                        _bp_up = BundlePredictor()
                        _bp_up.load(str(_bpath))
                        _probs_up = _bp_up.predict_batch(
                            features=customers_df.drop(columns=[c for c in ("customerID", "Churn") if c in customers_df.columns]),
                            customer_ids=customers_df["customerID"].tolist(),
                        )
                    else:
                        _probs_up = FakePredictor(seed=42).predict_batch(
                            features=customers_df,
                            customer_ids=customers_df["customerID"].tolist(),
                        )
                    predictions_df = pd.DataFrame({
                        "customer_id": customers_df["customerID"].tolist(),
                        "churn_prob": _probs_up,
                    })
                    st.success(f"Đã nạp `{uploaded_file.name}` ({len(customers_df):,} khách hàng).")
                else:
                    st.error(f"File có {len(_val_rep.errors)} lỗi cấu trúc Schema:")
                    st.dataframe(
                        pd.DataFrame([{"Dòng": e.row or "Bảng", "Cột": e.column, "Lỗi": e.reason} for e in _val_rep.errors]),
                        width="stretch",
                    )
            except Exception as _e:
                st.error(f"Lỗi đọc file CSV: {_e}")
        else:
            _def_val = validate_dataset(customers_df)
            if _def_val.is_valid:
                st.success(f"Kiểm định Schema: {_def_val.total_rows:,} dòng hợp lệ 100%.")

        st.markdown("---")
        st.caption("**Task Queue & Idempotency:**")
        if st.button("Chạy & Ghi Nhận Scoring Job", key="btn_run_scoring", width="stretch", disabled=not bundle_ready):
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
                    f"Job chấm điểm này đã tồn tại (trạng thái **{_existing_job.status.value}**). "
                    "Khóa chống trùng (idempotency key) chặn việc chạy và ghi trùng kết quả."
                )
            else:
                with st.spinner("Đang xử lý qua hàng đợi bằng CatBoost..."):
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
                        f"Scoring Job hoàn tất: đã lưu {_done.result.get('total_stored_in_db', 0):,} dự đoán "
                        f"(mới: {_done.result.get('inserted_new', 0):,})."
                    )
                elif _done is not None:
                    st.error(f"Scoring Job {_done.status.value}: {_done.error_reason}")
            st.session_state["scoring_run_id"] = _run_id

        from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
        _q_list = SqliteTaskQueue().list_jobs(limit=5)
        if _q_list:
            st.dataframe(
                pd.DataFrame([
                    {"Job": j.job_id[:6], "Trạng thái": j.status.value, "Retry": f"{j.retry_count}/{j.max_retries}"}
                    for j in _q_list
                ]),
                width="stretch",
            )

# Dựng lại danh mục hành động
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


@st.cache_resource
def get_cached_bundle_predictor():
    bpath = get_bundle_path()
    if bpath and bpath.exists():
        try:
            from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
            bp_obj = BundlePredictor()
            bp_obj.load(str(bpath))
            return bp_obj
        except Exception:
            return None
    return None


cached_bp = get_cached_bundle_predictor()

recommendations = recommend(
    customers=customers_df,
    predictions=predictions_df,
    actions=current_actions,
    margin=base_margin,
    horizon_months=horizon,
)

interventions = recommendations[recommendations["action_code"] != "A0"]

# --- EXECUTIVE HEADER (CHUẨN LINEAR / STRIPE MINIMAL) ---
st.markdown(
    f"""
    <div class="exec-header">
        <div class="exec-header-left">
            <div class="exec-eyebrow">TELECOM RETENTION DECISION SUPPORT SYSTEM</div>
            <div class="exec-title">Hệ Thống Tối Ưu Hóa Kinh Tế & Giữ Chân Khách Hàng</div>
            <div class="exec-subtitle">
                Cá nhân hóa chính sách can thiệp dựa trên mô hình dự báo CatBoost, giải thích TreeSHAP và tối đa hóa Lợi nhuận Ròng Kỳ vọng (ΔEV).
            </div>
        </div>
        <div class="exec-meta-group">
            <span class="meta-chip">Dataset: {active_dataset_name} ({len(customers_df):,})</span>
            <span class="meta-chip">Margin m = {base_margin:.0%} | H = {horizon}M</span>
            <span class="meta-chip meta-chip-active">{SVG_CHECK} CatBoost ROC-AUC 84.8%</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 4 TABS ĐIỀU HƯỚNG (KHÔNG EMOJI LÒE LOẸT) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "01. Tổng Quan Kinh Tế",
    "02. Hành Vi Churn & Giải Thích AI (SHAP)",
    "03. Hồ Sơ Khách Hàng 360°",
    "04. Tối Ưu Ngân Sách & Chiến Dịch",
])


# ==============================================================================
# TAB 1: TỔNG QUAN KINH TẾ (OVERVIEW)
# ==============================================================================
with tab1:
    n_total = len(recommendations)
    n_intervened = len(interventions)
    intervention_rate = (n_intervened / n_total) if n_total > 0 else 0
    total_cost = interventions["cost"].sum()
    total_net_value = interventions["net_value"].sum()
    total_benefit = interventions["benefit"].sum()
    avg_cost_per_user = (total_cost / n_intervened) if n_intervened > 0 else 0
    roi = ((total_net_value / total_cost) * 100) if total_cost > 0 else 0

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="modern-card card-accent-blue">
                <div class="card-label">
                    <span>Khách Hàng Hoạt Động</span>
                    <div class="card-icon-wrap" style="background:#eff6ff; color:#1e40af;">{SVG_USERS}</div>
                </div>
                <div class="card-value">{n_total:,}</div>
                <div class="card-footer">
                    <span>Đủ điều kiện xét duyệt chính sách</span>
                </div>
            </div>
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Quyết Định Can Thiệp</span>
                    <div class="card-icon-wrap" style="background:#eef2ff; color:#4f46e5;">{SVG_TARGET}</div>
                </div>
                <div class="card-value">{n_intervened:,}</div>
                <div class="card-footer">
                    Tỷ lệ tiếp cận: <span class="badge-indigo">{intervention_rate:.1%}</span>
                </div>
            </div>
            <div class="modern-card card-accent-amber">
                <div class="card-label">
                    <span>Ngân Sách Đề Xuất</span>
                    <div class="card-icon-wrap" style="background:#fffbeb; color:#d97706;">{SVG_WALLET}</div>
                </div>
                <div class="card-value">${total_cost:,.0f}</div>
                <div class="card-footer">
                    Trung bình: <strong>${avg_cost_per_user:,.1f}</strong> / khách
                </div>
            </div>
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Lợi Nhuận Ròng (ΔEV)</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">{SVG_TRENDING}</div>
                </div>
                <div class="card-value" style="color: #059669;">+${total_net_value:,.0f}</div>
                <div class="card-footer">
                    Gross: <strong>${total_benefit:,.0f}</strong> <span class="badge-emerald">ROI +{roi:.1f}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    col_chart_left, col_chart_right = st.columns([1, 1.2])

    with col_chart_left:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Cơ Cấu Phân Bổ Hành Động Can Thiệp</span><span class="panel-tag">SHARE OF WALLET</span></div>', unsafe_allow_html=True)

            colors_map = {
                "A0": "#94a3b8",
                "A1": "#1e40af",
                "A2": "#0284c7",
                "A3": "#059669",
            }
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=[f"{row.action_code}: {row.action_name}" for _, row in summary_df.iterrows()],
                        values=summary_df["n_customers"],
                        hole=0.60,
                        marker=dict(colors=[colors_map.get(code, "#4f46e5") for code in summary_df["action_code"]]),
                        textinfo="percent",
                        textposition="inside",
                        hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,} khách<br>Tỷ lệ: %{percent}<extra></extra>",
                    )
                ]
            )
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, font=dict(size=11)),
                height=340,
                margin=dict(l=10, r=10, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text=f"<b>{n_total:,}</b><br><span style='font-size:11px;color:#64748b;'>Khách hàng</span>",
                        x=0.5,
                        y=0.5,
                        font_size=15,
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_donut, width="stretch")

    with col_chart_right:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Cân Đối Dòng Tiền & Hiệu Quả Can Thiệp</span><span class="panel-tag">CASHFLOW BRIDGE</span></div>', unsafe_allow_html=True)

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
            flow_colors = ["#1e40af", "#e11d48", "#d97706", "#059669"]
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
                height=310,
                margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Giá trị tài chính ($)", gridcolor="rgba(148,163,184,0.22)"),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            )
            st.plotly_chart(fig_cashflow, width="stretch")
            st.caption("Lợi ích gộp bảo vệ được sau khi trừ chi phí tiếp cận và ưu đãi mang lại thặng dư kinh tế ròng (ΔEV > 0).")

    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>So Sánh Hiệu Quả Tài Chính Từng Gói Can Thiệp (A1, A2, A3)</span><span class="panel-tag">PACKAGE ECONOMICS</span></div>', unsafe_allow_html=True)

        interv_actions = summary_df[summary_df["action_code"] != "A0"].copy()
        col_bar, col_table = st.columns([0.75, 1.75])

        with col_bar:
            fig_grouped = go.Figure(
                data=[
                    go.Bar(
                        name="Tổng Chi Phí",
                        x=interv_actions["action_code"],
                        y=interv_actions["cost_sum"],
                        marker_color="#d97706",
                        text=[f"${v:,.0f}" for v in interv_actions["cost_sum"]],
                        textposition="auto",
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng (ΔEV)",
                        x=interv_actions["action_code"],
                        y=interv_actions["net_value_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in interv_actions["net_value_sum"]],
                        textposition="auto",
                    ),
                ]
            )
            fig_grouped.update_layout(
                barmode="group",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(gridcolor="rgba(148,163,184,0.22)"),
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
                height=260,
            )

    with st.expander("Xem & Tải Báo Cáo Tóm Tắt Điều Hành (Executive Summary Report)", expanded=False):
        top_action = summary_df.loc[summary_df[summary_df["action_code"] != "A0"]["net_value_sum"].idxmax()]
        a0_row = summary_df[summary_df['action_code'] == 'A0']
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
"""
        st.markdown(report_text)
        st.download_button(
            label="Tải Văn Bản Báo Cáo (.MD)",
            data=report_text.encode("utf-8"),
            file_name="executive_retention_summary.md",
            mime="text/markdown",
        )


# ==============================================================================
# TAB 2: HÀNH VI RỜI MẠNG & GIẢI THÍCH AI (EDA & GLOBAL SHAP)
# ==============================================================================
with tab2:
    eda_total = len(customers_df)
    churn_count = int((customers_df["Churn"] == "Yes").sum())
    churn_pct = (churn_count / eda_total) * 100 if eda_total > 0 else 0
    avg_mc = float(customers_df["MonthlyCharges"].mean())
    avg_tenure = float(customers_df["tenure"].mean())

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="modern-card card-accent-blue">
                <div class="card-label">
                    <span>Tổng Tệp Khảo Sát</span>
                    <div class="card-icon-wrap" style="background:#eff6ff; color:#1e40af;">{SVG_USERS}</div>
                </div>
                <div class="card-value">{eda_total:,}</div>
                <div class="card-footer"><span>Toàn bộ mẫu dữ liệu Telco</span></div>
            </div>
            <div class="modern-card card-accent-amber">
                <div class="card-label">
                    <span>Tỷ Lệ Churn Toàn Mạng</span>
                    <div class="card-icon-wrap" style="background:#fff1f2; color:#be123c;">{SVG_ALERT}</div>
                </div>
                <div class="card-value" style="color: #be123c;">{churn_pct:.1f}%</div>
                <div class="card-footer"><span>{churn_count:,} khách hàng rời mạng</span></div>
            </div>
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Cước Trung Bình Tháng</span>
                    <div class="card-icon-wrap" style="background:#eef2ff; color:#4f46e5;">{SVG_WALLET}</div>
                </div>
                <div class="card-value">${avg_mc:.1f}</div>
                <div class="card-footer"><span>MonthlyCharges trung bình</span></div>
            </div>
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Thời Gian Gắn Bó TB</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">{SVG_CLOCK}</div>
                </div>
                <div class="card-value">{avg_tenure:.1f}M</div>
                <div class="card-footer"><span>Khoảng ~{avg_tenure / 12:.1f} năm trung bình</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eda_row1_c1, eda_row1_c2 = st.columns(2)

    with eda_row1_c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Tỷ Lệ Churn Theo Loại Hợp Đồng (Contract)</span><span class="panel-tag">CONTRACT COHORT</span></div>', unsafe_allow_html=True)

            contract_churn = customers_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            contract_churn.columns = ["Contract", "ChurnRate"]
            contract_order = ["Month-to-month", "One year", "Two year"]
            contract_churn = contract_churn.sort_values(by="Contract", key=lambda x: x.map({v: i for i, v in enumerate(contract_order)}))

            fig_contract = go.Figure(go.Bar(
                x=contract_churn["Contract"],
                y=contract_churn["ChurnRate"],
                marker_color=["#e11d48", "#d97706", "#059669"],
                text=[f"{v:.1f}%" for v in contract_churn["ChurnRate"]],
                textposition="auto",
            ))
            fig_contract.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Loại hợp đồng", tickangle=0),
            )
            st.plotly_chart(fig_contract, width="stretch")
            _cr = dict(zip(contract_churn["Contract"], contract_churn["ChurnRate"]))
            _m2m, _two = _cr.get("Month-to-month"), _cr.get("Two year")
            if _m2m is not None and _two:
                st.caption(f"Nhận xét: Hợp đồng Month-to-month có tỷ lệ churn {_m2m:.1f}%, cao gấp **{_m2m / _two:.1f} lần** hợp đồng 2 năm ({_two:.1f}%). Đây là cơ sở thiết kế gói A3.")

    with eda_row1_c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Tỷ Lệ Churn Theo Dịch Vụ Internet</span><span class="panel-tag">SERVICE TIER</span></div>', unsafe_allow_html=True)

            internet_churn = customers_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            internet_churn.columns = ["InternetService", "ChurnRate"]

            fig_internet = go.Figure(go.Bar(
                x=internet_churn["InternetService"],
                y=internet_churn["ChurnRate"],
                marker_color=["#1e40af", "#e11d48", "#94a3b8"],
                text=[f"{v:.1f}%" for v in internet_churn["ChurnRate"]],
                textposition="auto",
            ))
            fig_internet.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Dịch vụ Internet", tickangle=0),
            )
            st.plotly_chart(fig_internet, width="stretch")
            _ir = dict(zip(internet_churn["InternetService"], internet_churn["ChurnRate"]))
            _fiber, _dsl = _ir.get("Fiber optic"), _ir.get("DSL")
            if _fiber is not None and _dsl:
                st.caption(f"Nhận xét: Nhóm Fiber optic có tỷ lệ churn {_fiber:.1f}%, gấp **{_fiber / _dsl:.1f} lần** DSL ({_dsl:.1f}%).")

    eda_row2_c1, eda_row2_c2 = st.columns(2)

    with eda_row2_c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Tác Động Của Hỗ Trợ Kỹ Thuật (Tech Support)</span><span class="panel-tag">SUPPORT IMPACT</span></div>', unsafe_allow_html=True)

            tech_churn = customers_df.groupby("TechSupport")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            tech_churn.columns = ["TechSupport", "ChurnRate"]

            fig_tech = go.Figure(go.Bar(
                x=tech_churn["TechSupport"],
                y=tech_churn["ChurnRate"],
                marker_color=["#e11d48", "#94a3b8", "#059669"],
                text=[f"{v:.1f}%" for v in tech_churn["ChurnRate"]],
                textposition="auto",
            ))
            fig_tech.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Trạng thái Tech Support", tickangle=0),
            )
            st.plotly_chart(fig_tech, width="stretch")
            _tr = dict(zip(tech_churn["TechSupport"], tech_churn["ChurnRate"]))
            _no_ts, _yes_ts = _tr.get("No"), _tr.get("Yes")
            if _no_ts is not None and _yes_ts is not None:
                st.caption(f"Nhận xét: Khách thiếu Tech Support rời mạng {_no_ts:.1f}%, so với {_yes_ts:.1f}% ở nhóm có hỗ trợ. Đây là căn cứ thiết kế gói A1.")

    with eda_row2_c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Tác Động Của Phương Thức Thanh Toán</span><span class="panel-tag">PAYMENT CHANNEL</span></div>', unsafe_allow_html=True)

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
                marker_color=["#059669", "#059669", "#e11d48", "#d97706"],
                text=[f"{v:.1f}%" for v in pay_churn["ChurnRate"]],
                textposition="auto",
            ))
            fig_pay.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Tỷ lệ rời mạng (%)", gridcolor="rgba(148,163,184,0.22)"),
            )
            st.plotly_chart(fig_pay, width="stretch")
            _pr = dict(zip(pay_churn["PaymentMethod"], pay_churn["ChurnRate"]))
            _echeck = _pr.get("Electronic check")
            _auto = [v for k, v in _pr.items() if "automatic" in str(k)]
            if _echeck is not None and _auto:
                st.caption(f"Nhận xét: Thanh toán Electronic check rời mạng {_echeck:.1f}%, trong khi thanh toán tự động chỉ {min(_auto):.1f}–{max(_auto):.1f}%. Đây là căn cứ thiết kế gói A2.")

    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>Giải Thích Mô Hình AI: Top 10 Đặc Trưng Quyết Định Rủi Ro Churn (Global TreeSHAP)</span><span class="panel-tag">MODEL EXPLAINABILITY</span></div>', unsafe_allow_html=True)

        shap_g_col1, shap_g_col2 = st.columns([1.3, 1])
        with shap_g_col1:
            df_importances = get_global_feature_importances(cached_bp) if cached_bp is not None else pd.DataFrame()
            if not df_importances.empty:
                top10_imp = df_importances.head(10).sort_values(by="importance", ascending=True)
                fig_shap_bar = go.Figure(go.Bar(
                    x=top10_imp["importance"],
                    y=top10_imp["feature"],
                    orientation="h",
                    marker_color="#1e40af",
                    text=[f"{v:.1f}%" for v in top10_imp["importance"]],
                    textposition="auto",
                ))
                fig_shap_bar.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Tầm quan trọng tương đối (%)", gridcolor="rgba(148,163,184,0.22)"),
                )
                st.plotly_chart(fig_shap_bar, width="stretch")
            else:
                st.info("Biểu đồ Global SHAP yêu cầu Model Bundle CatBoost thật.")

        with shap_g_col2:
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; font-size: 0.86rem; line-height: 1.65; color: #334155;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0f172a; margin-bottom: 8px;">Sự gắn kết giữa Mô hình AI (CatBoost) và Chính sách Kinh tế:</div>
                    <div style="margin-bottom: 8px;">• <b>Contract & tenure:</b> Hai yếu tố có trọng số SHAP cao nhất quyết định rủi ro rời bỏ, trực tiếp định hình gói <b>A3 (Ưu đãi cam kết dài hạn)</b> và <b>A2 (Chuyển đổi hợp đồng)</b>.</div>
                    <div style="margin-bottom: 8px;">• <b>InternetService, TechSupport & OnlineSecurity:</b> Nhóm đặc trưng kỹ thuật đứng tiếp theo về mức độ ảnh hưởng, làm căn cứ xây dựng gói <b>A1 (Hỗ trợ kỹ thuật & bảo mật chủ động)</b>.</div>
                    <div>• <b>MonthlyCharges & PaymentMethod:</b> Vừa tác động đến xác suất rời mạng ($p$), vừa quyết định Giá trị bảo vệ ($V = \\text{MonthlyCharges} \\times H \\times m$).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: HỒ SƠ KHÁCH HÀNG 360° & ĐỀ XUẤT CAN THIỆP
# ==============================================================================
with tab3:
    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>Bộ Lọc & Danh Sách Đề Xuất Can Thiệp</span><span class="panel-tag">CUSTOMER SEGMENTATION</span></div>', unsafe_allow_html=True)

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

        rec_with_features = recommendations.merge(
            customers_df[["customerID", "Contract", "MonthlyCharges", "tenure", "InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod"]],
            on="customerID",
            how="left"
        )

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

        st.caption(f"Tìm thấy **{len(filtered_df):,}** khách hàng thỏa mãn tiêu chí lọc.")

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
            height=240,
        )

        csv_data = filtered_df[[
            "rank", "customerID", "action_code", "action_name", "churn_prob",
            "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
        ]].to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Tải Danh Sách Đang Lọc (.CSV)",
            data=csv_data,
            file_name="retention_recommendations_filtered.csv",
            mime="text/csv",
        )

    if not filtered_df.empty:
        cust_list = filtered_df["customerID"].tolist()
        cust_options = [f"{cid}  |  Gói: {row.action_code}  |  ΔEV: ${row.net_value:+.2f}  |  Churn: {row.churn_prob:.1%}" for cid, row in zip(cust_list, filtered_df.itertuples())]

        selected_idx = st.selectbox(
            "Chọn khách hàng để phân tích chuyên sâu (Customer 360 & Local TreeSHAP):",
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

        cust_eval = evaluate_actions(
            customers=customers_df[customers_df["customerID"] == selected_cust_id],
            predictions=predictions_df[predictions_df["customer_id"] == selected_cust_id],
            actions=current_actions,
            margin=base_margin,
            horizon_months=horizon,
        )

        r1_col1, r1_col2, r1_col3 = st.columns([1, 1.15, 1.15])

        with r1_col1:
            with st.container(border=True):
                st.markdown(f'<div class="panel-title"><span>Hồ Sơ Khách Hàng: {selected_cust_id}</span><span class="panel-tag">PROFILE</span></div>', unsafe_allow_html=True)

                churn_val = float(cust_rec["churn_prob"]) * 100
                churn_after_val = float(cust_rec["churn_prob_after"]) * 100

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_val,
                    number={'suffix': "%", 'valueformat': ".1f"},
                    delta={'reference': churn_after_val, 'increasing': {'color': "#e11d48"}, 'decreasing': {'color': "#059669"}, 'position': "bottom"},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                        'bar': {'color': "#e11d48" if churn_val > 65 else ("#d97706" if churn_val > 35 else "#059669")},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 35], 'color': '#ecfdf5'},
                            {'range': [35, 65], 'color': '#fffbeb'},
                            {'range': [65, 100], 'color': '#fff1f2'},
                        ],
                        'threshold': {
                            'line': {'color': "#1e40af", 'width': 3},
                            'thickness': 0.75,
                            'value': churn_after_val
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=195,
                    margin=dict(l=15, r=15, t=15, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    annotations=[
                        dict(text="Vạch xanh: Rủi ro sau can thiệp", x=0.5, y=-0.15, showarrow=False, font=dict(size=11, color="#1e40af"))
                    ]
                )
                st.plotly_chart(fig_gauge, width="stretch")

                contract_badge = '<span class="badge-rose">Month-to-month</span>' if cust_info['Contract'] == 'Month-to-month' else f'<span class="badge-emerald">{cust_info["Contract"]}</span>'
                tech_badge = '<span class="badge-rose">Chưa có</span>' if cust_info['TechSupport'] == 'No' else '<span class="badge-emerald">Đã kích hoạt</span>'
                sec_badge = '<span class="badge-rose">Chưa có</span>' if cust_info['OnlineSecurity'] == 'No' else '<span class="badge-emerald">Đã kích hoạt</span>'
                pay_badge = '<span class="badge-amber">Thủ công</span>' if cust_info['PaymentMethod'] in MANUAL_PAYMENTS else '<span class="badge-indigo">Tự động</span>'

                st.markdown(
                    f"""
                    <div class="profile-box">
                        <div class="profile-item"><span class="profile-key">Loại Hợp Đồng</span> {contract_badge}</div>
                        <div class="profile-item"><span class="profile-key">Thời Gian Gắn Bó</span> <span class="profile-val">{cust_info['tenure']} tháng</span></div>
                        <div class="profile-item"><span class="profile-key">Cước Phí Tháng</span> <span class="profile-val" style="color:#1e40af;">${cust_info['MonthlyCharges']:.2f}</span></div>
                        <div class="profile-item"><span class="profile-key">Dịch Vụ Internet</span> <span class="profile-val">{cust_info['InternetService']}</span></div>
                        <div class="profile-item"><span class="profile-key">Hỗ Trợ Kỹ Thuật</span> {tech_badge}</div>
                        <div class="profile-item"><span class="profile-key">Bảo Mật Online</span> {sec_badge}</div>
                        <div class="profile-item"><span class="profile-key">Thanh Toán</span> {pay_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with r1_col2:
            with st.container(border=True):
                st.markdown('<div class="panel-title"><span>Quyết Định Tối Ưu & Điểm Hòa Vốn</span><span class="panel-tag">DECISION TICKET</span></div>', unsafe_allow_html=True)

                best_code = cust_rec["action_code"]
                best_name = cust_rec["action_name"]
                best_ev = cust_rec["net_value"]
                proxy_val = cust_rec["proxy_value"]
                breakeven = cust_rec["breakeven_prob"]

                if best_code != "A0":
                    st.markdown(
                        f"""
                        <div class="recommend-box">
                            <div class="recommend-title">Gói Đề Xuất: {best_code} — {best_name}</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                                <div style="background: #ffffff; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">LỢI NHUẬN RÒNG (ΔEV)</div>
                                    <div style="font-size: 1.25rem; font-weight: 700; color: #059669; font-family: 'Fira Code', monospace;">+${best_ev:.2f}</div>
                                </div>
                                <div style="background: #ffffff; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">GIÁ TRỊ BẢO VỆ (V)</div>
                                    <div style="font-size: 1.25rem; font-weight: 700; color: #1e40af; font-family: 'Fira Code', monospace;">${proxy_val:.2f}</div>
                                </div>
                            </div>
                            <div style="padding: 12px 14px; background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; line-height: 1.65; font-size: 0.85rem; color: #334155;">
                                <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">Căn cứ kinh tế & điều kiện hòa vốn:</div>
                                <div>• Ngưỡng rủi ro hòa vốn tối thiểu: <b style="font-family:'Fira Code',monospace;">p* = {breakeven:.1%}</b></div>
                                <div>• Rủi ro rời bỏ hiện tại: <b style="font-family:'Fira Code',monospace;">p = {churn_val:.1f}%</b></div>
                                <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #f1f5f9; color: #047857; font-weight: 600;">
                                    Vì p ({churn_val:.1f}%) &gt; p* ({breakeven:.1%}), phương án can thiệp mang lại kỳ vọng lời ròng +${best_ev:.2f}.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="recommend-box" style="border-left-color:#94a3b8;">
                            <div class="recommend-title" style="color:#475569;">Đề Xuất: Gói A0 — Không Can Thiệp</div>
                            <div style="color:#64748b; font-size: 0.86rem; line-height: 1.6;">
                                Tất cả phương án có chi phí đều cho lợi nhuận ròng âm (ΔEV ≤ 0) hoặc rủi ro rời bỏ thấp hơn ngưỡng hòa vốn (p &lt; p*). Giữ nguyên trạng thái để bảo toàn ngân sách.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with r1_col3:
            with st.container(border=True):
                st.markdown('<div class="panel-title"><span>Giải Thích AI Cá Nhân (Local TreeSHAP)</span><span class="panel-tag">SHAP DRIVERS</span></div>', unsafe_allow_html=True)

                local_shap_df = None
                if cached_bp is not None:
                    try:
                        cust_single_df = customers_df[customers_df["customerID"] == selected_cust_id]
                        local_shap_df = explain_single_customer_shap(cached_bp, cust_single_df)
                    except Exception:
                        local_shap_df = None

                if local_shap_df is not None and not local_shap_df.empty:
                    top_local = local_shap_df.head(6).iloc[::-1]
                    bar_colors = ["#e11d48" if v > 0 else "#059669" for v in top_local["shap_value"]]
                    fig_drivers = go.Figure(go.Bar(
                        x=top_local["shap_value"],
                        y=[f"{r.feature} ({r.value})" for r in top_local.itertuples()],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.2f}" for v in top_local["shap_value"]],
                        textposition="auto",
                    ))
                    fig_drivers.update_layout(
                        height=255,
                        margin=dict(l=10, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Đóng góp TreeSHAP (+ tăng rủi ro / - giảm rủi ro)", gridcolor="rgba(148,163,184,0.22)"),
                    )
                    st.plotly_chart(fig_drivers, width="stretch")
                    st.caption("Thanh đỏ đẩy rủi ro rời mạng tăng lên; thanh xanh giúp giữ chân khách hàng.")
                else:
                    st.info("Đang ở chế độ mô phỏng không có SHAP.")

        r2_col1, r2_col2 = st.columns([1.35, 1])

        with r2_col1:
            with st.container(border=True):
                st.markdown('<div class="panel-title"><span>So Sánh Giá Trị Ròng (ΔEV) Từng Phương Án Hợp Lệ</span><span class="panel-tag">ACTION MATRIX</span></div>', unsafe_allow_html=True)

                fig_action_comp = go.Figure()
                colors_comp = ["#059669" if code == best_code and best_code != "A0" else ("#94a3b8" if code == "A0" else "#1e40af") for code in cust_eval["action_code"]]

                fig_action_comp.add_trace(go.Bar(
                    x=cust_eval["action_code"],
                    y=cust_eval["net_value"],
                    text=[f"${v:+.2f}" for v in cust_eval["net_value"]],
                    textposition="auto",
                    marker_color=colors_comp,
                    hovertemplate="<b>%{x}</b><br>Giá trị ròng (ΔEV): %{y:$.2f}<extra></extra>",
                ))
                fig_action_comp.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Lợi nhuận ròng ΔEV ($)", gridcolor="rgba(148,163,184,0.22)"),
                    xaxis=dict(title="Mã hành động", tickangle=0),
                )
                st.plotly_chart(fig_action_comp, width="stretch")

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
                    height=165,
                )

        with r2_col2:
            with st.container(border=True):
                st.markdown('<div class="panel-title"><span>Luận Cứ Ra Quyết Định & Phê Duyệt</span><span class="panel-tag">EXECUTION</span></div>', unsafe_allow_html=True)

                if best_code != "A0":
                    be_val = cust_rec.get("breakeven_prob", None)
                    be_text = f"{be_val:.1%}" if (be_val is not None and pd.notna(be_val)) else "N/A"
                    st.markdown(
                        f"""
                        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px; font-size: 0.86rem; line-height: 1.65; color: #334155; margin-bottom: 12px;">
                            <div style="margin-bottom: 8px;">• <b>Lý do chọn gói {best_code}:</b> Mang lại thặng dư lợi nhuận kỳ vọng ròng cao nhất (+${best_ev:.2f}) sau khi trừ chi phí liên hệ và ưu đãi (${cust_rec['cost']:.1f}).</div>
                            <div style="margin-bottom: 8px;">• <b>Điều kiện hòa vốn (p* = {be_text}):</b> Rủi ro rời bỏ hiện tại là {churn_val:.1f}%, vượt ngưỡng an toàn {be_text}.</div>
                            <div>• <b>Khuyến nghị:</b> Phê duyệt gói can thiệp để bảo toàn dòng tiền cước <b>${cust_info['MonthlyCharges']:.1f}/tháng</b>.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("Phê Duyệt & Ghi Nhận Gói Can Thiệp", key="btn_apply", type="primary", width="stretch"):
                        st.success(f"Đã ghi nhận áp dụng gói {best_code} ({cust_rec['action_name']}) cho khách hàng {selected_cust_id}.")
                else:
                    st.info("Phương án tối ưu là **A0 (Không can thiệp)** vì mọi phương án có chi phí đều mang lại lợi nhuận ròng âm (ΔEV ≤ 0).")


# ==============================================================================
# TAB 4: TỐI ƯU NGÂN SÁCH & QUẢN LÝ CHIẾN DỊCH
# ==============================================================================
with tab4:
    st.info(
        "Lưu ý thuật toán: Khi ngân sách bị giới hạn, hệ thống sử dụng heuristic tham lam ưu tiên gói có **tỷ suất ΔEV trên mỗi USD chi phí** cao nhất "
        "(nên có thể chọn gói tiết kiệm hơn như A2 thay vì A3 ở tab Hồ sơ 360° để phục vụ được nhiều khách hàng hơn). Đây là nghiệm xấp xỉ thực tế."
    )

    current_total_cost = float(interventions["cost"].sum()) if not interventions.empty else 30000.0
    sim_col_left, sim_col_right = st.columns([1, 1.45])

    with sim_col_left:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Thiết Lập Hạn Mức Ngân Sách</span><span class="panel-tag">KNAPSACK SOLVER</span></div>', unsafe_allow_html=True)

            e2e_budget = st.slider(
                "Hạn mức ngân sách B (USD):",
                min_value=1000,
                max_value=max(int(current_total_cost) + 5000, 50000),
                value=15000,
                step=1000,
                key="e2e_budget_slider",
                help="Thuật toán ưu tiên khách hàng và gói can thiệp có tỷ số ΔEV / Cost cao nhất cho tới khi chạm trần ngân sách.",
            )

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
            budget_utilization = (spent_e2e / e2e_budget * 100) if e2e_budget > 0 else 0.0

            st.metric("Ngân sách thực tiêu", f"${spent_e2e:,.2f} / ${e2e_budget:,} ({budget_utilization:.1f}%)")
            st.metric("Lợi nhuận ròng thu về (ΔEV)", f"+${earned_e2e:,.2f}")
            st.metric("Số khách hàng được can thiệp", f"{len(interv_e2e):,} khách")

    with sim_col_right:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Phân Bổ Ngân Sách & Danh Sách Ưu Tiên Hàng Đầu</span><span class="panel-tag">TOP ALLOCATIONS</span></div>', unsafe_allow_html=True)

            if not interv_e2e.empty:
                sim_action_breakdown = interv_e2e.groupby("action_code").agg(
                    n_cust=("customerID", "count"),
                    cost_sum=("cost", "sum"),
                    net_sum=("net_value", "sum"),
                ).reset_index()

                fig_sim = go.Figure(data=[
                    go.Bar(
                        name="Ngân Sách Tiêu ($)",
                        x=sim_action_breakdown["action_code"],
                        y=sim_action_breakdown["cost_sum"],
                        marker_color="#d97706",
                        text=[f"${v:,.0f}" for v in sim_action_breakdown["cost_sum"]],
                        textposition="auto",
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng Thu Về ($)",
                        x=sim_action_breakdown["action_code"],
                        y=sim_action_breakdown["net_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in sim_action_breakdown["net_sum"]],
                        textposition="auto",
                    ),
                ])
                fig_sim.update_layout(
                    barmode="group",
                    height=210,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Giá trị ($)", gridcolor="rgba(148,163,184,0.22)"),
                    xaxis=dict(title="Gói can thiệp", tickangle=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_sim, width="stretch")

                st.dataframe(
                    interv_e2e[["customerID", "action_code", "action_name", "churn_prob", "cost", "benefit", "net_value", "efficiency_ratio"]].head(8).style.format({
                        "churn_prob": "{:.1%}",
                        "cost": "${:,.2f}",
                        "benefit": "${:,.2f}",
                        "net_value": "${:,.2f}",
                        "efficiency_ratio": "{:.2f}",
                    }),
                    width="stretch",
                    height=200,
                )
            else:
                st.info("Hạn mức ngân sách quá nhỏ để thực hiện bất kỳ hành động can thiệp nào.")

    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>Chốt Kịch Bản Ngân Sách & Xuất Chiến Dịch Thực Thi (.CSV)</span><span class="panel-tag">CAMPAIGN PERSISTENCE</span></div>', unsafe_allow_html=True)

        camp_c1, camp_c2, camp_c3 = st.columns([1.4, 1, 1.1])
        with camp_c1:
            e2e_camp_name = st.text_input(
                "Tên chiến dịch:",
                value=f"Chiến dịch ngân sách ${e2e_budget:,} - {pd.Timestamp.now().strftime('%d/%m/%Y')}",
                key="e2e_camp_name_input",
            )
        with camp_c2:
            e2e_camp_ds_ref = st.text_input(
                "Tập dữ liệu áp dụng:",
                value=active_dataset_name,
                key="e2e_camp_ds_ref",
            )
        with camp_c3:
            st.write("")
            if st.button("Lưu Chiến Dịch Vào Hệ Thống", type="primary", key="btn_save_e2e_camp", width="stretch"):
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
                st.success(f"Đã lưu chiến dịch '{new_saved_camp.name}' vào cơ sở dữ liệu.")

        last_id = st.session_state.get("last_saved_camp_id")
        if last_id:
            csv_bytes = export_campaign_csv(last_id, only_interventions=True)
            st.download_button(
                label="Tải Ngay File CSV Danh Sách Khách Hàng Được Chọn (customer_id, action_code, cost, benefit, net_value)",
                data=csv_bytes,
                file_name=f"campaign_{last_id[:8]}_interventions.csv",
                mime="text/csv",
                key="e2e_csv_dl_btn",
            )

    with st.container(border=True):
        st.markdown('<div class="panel-title"><span>Lịch Sử Chiến Dịch Đã Lưu & Tải Báo Cáo CSV</span><span class="panel-tag">IMMUTABLE ARCHIVE</span></div>', unsafe_allow_html=True)

        all_campaigns = list_campaigns()
        if not all_campaigns:
            st.info("Hiện chưa có chiến dịch nào được lưu. Hãy bấm 'Lưu Chiến Dịch Vào Hệ Thống' ở trên để khởi tạo.")
        else:
            camp_options = {c.campaign_id: f"{c.name} [{c.status.value}] — Ngân sách: ${c.budget:,.0f} — Ngày: {c.created_at[:10]}" for c in all_campaigns}
            selected_camp_id = st.selectbox(
                "Chọn chiến dịch đã lưu để tra cứu hoặc tải file CSV:",
                options=list(camp_options.keys()),
                format_func=lambda x: camp_options[x],
            )
            if selected_camp_id:
                selected_camp = get_campaign(selected_camp_id)
                summary_data = selected_camp.results.get("summary", {})

                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.metric("Hạn Mức Ngân Sách", f"${selected_camp.budget:,.2f}")
                with kpi_col2:
                    st.metric("Chi Phí Thực Tiêu", f"${summary_data.get('total_cost', 0.0):,.2f}")
                with kpi_col3:
                    st.metric("Lợi Nhuận Ròng (ΔEV)", f"+${summary_data.get('total_net_value', 0.0):,.2f}")
                with kpi_col4:
                    st.metric("Trạng Thái", selected_camp.status.value)

                camp_df = export_campaign_dataframe(selected_camp.campaign_id)
                interv_df = camp_df[camp_df["action_code"] != "A0"] if not camp_df.empty and "action_code" in camp_df.columns else camp_df
                if not interv_df.empty:
                    st.dataframe(
                        interv_df.head(10).style.format({"cost": "${:,.2f}", "benefit": "${:,.2f}", "net_value": "${:,.2f}"}),
                        width="stretch",
                        height=210,
                    )
                csv_data = export_campaign_csv(selected_camp.campaign_id, only_interventions=True)
                st.download_button(
                    label=f"Xuất File CSV Chiến Dịch '{selected_camp.name}'",
                    data=csv_data,
                    file_name=f"campaign_{selected_camp.campaign_id[:8]}.csv",
                    mime="text/csv",
                    key="history_csv_dl_btn",
                )

    sens_df = sensitivity_by_margin(
        customers=customers_df,
        predictions=predictions_df,
        margins=(0.20, 0.30, 0.40, 0.50),
        actions=current_actions,
        horizon_months=horizon,
    )

    sens_col1, sens_col2 = st.columns([1.2, 1])

    with sens_col1:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Độ Nhạy Dòng Tiền Khi Biên Lợi Nhuận (m) Biến Động</span><span class="panel-tag">MARGIN SENSITIVITY</span></div>', unsafe_allow_html=True)

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_net_value"],
                name="Lợi Nhuận Ròng (ΔEV)",
                line=dict(color="#059669", width=3),
                mode="lines+markers+text",
                text=[f"${v:,.0f}" for v in sens_df["total_net_value"]],
                textposition="top center",
            ))
            fig_sens.add_trace(go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_cost"],
                name="Tổng Chi Phí",
                line=dict(color="#d97706", width=2, dash="dash"),
                mode="lines+markers",
            ))
            fig_sens.add_trace(go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_benefit"],
                name="Lợi Ích Gộp",
                line=dict(color="#1e40af", width=2),
                mode="lines+markers",
            ))
            fig_sens.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Giá trị ($)", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Biên Lợi Nhuận (m)", tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sens, width="stretch")

    with sens_col2:
        with st.container(border=True):
            st.markdown('<div class="panel-title"><span>Cơ Cấu Gói Can Thiệp Khi Margin Tăng</span><span class="panel-tag">ACTION MIX</span></div>', unsafe_allow_html=True)

            action_cols = [c for c in sens_df.columns if c.startswith("n_A")]
            fig_stacked = go.Figure()
            colors_dict = {"n_A1": "#1e40af", "n_A2": "#0284c7", "n_A3": "#059669"}
            names_dict = {"n_A1": "A1 (Kỹ thuật)", "n_A2": "A2 (Hợp đồng)", "n_A3": "A3 (Giảm cước)"}

            for ac in action_cols:
                fig_stacked.add_trace(go.Bar(
                    name=names_dict.get(ac, ac),
                    x=[f"{m:.0%}" for m in sens_df["margin"]],
                    y=sens_df[ac],
                    marker_color=colors_dict.get(ac, "#4f46e5"),
                    text=sens_df[ac],
                    textposition="auto",
                ))

            fig_stacked.update_layout(
                barmode="stack",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Số lượng khách can thiệp", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Biên Lợi Nhuận (m)", tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_stacked, width="stretch")
