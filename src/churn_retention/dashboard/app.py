"""Streamlit Dashboard for Customer Retention & Economic Optimization (Person 3).

Design System: UI/UX Pro Max (Data-Dense Enterprise Dashboard — Stripe / Linear Minimalist Aesthetic)
Chạy dashboard bằng lệnh:
    streamlit run src/churn_retention/dashboard/app.py
"""

from pathlib import Path
import html
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
from churn_retention.modules.retention.service_advisor import (
    build_service_advisory,
    service_channel_for_customer,
    summarize_extended_services,
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

# --- DESIGN SYSTEM CSS (STRIPE DESIGN.MD + UI/UX PRO MAX DATA-DENSE) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --stripe-primary: #533afd;
        --stripe-primary-deep: #4434d4;
        --stripe-primary-press: #2e2b8c;
        --stripe-primary-soft: #665efd;
        --stripe-primary-subdued: #eef0ff;
        --stripe-dark-900: #1c1e54;
        --stripe-ink: #0d253d;
        --stripe-ink-secondary: #273951;
        --stripe-ink-mute: #64748d;
        --stripe-canvas: #ffffff;
        --stripe-canvas-soft: #f6f9fc;
        --stripe-hairline: #e3e8ee;
        --stripe-ruby: #ea2261;
        --stripe-emerald: #059669;
        --stripe-amber: #d97706;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--stripe-ink);
        font-feature-settings: "ss01" on, "cv01" on;
        background-color: var(--stripe-canvas-soft);
    }

    .stApp {
        background-color: #f6f9fc !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.75rem !important;
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 0.35rem !important;
        padding-bottom: 2.2rem !important;
        max-width: 96% !important;
    }

    /* --- STRIPE SIGNATURE ATMOSPHERIC MESH + DARK CONSOLE HERO HEADER (COMPACT & ELEVATED) --- */
    .exec-header {
        background: linear-gradient(135deg, #0d253d 0%, #1c1e54 68%, #2e2b8c 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 15px 22px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: nowrap;
        gap: 16px;
        box-shadow: rgba(0, 55, 112, 0.14) 0px 6px 18px, rgba(0, 55, 112, 0.06) 0px 2px 4px;
        position: relative;
        overflow: hidden;
    }
    .exec-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #533afd 0%, #665efd 32%, #ea2261 68%, #f96bee 100%);
    }
    .exec-header::after {
        content: "";
        position: absolute;
        top: -80px;
        right: -60px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(83, 58, 253, 0.28) 0%, rgba(234, 34, 97, 0.12) 45%, transparent 70%);
        pointer-events: none;
    }
    .exec-header-left {
        flex: 1;
        min-width: 0;
        position: relative;
        z-index: 1;
    }
    .exec-eyebrow {
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #b9b9f9;
        margin-bottom: 3px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .exec-title {
        font-size: 1.32rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.025em;
        margin-bottom: 3px;
        line-height: 1.2;
    }
    .exec-subtitle {
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.78);
        line-height: 1.4;
        font-weight: 400;
    }
    .exec-meta-group {
        display: flex;
        gap: 7px;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: center;
        position: relative;
        z-index: 1;
        flex-shrink: 0;
    }
    .meta-chip {
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 9999px;
        padding: 6px 13px;
        font-size: 0.76rem;
        font-weight: 500;
        color: #f6f9fc;
        font-family: 'Fira Code', monospace;
        font-feature-settings: "tnum" on;
        backdrop-filter: blur(8px);
    }
    .meta-chip-active {
        background: rgba(5, 150, 105, 0.22);
        border: 1px solid rgba(167, 243, 208, 0.45);
        color: #6ee7b7;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-weight: 600;
    }

    /* --- STRIPE BENTO KPI GRID WITH MICRO-PROGRESS BARS --- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .modern-card {
        background: var(--stripe-canvas);
        border: 1px solid var(--stripe-hairline);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: rgba(0, 55, 112, 0.06) 0px 1px 3px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .modern-card:hover {
        border-color: #a8c3de;
        box-shadow: rgba(0, 55, 112, 0.08) 0px 8px 24px, rgba(0, 55, 112, 0.04) 0px 2px 6px;
        transform: translateY(-1px);
    }
    .modern-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .card-accent-blue::before { background: #533afd; }
    .card-accent-indigo::before { background: #4434d4; }
    .card-accent-amber::before { background: #d97706; }
    .card-accent-emerald::before { background: #059669; }
    .card-accent-ruby::before { background: #ea2261; }

    .card-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--stripe-ink-mute);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        white-space: nowrap;
        gap: 6px;
    }
    .card-icon-wrap {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .card-value {
        font-family: 'Fira Code', monospace;
        font-size: 1.68rem;
        font-weight: 700;
        color: var(--stripe-ink);
        line-height: 1.15;
        letter-spacing: -0.04em;
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" on;
        white-space: nowrap;
    }
    .kpi-progress-track {
        width: 100%;
        height: 5px;
        background: #edf2f7;
        border-radius: 9999px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .kpi-progress-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s ease;
    }
    .card-footer {
        font-size: 0.78rem;
        color: var(--stripe-ink-mute);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }

    /* --- STRIPE PILL BADGES ({rounded.pill}: 9999px) --- */
    .badge-emerald {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 2px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.72rem;
        font-family: 'Fira Code', monospace;
        font-feature-settings: "tnum" on;
        display: inline-block;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .badge-indigo {
        background: #eef0ff;
        color: #4434d4;
        border: 1px solid #c7d2fe;
        padding: 2px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.72rem;
        font-family: 'Fira Code', monospace;
        font-feature-settings: "tnum" on;
        display: inline-block;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .badge-rose {
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
        padding: 2px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.72rem;
        display: inline-block;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .badge-amber {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 2px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.72rem;
        display: inline-block;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: break-word;
    }
    .badge-slate {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        padding: 2px 9px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.72rem;
        display: inline-block;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: break-word;
    }

    /* --- NATIVE BORDERED CONTAINERS AS STRIPE LEVEL-1 ELEVATION CARDS --- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e3e8ee !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        box-shadow: rgba(0, 55, 112, 0.06) 0px 1px 3px !important;
        margin-bottom: 14px !important;
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0d253d;
        margin-bottom: 12px;
        padding-bottom: 9px;
        border-bottom: 1px solid #e3e8ee;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        letter-spacing: -0.015em;
    }
    .panel-tag {
        font-size: 0.66rem;
        font-weight: 600;
        color: #4434d4;
        background: #eef0ff;
        border: 1px solid #d8dcfb;
        padding: 2px 9px;
        border-radius: 9999px;
        font-family: 'Fira Code', monospace;
        letter-spacing: 0.04em;
    }

    /* --- CUSTOM STRIPE SAAS HTML TABLES (REPLACING GRAY ST.DATAFRAME) --- */
    .stripe-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #e3e8ee;
        border-radius: 10px;
        background: #ffffff;
        margin-top: 6px;
        margin-bottom: 10px;
    }
    .stripe-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        color: #0d253d;
    }
    .stripe-table thead tr {
        background: #f6f9fc;
        border-bottom: 1px solid #e3e8ee;
    }
    .stripe-table th {
        padding: 10px 12px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748d;
        text-align: left;
        white-space: nowrap;
    }
    .stripe-table th.num, .stripe-table td.num {
        text-align: right;
        font-family: 'Fira Code', monospace;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" on;
    }
    .stripe-table tbody tr {
        border-bottom: 1px solid #eef2f6;
        transition: background-color 0.12s ease;
    }
    .stripe-table tbody tr:last-child {
        border-bottom: none;
    }
    .stripe-table tbody tr:hover {
        background-color: #f8fafd;
    }
    .stripe-table tbody tr.optimal-row {
        background-color: #f0fdf4;
    }
    .stripe-table td {
        padding: 9px 12px;
        vertical-align: middle;
        white-space: nowrap;
    }
    .cust-id-chip {
        font-family: 'Fira Code', monospace;
        font-weight: 600;
        font-size: 0.78rem;
        color: #0d253d;
        background: #f6f9fc;
        border: 1px solid #e3e8ee;
        padding: 2px 7px;
        border-radius: 5px;
    }
    .risk-cell-wrap {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 115px;
    }
    .risk-bar-bg {
        flex: 1;
        height: 6px;
        background: #edf2f7;
        border-radius: 9999px;
        overflow: hidden;
        min-width: 52px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* --- CUSTOMER 360 IDENTITY CARD & DECISION TICKET --- */
    .profile-box {
        background: #f6f9fc;
        border: 1px solid #e3e8ee;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 8px;
    }
    .profile-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #e3e8ee;
        font-size: 0.83rem;
    }
    .profile-item:last-child {
        border-bottom: none;
    }
    .profile-key {
        color: #64748d;
        font-weight: 500;
    }
    .profile-val {
        color: #0d253d;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
        font-feature-settings: "tnum" on;
    }

    .recommend-box {
        background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
        border: 1px solid #e3e8ee;
        border-left: 4px solid #533afd;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .recommend-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0d253d;
        margin-bottom: 10px;
    }

    /* --- STRIPE PILL BUTTONS & TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #ffffff;
        padding: 5px 6px;
        border-radius: 12px;
        border: 1px solid #e3e8ee;
        margin-bottom: 16px;
        box-shadow: rgba(0, 55, 112, 0.04) 0px 1px 2px;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        border-radius: 8px !important;
        padding: 0 18px !important;
        font-size: 0.83rem;
        font-weight: 600;
        color: #64748d;
        border: none !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #533afd !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        box-shadow: rgba(83, 58, 253, 0.22) 0px 2px 6px;
    }
    .stButton > button[kind="primary"] {
        background-color: #533afd !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        box-shadow: rgba(83, 58, 253, 0.2) 0px 2px 5px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4434d4 !important;
    }
    .stDownloadButton > button {
        border-radius: 9999px !important;
        border: 1px solid #533afd !important;
        color: #533afd !important;
        font-weight: 600 !important;
        background: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _action_pill_html(code: str, name: str = "") -> str:
    label = f"{code} · {name}" if name else code
    if code == "A0":
        return f'<span class="badge-slate">{label}</span>'
    elif code == "A1":
        return f'<span class="badge-indigo">{label}</span>'
    elif code == "A2":
        return f'<span class="badge-amber">{label}</span>'
    else:
        return f'<span class="badge-emerald">{label}</span>'


def _risk_bar_html(prob: float) -> str:
    pct = max(0.0, min(100.0, float(prob) * 100.0))
    color = "#ea2261" if pct >= 65.0 else ("#d97706" if pct >= 35.0 else "#059669")
    return (
        f'<div class="risk-cell-wrap">'
        f'<span style="font-family:Fira Code,monospace;font-weight:600;color:{color};min-width:44px;">{pct:.1f}%</span>'
        f'<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{pct:.0f}%;background:{color};"></div></div>'
        f'</div>'
    )


def render_package_economics_html_table(summary_df: pd.DataFrame) -> str:
    rows_html = []
    for _, r in summary_df.iterrows():
        code = str(r["action_code"])
        name = str(r["action_name"])
        n_cust = int(r["n_customers"])
        pct = float(r["percentage"])
        cost_val = float(r["cost_sum"])
        ben_val = float(r["benefit_sum"])
        net_val = float(r["net_value_sum"])
        net_style = "color:#059669;font-weight:700;" if net_val > 0 else "color:#64748d;"
        net_str = f"+${net_val:,.0f}" if net_val > 0 else f"${net_val:,.0f}"
        bar_color = "#533afd" if code != "A0" else "#94a3b8"
        rows_html.append(
            f"<tr>"
            f"<td>{_action_pill_html(code)}</td>"
            f"<td style='font-weight:500;color:#273951;'>{name}</td>"
            f"<td class='num'>{n_cust:,}</td>"
            f"<td>"
            f"<div class='risk-cell-wrap'>"
            f"<span style='font-family:Fira Code,monospace;font-size:0.78rem;min-width:42px;'>{pct:.1f}%</span>"
            f"<div class='risk-bar-bg'><div class='risk-bar-fill' style='width:{min(100.0, pct):.0f}%;background:{bar_color};'></div></div>"
            f"</div>"
            f"</td>"
            f"<td class='num'>${cost_val:,.0f}</td>"
            f"<td class='num'>${ben_val:,.0f}</td>"
            f"<td class='num' style='{net_style}'>{net_str}</td>"
            f"</tr>"
        )
    return (
        '<div class="stripe-table-wrap"><table class="stripe-table">'
        '<thead><tr>'
        '<th>Mã</th><th>Tên Gói Chiến Lược</th><th class="num">Số Khách</th>'
        '<th>Tỷ Trọng Tệp</th><th class="num">Tổng Chi Phí (C)</th>'
        '<th class="num">Lợi Ích Gộp (B)</th><th class="num">Lợi Nhuận Ròng (ΔEV)</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table></div>'
    )


def render_priority_customers_html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    rows_html = []
    sub = df.head(max_rows)
    for idx, r in enumerate(sub.itertuples(), start=1):
        cid = str(r.customerID)
        code = str(r.action_code)
        name = str(r.action_name)
        p_before = float(r.churn_prob)
        p_after = float(getattr(r, "churn_prob_after", p_before))
        p_star = getattr(r, "breakeven_prob", None)
        p_star_str = f"{float(p_star):.1%}" if (p_star is not None and pd.notna(p_star)) else "—"
        v_val = float(getattr(r, "proxy_value", 0.0))
        c_val = float(r.cost)
        b_val = float(r.benefit)
        ev_val = float(r.net_value)
        ev_badge = f'<span class="badge-emerald">+${ev_val:,.2f}</span>' if ev_val > 0 else f'<span class="badge-slate">${ev_val:,.2f}</span>'
        rows_html.append(
            f"<tr>"
            f"<td class='num' style='color:#64748d;font-weight:600;'>#{idx:02d}</td>"
            f"<td><span class='cust-id-chip'>{cid}</span></td>"
            f"<td>{_action_pill_html(code, name)}</td>"
            f"<td>{_risk_bar_html(p_before)}</td>"
            f"<td class='num' style='color:#64748d;'>{p_star_str}</td>"
            f"<td class='num' style='color:#059669;'>{p_after:.1%}</td>"
            f"<td class='num'>${v_val:,.2f}</td>"
            f"<td class='num'>${c_val:,.2f}</td>"
            f"<td class='num'>${b_val:,.2f}</td>"
            f"<td class='num'>{ev_badge}</td>"
            f"</tr>"
        )
    return (
        '<div class="stripe-table-wrap"><table class="stripe-table">'
        '<thead><tr>'
        '<th class="num">Hạng</th><th>Mã Khách Hàng</th><th>Gói Đề Xuất</th>'
        '<th>Rủi Ro Churn (p)</th><th class="num">Hòa Vốn (p*)</th><th class="num">Sau Can Thiệp</th>'
        '<th class="num">Giá Trị (V)</th><th class="num">Chi Phí (C)</th>'
        '<th class="num">Lợi Ích (B)</th><th class="num">Lợi Nhuận (ΔEV)</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table></div>'
    )


def render_action_matrix_html_table(cust_eval: pd.DataFrame, best_code: str) -> str:
    rows_html = []
    for _, r in cust_eval.iterrows():
        code = str(r["action_code"])
        name = str(r["action_name"])
        c_val = float(r["cost"])
        b_val = float(r["benefit"])
        ev_val = float(r["net_value"])
        be_val = r.get("breakeven_prob", None)
        be_str = f"{float(be_val):.1%}" if (be_val is not None and pd.notna(be_val)) else "—"
        is_opt = (code == best_code)
        row_cls = " class='optimal-row'" if is_opt else ""
        opt_tag = ' <span class="badge-emerald">TỐI ƯU</span>' if is_opt else ""
        ev_color = "#059669" if ev_val > 0 else ("#ea2261" if ev_val < 0 else "#64748d")
        rows_html.append(
            f"<tr{row_cls}>"
            f"<td>{_action_pill_html(code)}{opt_tag}</td>"
            f"<td style='font-weight:500;'>{name}</td>"
            f"<td class='num'>${c_val:,.2f}</td>"
            f"<td class='num'>${b_val:,.2f}</td>"
            f"<td class='num' style='color:{ev_color};font-weight:700;'>{'+$' if ev_val >= 0 else '-$'}{abs(ev_val):,.2f}</td>"
            f"<td class='num'>{be_str}</td>"
            f"</tr>"
        )
    return (
        '<div class="stripe-table-wrap"><table class="stripe-table">'
        '<thead><tr>'
        '<th>Mã</th><th>Phương Án Hợp Lệ</th><th class="num">Chi Phí (C)</th>'
        '<th class="num">Lợi Ích (B)</th><th class="num">Giá Trị Ròng (ΔEV)</th><th class="num">Hòa Vốn (p*)</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table></div>'
    )


def render_knapsack_allocations_html_table(interv_df: pd.DataFrame, max_rows: int = 10) -> str:
    rows_html = []
    has_prob = "churn_prob" in interv_df.columns
    for idx, (_, r) in enumerate(interv_df.head(max_rows).iterrows(), start=1):
        cid = str(r.get("customerID", r.get("customer_id", "N/A")))
        code = str(r.get("action_code", "A0"))
        name = str(r.get("action_name", ""))
        p_val = float(r.get("churn_prob", 0.0)) if has_prob else None
        c_val = float(r.get("cost", 0.0))
        b_val = float(r.get("benefit", 0.0))
        ev_val = float(r.get("net_value", 0.0))
        eff = float(r.get("efficiency_ratio", (ev_val / c_val) if c_val > 0 else 0.0))
        prob_cell = f"<td>{_risk_bar_html(p_val)}</td>" if p_val is not None else "<td style='color:#64748d;'>—</td>"
        rows_html.append(
            f"<tr>"
            f"<td class='num' style='color:#64748d;'>#{idx:02d}</td>"
            f"<td><span class='cust-id-chip'>{cid}</span></td>"
            f"<td>{_action_pill_html(code, name)}</td>"
            f"{prob_cell}"
            f"<td class='num'>${c_val:,.2f}</td>"
            f"<td class='num'>${b_val:,.2f}</td>"
            f"<td class='num'><span class='badge-emerald'>+${ev_val:,.2f}</span></td>"
            f"<td class='num'><span class='badge-indigo'>{eff:.2f}x</span></td>"
            f"</tr>"
        )
    return (
        '<div class="stripe-table-wrap"><table class="stripe-table">'
        '<thead><tr>'
        '<th class="num">#</th><th>Mã Khách Hàng</th><th>Gói Phân Bổ</th>'
        '<th>Xác Suất Churn</th><th class="num">Chi Phí (C)</th><th class="num">Lợi Ích (B)</th>'
        '<th class="num">Lợi Nhuận (ΔEV)</th><th class="num">Hiệu Suất (ΔEV/C)</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table></div>'
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
    _ds_caption_ph = st.empty()
    _ds_caption_ph.caption(f"• Tập dữ liệu: **`{active_dataset_name}`** ({len(customers_df):,} bản ghi)")
    st.caption(f"• Mô hình: **{'CatBoost' if bundle_ready else predictor_info}**")

    with st.expander("Đổi file CSV / Hạ tầng hệ thống", expanded=False):
        uploaded_file = st.file_uploader(
            "Tải lên file CSV khách hàng mới (cùng cấu trúc Telco):",
            type=["csv"],
            key="sidebar_csv_uploader",
        )
        uploaded_active_path = None
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
                    _up_target = ROOT_DIR / "data" / "_uploaded_active.csv"
                    _up_target.parent.mkdir(parents=True, exist_ok=True)
                    customers_df.to_csv(_up_target, index=False)
                    uploaded_active_path = _up_target
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
                    _ds_caption_ph.caption(f"• Tập dữ liệu: **`{active_dataset_name}`** ({len(customers_df):,} bản ghi)")
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
        force_new_scoring = st.checkbox(
            "Tạo lượt chạy mới (bỏ qua khóa chống trùng cũ)",
            value=False,
            key="chk_force_new_scoring",
            help="Mặc định khóa idempotency chặn chạy lặp lại cùng một file. Tích ô này nếu muốn ép tạo Job mới.",
        )
        if st.button("Chạy & Ghi Nhận Scoring Job", key="btn_run_scoring", width="stretch", disabled=not bundle_ready):
            import time
            from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
            from churn_retention.contracts.jobs.models import JobType, TaskJob
            from churn_retention.entrypoints.worker.worker import JobWorker

            _ds_path = uploaded_active_path or next(
                (p for p in [
                    ROOT_DIR / "data" / "raw" / "telco_churn.csv",
                    ROOT_DIR / "data" / "telco_churn.csv",
                    ROOT_DIR / "telco_churn.csv",
                ] if p.exists()),
                None,
            )
            _suffix = f"-{int(time.time())}" if force_new_scoring else ""
            _run_id = f"run-batch-{active_dataset_name}-{len(customers_df)}-{active_model_version[:10]}{_suffix}"
            _idem_key = f"scoring:{active_dataset_name}:{len(customers_df)}:{active_model_version}{_suffix}"
            _queue = SqliteTaskQueue()
            _existing_job = _queue.get_by_idempotency_key(_idem_key)
            if _existing_job is not None:
                _res = _existing_job.result or {}
                _n_stored = int(_res.get("total_stored_in_db", len(customers_df)))
                st.info(
                    f"File `{active_dataset_name}` ({len(customers_df):,} KH) đã có Job `{_existing_job.job_id[:6]}` "
                    f"(trạng thái **{_existing_job.status.value}**, đã lưu **{_n_stored:,}** dự đoán). "
                    "Khóa chống trùng (`idempotency_key`) chặn ghi đè. Tích ô phía trên nếu muốn tạo lượt chạy mới."
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
                    {
                        "Job": j.job_id[:6],
                        "Số KH": (j.result or {}).get("total_customers", "—"),
                        "Trạng thái": j.status.value,
                        "Retry": f"{j.retry_count}/{j.max_retries}",
                    }
                    for j in _q_list
                ]),
                width="stretch",
                hide_index=True,
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
    <div class="exec-header" style="margin-bottom: 8px;">
        <div class="exec-header-left">
            <div class="exec-eyebrow">TELECOM RETENTION DECISION SUPPORT SYSTEM · KHOA HỌC DỊCH VỤ</div>
            <div class="exec-title">Hệ Thống Hỗ Trợ Quyết Định &amp; Giữ Chân Khách Hàng</div>
            <div class="exec-subtitle">
                Dự báo rủi ro rời mạng bằng CatBoost, phân tích yếu tố tác động (TreeSHAP) và tối ưu hóa lợi nhuận ròng giữ chân (ΔEV).
            </div>
        </div>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; background:#ffffff; border:1px solid #e3e8ee; border-radius:10px; padding:8px 16px; margin-bottom:14px; font-size:0.8rem; color:#273951;">
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <span style="font-weight:700; color:#0d253d; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.05em;">Kịch bản kinh tế đang áp dụng:</span>
            <span class="badge-indigo">Biên lợi nhuận m = {base_margin:.0%}</span>
            <span class="badge-indigo">Kỳ hạn đánh giá H = {horizon} tháng</span>
            <span class="badge-slate">{len(customers_df):,} hồ sơ đang nạp</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span class="badge-emerald">{SVG_CHECK} CatBoost Model Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 4 TABS ĐIỀU HƯỚNG (KHÔNG EMOJI LÒE LOẸT) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "01. Tổng Quan Kinh Tế",
    "02. Hành Vi Churn & Giải Thích Mô Hình",
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

    cost_to_benefit_pct = min(100.0, (total_cost / total_benefit * 100.0)) if total_benefit > 0 else 0.0
    net_to_benefit_pct = min(100.0, (total_net_value / total_benefit * 100.0)) if total_benefit > 0 else 0.0
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="modern-card card-accent-blue">
                <div class="card-label">
                    <span>Khách Hàng Hoạt Động</span>
                    <div class="card-icon-wrap" style="background:#eef0ff; color:#533afd;">{SVG_USERS}</div>
                </div>
                <div class="card-value">{n_total:,}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:100%; background:#533afd;"></div></div>
                <div class="card-footer">
                    <span>Toàn bộ danh mục quản lý</span>
                    <span class="badge-indigo">100% Active</span>
                </div>
            </div>
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Quyết Định Can Thiệp</span>
                    <div class="card-icon-wrap" style="background:#eef0ff; color:#4434d4;">{SVG_TARGET}</div>
                </div>
                <div class="card-value">{n_intervened:,}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{intervention_rate * 100:.1f}%; background:#4434d4;"></div></div>
                <div class="card-footer">
                    <span>Tỷ lệ khách có ΔEV &gt; 0</span>
                    <span class="badge-indigo">{intervention_rate:.1%}</span>
                </div>
            </div>
            <div class="modern-card card-accent-amber">
                <div class="card-label">
                    <span>Ngân Sách Đầu Tư (C)</span>
                    <div class="card-icon-wrap" style="background:#fffbeb; color:#d97706;">{SVG_WALLET}</div>
                </div>
                <div class="card-value">${total_cost:,.0f}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{cost_to_benefit_pct:.1f}%; background:#d97706;"></div></div>
                <div class="card-footer">
                    <span>Chi phí TB / khách</span>
                    <span class="badge-amber">${avg_cost_per_user:,.1f}/KH</span>
                </div>
            </div>
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Lợi Nhuận Ròng (ΔEV)</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">{SVG_TRENDING}</div>
                </div>
                <div class="card-value" style="color: #059669;">+${total_net_value:,.0f}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{net_to_benefit_pct:.1f}%; background:#059669;"></div></div>
                <div class="card-footer">
                    <span>Gross: <strong>${total_benefit:,.0f}</strong></span>
                    <span class="badge-emerald">ROI +{roi:.1f}%</span>
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
            flow_values = [total_benefit, -total_contact_cost, -total_offer_cost, total_net_value]
            flow_colors = ["#1e40af", "#e11d48", "#d97706", "#059669"]
            flow_texts = [
                f"+${total_benefit:,.0f}",
                f"-${total_contact_cost:,.0f}",
                f"-${total_offer_cost:,.0f}",
                f"+${total_net_value:,.0f}"
            ]

            min_flow_x = min(0.0, min(flow_values)) if flow_values else -100.0
            max_flow_x = max(1000.0, max(flow_values)) if flow_values else 1000.0
            fig_cashflow = go.Figure(go.Bar(
                y=flow_labels,
                x=flow_values,
                orientation="h",
                text=flow_texts,
                textposition="outside",
                textangle=0,
                cliponaxis=False,
                textfont=dict(family="Fira Code, monospace", size=12, color="#0d253d"),
                marker=dict(color=flow_colors, cornerradius=4),
                hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
            ))
            fig_cashflow.update_layout(
                height=310,
                margin=dict(l=10, r=75, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="Giá trị tài chính ($)",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="#64748b",
                    gridcolor="rgba(148,163,184,0.22)",
                    range=[min_flow_x * 1.42 - max_flow_x * 0.08, max_flow_x * 1.24],
                ),
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
                        textposition="outside", textangle=0, cliponaxis=False,
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng (ΔEV)",
                        x=interv_actions["action_code"],
                        y=interv_actions["net_value_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in interv_actions["net_value_sum"]],
                        textposition="outside", textangle=0, cliponaxis=False,
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
            st.markdown(render_package_economics_html_table(summary_df), unsafe_allow_html=True)

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
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e3e8ee; border-left:4px solid #533afd; border-radius:10px; padding:18px 22px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; border-bottom:1px solid #e3e8ee; padding-bottom:10px; margin-bottom:14px;">
                    <div>
                        <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#533afd;">EXECUTIVE SUMMARY REPORT</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#0d253d;">Báo Cáo Tóm Tắt Chiến Lược Giữ Chân Khách Hàng</div>
                    </div>
                    <div style="display:flex; gap:6px;">
                        <span class="badge-indigo">Biên lợi nhuận m = {base_margin:.0%}</span>
                        <span class="badge-slate">Kỳ hạn H = {horizon} tháng</span>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:0.84rem; line-height:1.65; color:#273951;">
                    <div style="background:#f6f9fc; border:1px solid #e3e8ee; border-radius:8px; padding:12px 16px;">
                        <div style="font-weight:700; font-size:0.86rem; color:#0d253d; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.03em;">1. Kết Quả Điều Hành Tổng Thể</div>
                        <div>• <b>Quy mô tệp khách hàng hoạt động:</b> <span style="font-family:Fira Code,monospace;font-weight:600;">{n_total:,}</span> khách hàng</div>
                        <div>• <b>Quy mô can thiệp được duyệt (ΔEV &gt; 0):</b> <span style="font-family:Fira Code,monospace;font-weight:600;color:#4434d4;">{n_intervened:,}</span> khách ({intervention_rate:.1%} tổng tệp)</div>
                        <div>• <b>Tổng ngân sách đề xuất đầu tư:</b> <span style="font-family:Fira Code,monospace;font-weight:600;color:#d97706;">&#36;{total_cost:,.2f}</span> (TB: <span style="font-family:Fira Code,monospace;">&#36;{avg_cost_per_user:,.2f}</span>/khách)</div>
                        <div>• <b>Doanh thu giữ lại dự kiến (Gross B):</b> <span style="font-family:Fira Code,monospace;font-weight:600;">&#36;{total_benefit:,.2f}</span></div>
                        <div>• <b>Lợi nhuận ròng kỳ vọng (Net ΔEV):</b> <span style="font-family:Fira Code,monospace;font-weight:700;color:#059669;">+&#36;{total_net_value:,.2f}</span> (ROI <span class="badge-emerald">+{roi:.1f}%</span>)</div>
                    </div>
                    <div style="background:#f6f9fc; border:1px solid #e3e8ee; border-radius:8px; padding:12px 16px;">
                        <div style="font-weight:700; font-size:0.86rem; color:#0d253d; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.03em;">2. Phân Bổ Theo Gói Hành Động</div>
                        <div style="margin-bottom:6px;">• <b>Gói hiệu quả tài chính cao nhất:</b> <span class="badge-emerald">{top_action['action_code']}</span> <b>{top_action['action_name']}</b> mang lại lợi nhuận ròng <span style="font-family:Fira Code,monospace;font-weight:700;color:#059669;">+&#36;{top_action['net_value_sum']:,.2f}</span>.</div>
                        <div>• <b>Phương án không can thiệp (A0):</b> <span style="font-family:Fira Code,monospace;font-weight:600;">{a0_count:,}</span> khách hàng được giữ nguyên trạng thái do chi phí can thiệp lớn hơn lợi ích bảo vệ được (ΔEV ≤ 0), giúp tiết kiệm ngân sách.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                    <div class="card-icon-wrap" style="background:#eef0ff; color:#533afd;">{SVG_USERS}</div>
                </div>
                <div class="card-value">{eda_total:,}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:100%; background:#533afd;"></div></div>
                <div class="card-footer"><span>Toàn bộ mẫu dữ liệu Telco</span><span class="badge-indigo">100%</span></div>
            </div>
            <div class="modern-card card-accent-ruby">
                <div class="card-label">
                    <span>Tỷ Lệ Churn Toàn Mạng</span>
                    <div class="card-icon-wrap" style="background:#fff1f2; color:#ea2261;">{SVG_ALERT}</div>
                </div>
                <div class="card-value" style="color: #ea2261;">{churn_pct:.1f}%</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(100.0, churn_pct):.1f}%; background:#ea2261;"></div></div>
                <div class="card-footer"><span>Số lượng rời bỏ</span><span class="badge-rose">{churn_count:,} KH</span></div>
            </div>
            <div class="modern-card card-accent-indigo">
                <div class="card-label">
                    <span>Cước Trung Bình Tháng</span>
                    <div class="card-icon-wrap" style="background:#eef0ff; color:#4434d4;">{SVG_WALLET}</div>
                </div>
                <div class="card-value">${avg_mc:.1f}</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(100.0, avg_mc / 1.2):.1f}%; background:#4434d4;"></div></div>
                <div class="card-footer"><span>ARPU / MonthlyCharges</span><span class="badge-indigo">USD/Tháng</span></div>
            </div>
            <div class="modern-card card-accent-emerald">
                <div class="card-label">
                    <span>Thời Gian Gắn Bó TB</span>
                    <div class="card-icon-wrap" style="background:#ecfdf5; color:#059669;">{SVG_CLOCK}</div>
                </div>
                <div class="card-value">{avg_tenure:.1f}M</div>
                <div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{min(100.0, avg_tenure / 72.0 * 100.0):.1f}%; background:#059669;"></div></div>
                <div class="card-footer"><span>Chu kỳ gắn bó</span><span class="badge-emerald">~{avg_tenure / 12:.1f} năm</span></div>
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
                textposition="outside", textangle=0, cliponaxis=False,
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
                textposition="outside", textangle=0, cliponaxis=False,
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
                textposition="outside", textangle=0, cliponaxis=False,
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
                textposition="outside", textangle=0, cliponaxis=False,
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
        st.markdown('<div class="panel-title"><span>Giải Thích Mô Hình AI: Top 10 Đặc Trưng Ảnh Hưởng Nhất (CatBoost Feature Importances)</span><span class="panel-tag">MODEL FEATURE IMPORTANCE</span></div>', unsafe_allow_html=True)

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
                    textposition="outside", textangle=0, cliponaxis=False,
                ))
                fig_shap_bar.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Tầm quan trọng tương đối (%)", gridcolor="rgba(148,163,184,0.22)"),
                )
                st.plotly_chart(fig_shap_bar, width="stretch")
                st.caption("Biểu đồ phản ánh mức độ quan trọng toàn cục của các đặc trưng từ mô hình CatBoost; phân tích đóng góp riêng cho từng khách hàng xem tại Tab 03.")
            else:
                st.info("Biểu đồ Feature Importance yêu cầu Model Bundle CatBoost thật.")

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

        st.markdown(render_priority_customers_html_table(filtered_df, max_rows=10), unsafe_allow_html=True)
        with st.expander("Mở bảng dữ liệu tương tác toàn phần (Sắp xếp / Tra cứu toàn bộ danh sách)", expanded=False):
            filtered_display = filtered_df[[
                "rank", "customerID", "action_code", "action_name", "churn_prob",
                "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
            ]].copy().reset_index(drop=True)
            filtered_display["rank"] = range(1, len(filtered_display) + 1)
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
                height=220,
            )

        _export_source = filtered_df[[
            "rank", "customerID", "action_code", "action_name", "churn_prob",
            "proxy_value", "cost", "benefit", "net_value", "breakeven_prob", "churn_prob_after"
        ]].copy()
        _cust_lookup = customers_df.set_index("customerID")
        _export_source["kenh_cham_soc"] = [
            service_channel_for_customer(str(row.action_code), float(row.churn_prob))
            for row in _export_source.itertuples()
        ]
        _export_source["dich_vu_mo_rong_goi_y"] = [
            summarize_extended_services(_cust_lookup.loc[row.customerID].to_dict())
            if row.customerID in _cust_lookup.index
            else ""
            for row in _export_source.itertuples()
        ]
        csv_data = _export_source.to_csv(index=False).encode('utf-8')

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
            "Chọn khách hàng cần phân tích chi tiết:",
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

                _delta_val = churn_after_val - churn_val
                _delta_color = "#059669" if _delta_val <= 0 else "#e11d48"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=churn_val,
                    domain={'x': [0, 1], 'y': [0.12, 1]},
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
                    margin=dict(l=22, r=22, t=18, b=28),
                    paper_bgcolor="rgba(0,0,0,0)",
                    annotations=[
                        dict(
                            text=f"<b>{churn_val:.1f}%</b>",
                            x=0.5, y=0.34, xanchor="center", yanchor="middle",
                            showarrow=False,
                            font=dict(size=26, color="#0d253d", family="Fira Code"),
                        ),
                        dict(
                            text=f"Sau can thiệp: <b>{churn_after_val:.1f}%</b> ({_delta_val:+.1f}%)",
                            x=0.5, y=0.13, xanchor="center", yanchor="middle",
                            showarrow=False,
                            font=dict(size=11, color=_delta_color),
                        ),
                        dict(
                            text="Vạch xanh: Rủi ro sau can thiệp",
                            x=0.5, y=-0.14, xanchor="center", yanchor="middle",
                            showarrow=False,
                            font=dict(size=11, color="#1e40af"),
                        ),
                    ]
                )
                st.plotly_chart(fig_gauge, width="stretch", config={"displayModeBar": False})

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
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); gap: 8px; margin-bottom: 12px;">
                                <div style="background: #ffffff; padding: 10px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                                    <div style="font-size: 0.68rem; color: #64748b; font-weight: 700; white-space: nowrap;">LỢI NHUẬN RÒNG (ΔEV)</div>
                                    <div style="font-size: 1.08rem; font-weight: 700; color: #059669; font-family: 'Fira Code', monospace; white-space: nowrap;">+${best_ev:.2f}</div>
                                </div>
                                <div style="background: #ffffff; padding: 10px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                                    <div style="font-size: 0.68rem; color: #64748b; font-weight: 700; white-space: nowrap;">GIÁ TRỊ BẢO VỆ (V)</div>
                                    <div style="font-size: 1.08rem; font-weight: 700; color: #1e40af; font-family: 'Fira Code', monospace; white-space: nowrap;">${proxy_val:.2f}</div>
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
                st.markdown('<div class="panel-title"><span>Yếu Tố Tác Động Rủi Ro Đến Cá Nhân</span><span class="panel-tag">RISK DRIVERS</span></div>', unsafe_allow_html=True)

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
                    min_x = min(0.0, float(top_local["shap_value"].min()))
                    max_x = max(0.0, float(top_local["shap_value"].max()))
                    span = max(max_x - min_x, 0.25)
                    _short_feat = {
                        "MonthlyCharges": "Cước tháng",
                        "tenure": "Số tháng",
                        "Contract": "Hợp đồng",
                        "InternetService": "Internet",
                        "TechSupport": "TechSupport",
                        "OnlineSecurity": "OnlineSec",
                        "PaymentMethod": "Thanh toán",
                        "PaperlessBilling": "Hóa đơn ĐT",
                        "MultipleLines": "Đa đường dây",
                        "OnlineBackup": "Backup",
                        "DeviceProtection": "Thiết bị",
                        "SeniorCitizen": "Cao tuổi",
                    }
                    _short_val = {
                        "Month-to-month": "Tháng",
                        "One year": "1 năm",
                        "Two year": "2 năm",
                        "Fiber optic": "Cáp quang",
                        "Electronic check": "E-check",
                        "Mailed check": "Bưu điện",
                        "Bank transfer (automatic)": "CK tự động",
                        "Credit card (automatic)": "Thẻ tự động",
                        "No internet service": "Không net",
                        "No phone service": "Không ĐT",
                    }
                    y_labels = [
                        f"{_short_feat.get(str(r.feature), str(r.feature))} ({_short_val.get(str(r.value), str(r.value))}) [{r.shap_value:+.2f}]"
                        for r in top_local.itertuples()
                    ]
                    fig_drivers = go.Figure(go.Bar(
                        x=top_local["shap_value"],
                        y=y_labels,
                        orientation="h",
                        marker_color=bar_colors,
                        hovertemplate="<b>%{y}</b><br>TreeSHAP: %{x:+.3f}<extra></extra>",
                    ))
                    fig_drivers.update_layout(
                        height=265,
                        margin=dict(l=4, r=14, t=10, b=36),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(automargin=True, tickfont=dict(size=11)),
                        xaxis=dict(
                            title="TreeSHAP (+ / -)",
                            title_font=dict(size=11),
                            range=[min_x - span * 0.12, max_x + span * 0.12],
                            zeroline=True,
                            zerolinecolor="#cbd5e1",
                            gridcolor="rgba(148,163,184,0.22)",
                        ),
                    )
                    st.plotly_chart(fig_drivers, width="stretch")
                    st.caption("Màu đỏ (+): làm tăng nguy cơ rời mạng · Màu xanh (−): giúp giảm nguy cơ.")
                else:
                    st.info("Đang ở chế độ mô phỏng không có SHAP.")

        advisory = build_service_advisory(
            customer_row=cust_info.to_dict(),
            recommendation_row=cust_rec.to_dict(),
            local_shap_df=local_shap_df,
        )
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">'
                '<span>Gợi Ý Hành Động Cho Nhân Viên CSKH</span>'
                '<span class="panel-tag">CSKH ACTION CARD</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            import re as _re
            adv_c1, adv_c2, adv_c3 = st.columns([1.15, 1.05, 1.15])

            clean_drivers = [
                _re.sub(r"\s*\(SHAP [^)]+\)", "", d).strip()
                for d in advisory.risk_drivers_vi[:3]
            ]
            drivers_html = "".join(f"<li>{html.escape(d)}</li>" for d in clean_drivers)

            _pkg_label = (
                f"{cust_rec['action_code']} — {cust_rec['action_name']} (ΔEV +${cust_rec['net_value']:.2f})"
                if cust_rec["action_code"] != "A0"
                else "A0 — Không cần can thiệp ưu đãi (rủi ro thấp)"
            )

            if advisory.extended_services:
                ext_html = "".join(
                    f"<div style='margin-bottom:6px;'>"
                    f"<div style='font-weight:600;color:#0d253d;font-size:0.82rem;'>• {html.escape(ext.title)}</div>"
                    f"<div style='font-size:0.78rem;color:#475569;padding-left:10px;'>{html.escape(ext.description)}</div>"
                    f"</div>"
                    for ext in advisory.extended_services[:2]
                )
            else:
                ext_html = "<div style='font-size:0.81rem;color:#64748d;'>Tập trung xử lý gói giữ chân chính theo đề xuất bên cạnh.</div>"

            with adv_c1:
                st.markdown(
                    f"""
                    <div style="background:#ffffff;border:1px solid #e3e8ee;border-radius:10px;padding:13px 15px;height:100%;">
                        <div style="font-size:0.73rem;font-weight:700;text-transform:uppercase;color:#ea2261;margin-bottom:6px;">
                            1. Nguyên Nhân Rủi Ro Chính
                        </div>
                        <ul style="margin:0 0 0 16px;padding:0;font-size:0.82rem;color:#273951;line-height:1.55;">
                            {drivers_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with adv_c2:
                st.markdown(
                    f"""
                    <div style="background:#ffffff;border:1px solid #e3e8ee;border-radius:10px;padding:13px 15px;height:100%;">
                        <div style="font-size:0.73rem;font-weight:700;text-transform:uppercase;color:#059669;margin-bottom:6px;">
                            2. Đề Xuất &amp; Kênh Liên Hệ
                        </div>
                        <div style="font-size:0.83rem;font-weight:600;color:#0f172a;margin-bottom:8px;line-height:1.45;">
                            {html.escape(_pkg_label)}
                        </div>
                        <div>
                            <span style="font-size:0.76rem;color:#64748d;">Kênh ưu tiên: </span>
                            <span class="badge-emerald">{html.escape(advisory.service_channel)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with adv_c3:
                st.markdown(
                    f"""
                    <div style="background:#f8fafc;border:1px solid #e3e8ee;border-radius:10px;padding:13px 15px;height:100%;">
                        <div style="font-size:0.73rem;font-weight:700;text-transform:uppercase;color:#4434d4;margin-bottom:6px;">
                            3. Gợi Ý Chăm Sóc Mở Rộng
                        </div>
                        {ext_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
                    textposition="outside", textangle=0, cliponaxis=False,
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

                st.markdown(render_action_matrix_html_table(cust_eval, best_code), unsafe_allow_html=True)

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
                        approved = st.session_state.setdefault("approved_tickets", {})
                        approved[selected_cust_id] = {
                            "action_code": best_code,
                            "action_name": cust_rec["action_name"],
                            "channel": advisory.service_channel,
                            "net_value": float(best_ev),
                        }
                        st.success(
                            f"Đã ghi nhận áp dụng gói {best_code} ({cust_rec['action_name']}) "
                            f"qua kênh '{advisory.service_channel}' cho khách hàng {selected_cust_id}."
                        )
                    if st.session_state.get("approved_tickets"):
                        st.caption(
                            f"Đã duyệt trong phiên: **{len(st.session_state['approved_tickets'])}** hồ sơ cá nhân "
                            "(sử dụng Tab 04 để chạy ngân sách toàn cục & xuất CSV)."
                        )
                else:
                    st.info("Phương án tối ưu là **A0 (Không can thiệp)** vì mọi phương án có chi phí đều mang lại lợi nhuận ròng âm (ΔEV ≤ 0).")


# ==============================================================================
# TAB 4: TỐI ƯU NGÂN SÁCH & QUẢN LÝ CHIẾN DỊCH
# ==============================================================================
with tab4:
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

            roi_e2e = ((earned_e2e / spent_e2e) * 100.0) if spent_e2e > 0 else 0.0
            st.markdown(
                f"""
                <div class="profile-box" style="background:#ffffff; border:1px solid #e3e8ee; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.74rem; font-weight:700; text-transform:uppercase; color:#64748d;">Giải Ngân Ngân Sách Knapsack</span>
                        <span class="badge-indigo">{budget_utilization:.1f}% Hạn mức</span>
                    </div>
                    <div style="font-family:'Fira Code',monospace; font-size:1.35rem; font-weight:700; color:#0d253d; margin-bottom:8px;">
                        ${spent_e2e:,.2f} <span style="font-size:0.88rem; color:#64748d; font-weight:500;">/ ${e2e_budget:,}</span>
                    </div>
                    <div class="kpi-progress-track" style="height:7px; margin-bottom:14px;">
                        <div class="kpi-progress-fill" style="width:{min(100.0, budget_utilization):.1f}%; background:#533afd;"></div>
                    </div>
                    <div class="profile-item">
                        <span class="profile-key">Lợi nhuận ròng thu về (ΔEV)</span>
                        <span class="badge-emerald" style="font-size:0.85rem;">+${earned_e2e:,.2f}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-key">Lợi ích gộp bảo vệ (Gross B)</span>
                        <span class="profile-val">${benefit_e2e:,.2f}</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-key">Tỷ suất sinh lời (ROI)</span>
                        <span class="profile-val" style="color:#059669;">+{roi_e2e:.1f}%</span>
                    </div>
                    <div class="profile-item">
                        <span class="profile-key">Số khách hàng được duyệt</span>
                        <span class="profile-val">{len(interv_e2e):,} khách hàng</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                        textposition="outside", textangle=0, cliponaxis=False,
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng Thu Về ($)",
                        x=sim_action_breakdown["action_code"],
                        y=sim_action_breakdown["net_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in sim_action_breakdown["net_sum"]],
                        textposition="outside", textangle=0, cliponaxis=False,
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

                st.markdown(render_knapsack_allocations_html_table(interv_e2e, max_rows=8), unsafe_allow_html=True)
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

                c_budget = float(selected_camp.budget)
                c_cost = float(summary_data.get("total_cost", 0.0))
                c_net = float(summary_data.get("total_net_value", 0.0))
                c_n = int(summary_data.get("n_intervened", 0))
                st.markdown(
                    f"""
                    <div class="kpi-grid" style="margin-top:8px; margin-bottom:12px;">
                        <div class="modern-card card-accent-blue" style="padding:14px 16px;">
                            <div class="card-label"><span>Hạn Mức Ngân Sách</span></div>
                            <div class="card-value" style="font-size:1.3rem;">${c_budget:,.2f}</div>
                        </div>
                        <div class="modern-card card-accent-amber" style="padding:14px 16px;">
                            <div class="card-label"><span>Chi Phí Thực Tiêu</span></div>
                            <div class="card-value" style="font-size:1.3rem;">${c_cost:,.2f}</div>
                        </div>
                        <div class="modern-card card-accent-emerald" style="padding:14px 16px;">
                            <div class="card-label"><span>Lợi Nhuận Ròng (ΔEV)</span></div>
                            <div class="card-value" style="font-size:1.3rem; color:#059669;">+${c_net:,.2f}</div>
                        </div>
                        <div class="modern-card card-accent-indigo" style="padding:14px 16px;">
                            <div class="card-label"><span>Trạng Thái & Quy Mô</span></div>
                            <div style="margin-top:4px;"><span class="badge-indigo">{selected_camp.status.value}</span> <span class="badge-emerald">{c_n:,} KH</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                camp_df = export_campaign_dataframe(selected_camp.campaign_id)
                interv_df = camp_df[camp_df["action_code"] != "A0"] if not camp_df.empty and "action_code" in camp_df.columns else camp_df
                if not interv_df.empty:
                    st.markdown(render_knapsack_allocations_html_table(interv_df, max_rows=8), unsafe_allow_html=True)
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
            _net_text_pos = [
                "top center" if float(nv) >= float(c) else "bottom center"
                for nv, c in zip(sens_df["total_net_value"], sens_df["total_cost"])
            ]
            fig_sens.add_trace(go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_net_value"],
                name="Lợi Nhuận Ròng (ΔEV)",
                line=dict(color="#059669", width=3),
                mode="lines+markers+text",
                text=[f"${v:,.0f}" for v in sens_df["total_net_value"]],
                textposition=_net_text_pos,
                textfont=dict(size=11, color="#047857"),
                cliponaxis=False,
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
                mode="lines+markers+text",
                text=[f"${v:,.0f}" for v in sens_df["total_benefit"]],
                textposition="top center",
                textfont=dict(size=11, color="#1e40af"),
                cliponaxis=False,
            ))
            _max_ben = float(sens_df["total_benefit"].max()) if not sens_df.empty else 1000.0
            _min_net = min(0.0, float(sens_df["total_net_value"].min()) if not sens_df.empty else 0.0)
            fig_sens.update_layout(
                height=320,
                margin=dict(l=16, r=38, t=22, b=72),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    title="Giá trị ($)",
                    range=[_min_net - max(_max_ben * 0.14, 40.0), _max_ben * 1.18 + 40.0],
                    gridcolor="rgba(148,163,184,0.22)",
                ),
                xaxis=dict(
                    title="Biên Lợi Nhuận (m)",
                    type="category",
                    tickangle=0,
                    range=[-0.35, max(len(sens_df) - 0.55, 0.5)],
                ),
                legend=dict(orientation="h", yanchor="top", y=-0.26, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_sens, width="stretch", config={"displayModeBar": False})

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
                    text=[str(int(v)) if v > 0 else "" for v in sens_df[ac]],
                    textposition="inside",
                    insidetextanchor="middle",
                    textangle=0,
                ))

            fig_stacked.update_layout(
                barmode="stack",
                height=320,
                margin=dict(l=12, r=16, t=18, b=72),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Số lượng khách can thiệp", gridcolor="rgba(148,163,184,0.22)"),
                xaxis=dict(title="Biên Lợi Nhuận (m)", type="category", tickangle=0),
                legend=dict(orientation="h", yanchor="top", y=-0.26, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_stacked, width="stretch", config={"displayModeBar": False})
