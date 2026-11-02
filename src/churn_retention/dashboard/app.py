"""Streamlit Dashboard for Customer Retention & Economic Optimization (Person 3).

Design System: UI/UX Pro Max (Data-Dense Enterprise Dashboard — Stripe / Linear Minimalist
Aesthetic)
Chạy dashboard bằng lệnh:
PYTHONPATH=src python3 -m churn_retention.dashboard.demo
"""

import hashlib
import json
import math
from dataclasses import asdict
from html import escape
from typing import cast

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from churn_retention.entrypoints.streamlit.dashboard_llm_ui import (
    CampaignStatus,
    create_campaign,
    execution_label,
    export_campaign_csv,
    export_campaign_dataframe,
    get_campaign,
    list_campaigns,
    mark_exported,
    prediction_controls,
)
from churn_retention.entrypoints.streamlit.demo_security import require_safe_runtime
from churn_retention.modules.campaigns.use_cases import safe_export_frame
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.retention.use_cases import (
    MANUAL_PAYMENTS,
    ActionConfig,
    Condition,
    evaluate_actions,
    recommend,
    sensitivity_by_margin,
)

try:
    require_safe_runtime()
except ValueError:
    st.error("Dashboard chỉ hoạt động qua launcher demo cục bộ an toàn.")
    st.stop()


# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Retention Decision Support System | Telecom Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LUCIDE SVG ICONS (CHUẨN UI/UX PRO MAX - KHÔNG DÙNG EMOJI LÒE LOẸT) ---
SVG_USERS = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circl'
    'e cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/'
    '><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
)
SVG_TARGET = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="'
    '6"/><circle cx="12" cy="12" r="2"/></svg>'
)
SVG_WALLET = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1'
    ' 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/><'
    'path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/></svg>'
)
SVG_TRENDING = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline '
    'points="16 7 22 7 22 13"/></svg>'
)
SVG_ALERT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 '
    '21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17'
    'h.01"/></svg>'
)
SVG_CLOCK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="1'
    '8" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 1'
    '2 16 14"/></svg>'
)
SVG_CHECK = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="1'
    '4" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2.5" stroke-linecap="round" stroke-linejoin="roun'
    'd"><polyline points="20 6 9 17 4 12"/></svg>'
)
SVG_LAYERS = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="1'
    '6" viewBox="0 0 24 24" fill="none" stroke="currentColor" str'
    'oke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    '><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline point'
    's="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/><'
    "/svg>"
)

# --- DESIGN SYSTEM CSS (STRIPE DESIGN.MD + UI/UX PRO MAX DATA-DENSE) ---
st.markdown(
    (
        "\n"
        "    <style>\n"
        "    /* Offline local preview: use local/system font fallback"
        "s. */\n"
        "\n"
        "    :root {\n"
        "        --stripe-primary: #533afd;\n"
        "        --stripe-primary-deep: #4434d4;\n"
        "        --stripe-primary-press: #2e2b8c;\n"
        "        --stripe-primary-soft: #665efd;\n"
        "        --stripe-primary-subdued: #eef0ff;\n"
        "        --stripe-dark-900: #1c1e54;\n"
        "        --stripe-ink: #0d253d;\n"
        "        --stripe-ink-secondary: #273951;\n"
        "        --stripe-ink-mute: #64748d;\n"
        "        --stripe-canvas: #ffffff;\n"
        "        --stripe-canvas-soft: #f6f9fc;\n"
        "        --stripe-hairline: #e3e8ee;\n"
        "        --stripe-ruby: #ea2261;\n"
        "        --stripe-emerald: #059669;\n"
        "        --stripe-amber: #d97706;\n"
        "    }\n"
        "\n"
        '    html, body, [class*="css"] {\n'
        "        font-family: 'Inter', -apple-system, BlinkMacSystemF"
        "ont, sans-serif;\n"
        "        color: var(--stripe-ink);\n"
        '        font-feature-settings: "ss01" on, "cv01" on;\n'
        "        background-color: var(--stripe-canvas-soft);\n"
        "    }\n"
        "\n"
        "    .stApp {\n"
        "        background-color: #f6f9fc !important;\n"
        "    }\n"
        "\n"
        '    header[data-testid="stHeader"] {\n'
        "        background: transparent !important;\n"
        "        height: 1.75rem !important;\n"
        "    }\n"
        "\n"
        "    .block-container {\n"
        "        padding-top: 0.35rem !important;\n"
        "        padding-bottom: 2.2rem !important;\n"
        "        max-width: 96% !important;\n"
        "    }\n"
        "\n"
        "    /* --- STRIPE SIGNATURE ATMOSPHERIC MESH + DARK CONSOLE "
        "HERO HEADER (COMPACT & ELEVATED) --- */\n"
        "    .exec-header {\n"
        "        background: linear-gradient(135deg, #0d253d 0%, #1c1"
        "e54 68%, #2e2b8c 100%);\n"
        "        border: 1px solid rgba(255, 255, 255, 0.12);\n"
        "        border-radius: 14px;\n"
        "        padding: 15px 22px;\n"
        "        margin-bottom: 14px;\n"
        "        display: flex;\n"
        "        justify-content: space-between;\n"
        "        align-items: center;\n"
        "        flex-wrap: nowrap;\n"
        "        gap: 16px;\n"
        "        box-shadow: rgba(0, 55, 112, 0.14) 0px 6px 18px, rgb"
        "a(0, 55, 112, 0.06) 0px 2px 4px;\n"
        "        position: relative;\n"
        "        overflow: hidden;\n"
        "    }\n"
        "    .exec-header::before {\n"
        '        content: "";\n'
        "        position: absolute;\n"
        "        top: 0;\n"
        "        left: 0;\n"
        "        right: 0;\n"
        "        height: 4px;\n"
        "        background: linear-gradient(90deg, #533afd 0%, #665e"
        "fd 32%, #ea2261 68%, #f96bee 100%);\n"
        "    }\n"
        "    .exec-header::after {\n"
        '        content: "";\n'
        "        position: absolute;\n"
        "        top: -80px;\n"
        "        right: -60px;\n"
        "        width: 280px;\n"
        "        height: 280px;\n"
        "        background: radial-gradient(circle, rgba(83, 58, 253"
        ", 0.28) 0%, rgba(234, 34, 97, 0.12) 45%, transparent 70%);\n"
        "        pointer-events: none;\n"
        "    }\n"
        "    .exec-header-left {\n"
        "        flex: 1;\n"
        "        min-width: 0;\n"
        "        position: relative;\n"
        "        z-index: 1;\n"
        "    }\n"
        "    .exec-eyebrow {\n"
        "        font-size: 0.66rem;\n"
        "        font-weight: 700;\n"
        "        text-transform: uppercase;\n"
        "        letter-spacing: 0.1em;\n"
        "        color: #b9b9f9;\n"
        "        margin-bottom: 3px;\n"
        "        display: inline-flex;\n"
        "        align-items: center;\n"
        "        gap: 6px;\n"
        "    }\n"
        "    .exec-title {\n"
        "        font-size: 1.32rem;\n"
        "        font-weight: 700;\n"
        "        color: #ffffff;\n"
        "        letter-spacing: -0.025em;\n"
        "        margin-bottom: 3px;\n"
        "        line-height: 1.2;\n"
        "    }\n"
        "    .exec-subtitle {\n"
        "        font-size: 0.82rem;\n"
        "        color: rgba(255, 255, 255, 0.78);\n"
        "        line-height: 1.4;\n"
        "        font-weight: 400;\n"
        "    }\n"
        "    .exec-meta-group {\n"
        "        display: flex;\n"
        "        gap: 7px;\n"
        "        flex-wrap: wrap;\n"
        "        justify-content: flex-end;\n"
        "        align-items: center;\n"
        "        position: relative;\n"
        "        z-index: 1;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "    .meta-chip {\n"
        "        background: rgba(255, 255, 255, 0.09);\n"
        "        border: 1px solid rgba(255, 255, 255, 0.18);\n"
        "        border-radius: 9999px;\n"
        "        padding: 6px 13px;\n"
        "        font-size: 0.76rem;\n"
        "        font-weight: 500;\n"
        "        color: #f6f9fc;\n"
        "        font-family: 'Fira Code', monospace;\n"
        '        font-feature-settings: "tnum" on;\n'
        "        backdrop-filter: blur(8px);\n"
        "    }\n"
        "    .meta-chip-active {\n"
        "        background: rgba(5, 150, 105, 0.22);\n"
        "        border: 1px solid rgba(167, 243, 208, 0.45);\n"
        "        color: #6ee7b7;\n"
        "        display: inline-flex;\n"
        "        align-items: center;\n"
        "        gap: 5px;\n"
        "        font-weight: 600;\n"
        "    }\n"
        "\n"
        "    /* --- STRIPE BENTO KPI GRID WITH MICRO-PROGRESS BARS --"
        "- */\n"
        "    .kpi-grid {\n"
        "        display: grid;\n"
        "        grid-template-columns: repeat(4, 1fr);\n"
        "        gap: 16px;\n"
        "        margin-bottom: 20px;\n"
        "    }\n"
        "    .modern-card {\n"
        "        background: var(--stripe-canvas);\n"
        "        border: 1px solid var(--stripe-hairline);\n"
        "        border-radius: 12px;\n"
        "        padding: 18px 20px;\n"
        "        box-shadow: rgba(0, 55, 112, 0.06) 0px 1px 3px;\n"
        "        transition: transform 0.15s ease, box-shadow 0.15s e"
        "ase, border-color 0.15s ease;\n"
        "        position: relative;\n"
        "        overflow: hidden;\n"
        "    }\n"
        "    .modern-card:hover {\n"
        "        border-color: #a8c3de;\n"
        "        box-shadow: rgba(0, 55, 112, 0.08) 0px 8px 24px, rgb"
        "a(0, 55, 112, 0.04) 0px 2px 6px;\n"
        "        transform: translateY(-1px);\n"
        "    }\n"
        "    .modern-card::before {\n"
        '        content: "";\n'
        "        position: absolute;\n"
        "        top: 0;\n"
        "        left: 0;\n"
        "        right: 0;\n"
        "        height: 3px;\n"
        "    }\n"
        "    .card-accent-blue::before { background: #533afd; }\n"
        "    .card-accent-indigo::before { background: #4434d4; }\n"
        "    .card-accent-amber::before { background: #d97706; }\n"
        "    .card-accent-emerald::before { background: #059669; }\n"
        "    .card-accent-ruby::before { background: #ea2261; }\n"
        "\n"
        "    .card-label {\n"
        "        font-size: 0.7rem;\n"
        "        font-weight: 700;\n"
        "        text-transform: uppercase;\n"
        "        letter-spacing: 0.04em;\n"
        "        color: var(--stripe-ink-mute);\n"
        "        margin-bottom: 8px;\n"
        "        display: flex;\n"
        "        justify-content: space-between;\n"
        "        align-items: center;\n"
        "        white-space: nowrap;\n"
        "        gap: 6px;\n"
        "    }\n"
        "    .card-icon-wrap {\n"
        "        width: 30px;\n"
        "        height: 30px;\n"
        "        border-radius: 8px;\n"
        "        display: flex;\n"
        "        align-items: center;\n"
        "        justify-content: center;\n"
        "    }\n"
        "    .card-value {\n"
        "        font-family: 'Fira Code', monospace;\n"
        "        font-size: 1.68rem;\n"
        "        font-weight: 700;\n"
        "        color: var(--stripe-ink);\n"
        "        line-height: 1.15;\n"
        "        letter-spacing: -0.04em;\n"
        "        margin-bottom: 8px;\n"
        "        font-variant-numeric: tabular-nums;\n"
        '        font-feature-settings: "tnum" on;\n'
        "        white-space: nowrap;\n"
        "    }\n"
        "    .kpi-progress-track {\n"
        "        width: 100%;\n"
        "        height: 5px;\n"
        "        background: #edf2f7;\n"
        "        border-radius: 9999px;\n"
        "        overflow: hidden;\n"
        "        margin-bottom: 8px;\n"
        "    }\n"
        "    .kpi-progress-fill {\n"
        "        height: 100%;\n"
        "        border-radius: 9999px;\n"
        "        transition: width 0.3s ease;\n"
        "    }\n"
        "    .card-footer {\n"
        "        font-size: 0.78rem;\n"
        "        color: var(--stripe-ink-mute);\n"
        "        display: flex;\n"
        "        align-items: center;\n"
        "        justify-content: space-between;\n"
        "        gap: 6px;\n"
        "    }\n"
        "\n"
        "    /* --- STRIPE PILL BADGES ({rounded.pill}: 9999px) --- *"
        "/\n"
        "    .badge-emerald {\n"
        "        background: #ecfdf5;\n"
        "        color: #047857;\n"
        "        border: 1px solid #a7f3d0;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.72rem;\n"
        "        font-family: 'Fira Code', monospace;\n"
        '        font-feature-settings: "tnum" on;\n'
        "        display: inline-block;\n"
        "        white-space: nowrap;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "    .badge-indigo {\n"
        "        background: #eef0ff;\n"
        "        color: #4434d4;\n"
        "        border: 1px solid #c7d2fe;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.72rem;\n"
        "        font-family: 'Fira Code', monospace;\n"
        '        font-feature-settings: "tnum" on;\n'
        "        display: inline-block;\n"
        "        white-space: nowrap;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "    .badge-rose {\n"
        "        background: #fff1f2;\n"
        "        color: #be123c;\n"
        "        border: 1px solid #fecdd3;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.72rem;\n"
        "        display: inline-block;\n"
        "        white-space: nowrap;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "    .badge-amber {\n"
        "        background: #fffbeb;\n"
        "        color: #b45309;\n"
        "        border: 1px solid #fde68a;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.72rem;\n"
        "        display: inline-block;\n"
        "        white-space: nowrap;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "    .badge-slate {\n"
        "        background: #f1f5f9;\n"
        "        color: #475569;\n"
        "        border: 1px solid #cbd5e1;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.72rem;\n"
        "        display: inline-block;\n"
        "        white-space: nowrap;\n"
        "        flex-shrink: 0;\n"
        "    }\n"
        "\n"
        "    /* --- NATIVE BORDERED CONTAINERS AS STRIPE LEVEL-1 ELEV"
        "ATION CARDS --- */\n"
        '    [data-testid="stVerticalBlockBorderWrapper"] {\n'
        "        background-color: #ffffff !important;\n"
        "        border: 1px solid #e3e8ee !important;\n"
        "        border-radius: 12px !important;\n"
        "        padding: 18px 22px !important;\n"
        "        box-shadow: rgba(0, 55, 112, 0.06) 0px 1px 3px !impo"
        "rtant;\n"
        "        margin-bottom: 14px !important;\n"
        "    }\n"
        "\n"
        "    .panel-title {\n"
        "        font-size: 0.95rem;\n"
        "        font-weight: 600;\n"
        "        color: #0d253d;\n"
        "        margin-bottom: 12px;\n"
        "        padding-bottom: 9px;\n"
        "        border-bottom: 1px solid #e3e8ee;\n"
        "        display: flex;\n"
        "        align-items: center;\n"
        "        justify-content: space-between;\n"
        "        letter-spacing: -0.015em;\n"
        "    }\n"
        "    .panel-tag {\n"
        "        font-size: 0.66rem;\n"
        "        font-weight: 600;\n"
        "        color: #4434d4;\n"
        "        background: #eef0ff;\n"
        "        border: 1px solid #d8dcfb;\n"
        "        padding: 2px 9px;\n"
        "        border-radius: 9999px;\n"
        "        font-family: 'Fira Code', monospace;\n"
        "        letter-spacing: 0.04em;\n"
        "    }\n"
        "\n"
        "    /* --- CUSTOM STRIPE SAAS HTML TABLES (REPLACING GRAY ST"
        ".DATAFRAME) --- */\n"
        "    .stripe-table-wrap {\n"
        "        width: 100%;\n"
        "        overflow-x: auto;\n"
        "        border: 1px solid #e3e8ee;\n"
        "        border-radius: 10px;\n"
        "        background: #ffffff;\n"
        "        margin-top: 6px;\n"
        "        margin-bottom: 10px;\n"
        "    }\n"
        "    .stripe-table {\n"
        "        width: 100%;\n"
        "        border-collapse: collapse;\n"
        "        font-size: 0.82rem;\n"
        "        color: #0d253d;\n"
        "    }\n"
        "    .stripe-table thead tr {\n"
        "        background: #f6f9fc;\n"
        "        border-bottom: 1px solid #e3e8ee;\n"
        "    }\n"
        "    .stripe-table th {\n"
        "        padding: 10px 12px;\n"
        "        font-size: 0.68rem;\n"
        "        font-weight: 700;\n"
        "        text-transform: uppercase;\n"
        "        letter-spacing: 0.06em;\n"
        "        color: #64748d;\n"
        "        text-align: left;\n"
        "        white-space: nowrap;\n"
        "    }\n"
        "    .stripe-table th.num, .stripe-table td.num {\n"
        "        text-align: right;\n"
        "        font-family: 'Fira Code', monospace;\n"
        "        font-variant-numeric: tabular-nums;\n"
        '        font-feature-settings: "tnum" on;\n'
        "    }\n"
        "    .stripe-table tbody tr {\n"
        "        border-bottom: 1px solid #eef2f6;\n"
        "        transition: background-color 0.12s ease;\n"
        "    }\n"
        "    .stripe-table tbody tr:last-child {\n"
        "        border-bottom: none;\n"
        "    }\n"
        "    .stripe-table tbody tr:hover {\n"
        "        background-color: #f8fafd;\n"
        "    }\n"
        "    .stripe-table tbody tr.optimal-row {\n"
        "        background-color: #f0fdf4;\n"
        "    }\n"
        "    .stripe-table td {\n"
        "        padding: 9px 12px;\n"
        "        vertical-align: middle;\n"
        "        white-space: nowrap;\n"
        "    }\n"
        "    .cust-id-chip {\n"
        "        font-family: 'Fira Code', monospace;\n"
        "        font-weight: 600;\n"
        "        font-size: 0.78rem;\n"
        "        color: #0d253d;\n"
        "        background: #f6f9fc;\n"
        "        border: 1px solid #e3e8ee;\n"
        "        padding: 2px 7px;\n"
        "        border-radius: 5px;\n"
        "    }\n"
        "    .risk-cell-wrap {\n"
        "        display: flex;\n"
        "        align-items: center;\n"
        "        gap: 8px;\n"
        "        min-width: 115px;\n"
        "    }\n"
        "    .risk-bar-bg {\n"
        "        flex: 1;\n"
        "        height: 6px;\n"
        "        background: #edf2f7;\n"
        "        border-radius: 9999px;\n"
        "        overflow: hidden;\n"
        "        min-width: 52px;\n"
        "    }\n"
        "    .risk-bar-fill {\n"
        "        height: 100%;\n"
        "        border-radius: 9999px;\n"
        "    }\n"
        "\n"
        "    /* --- CUSTOMER 360 IDENTITY CARD & DECISION TICKET --- "
        "*/\n"
        "    .profile-box {\n"
        "        background: #f6f9fc;\n"
        "        border: 1px solid #e3e8ee;\n"
        "        border-radius: 10px;\n"
        "        padding: 12px 16px;\n"
        "        margin-top: 8px;\n"
        "    }\n"
        "    .profile-item {\n"
        "        display: flex;\n"
        "        justify-content: space-between;\n"
        "        align-items: center;\n"
        "        padding: 6px 0;\n"
        "        border-bottom: 1px solid #e3e8ee;\n"
        "        font-size: 0.83rem;\n"
        "    }\n"
        "    .profile-item:last-child {\n"
        "        border-bottom: none;\n"
        "    }\n"
        "    .profile-key {\n"
        "        color: #64748d;\n"
        "        font-weight: 500;\n"
        "    }\n"
        "    .profile-val {\n"
        "        color: #0d253d;\n"
        "        font-weight: 600;\n"
        "        font-family: 'Fira Code', monospace;\n"
        '        font-feature-settings: "tnum" on;\n'
        "    }\n"
        "\n"
        "    .recommend-box {\n"
        "        background: linear-gradient(180deg, #ffffff 0%, #f6f"
        "9fc 100%);\n"
        "        border: 1px solid #e3e8ee;\n"
        "        border-left: 4px solid #533afd;\n"
        "        border-radius: 10px;\n"
        "        padding: 16px;\n"
        "        margin-bottom: 10px;\n"
        "    }\n"
        "    .recommend-title {\n"
        "        font-size: 0.95rem;\n"
        "        font-weight: 700;\n"
        "        color: #0d253d;\n"
        "        margin-bottom: 10px;\n"
        "    }\n"
        "\n"
        "    /* --- STRIPE PILL BUTTONS & TABS --- */\n"
        '    .stTabs [data-baseweb="tab-list"] {\n'
        "        gap: 6px;\n"
        "        background-color: #ffffff;\n"
        "        padding: 5px 6px;\n"
        "        border-radius: 12px;\n"
        "        border: 1px solid #e3e8ee;\n"
        "        margin-bottom: 16px;\n"
        "        box-shadow: rgba(0, 55, 112, 0.04) 0px 1px 2px;\n"
        "    }\n"
        '    .stTabs [data-baseweb="tab-highlight"],\n'
        '    .stTabs [data-baseweb="tab-border"] {\n'
        "        display: none !important;\n"
        "    }\n"
        '    .stTabs [data-baseweb="tab"] {\n'
        "        height: 36px;\n"
        "        border-radius: 8px !important;\n"
        "        padding: 0 18px !important;\n"
        "        font-size: 0.83rem;\n"
        "        font-weight: 600;\n"
        "        color: #64748d;\n"
        "        border: none !important;\n"
        "        background: transparent;\n"
        "    }\n"
        '    .stTabs [aria-selected="true"] {\n'
        "        background-color: #533afd !important;\n"
        "        color: #ffffff !important;\n"
        "        border-radius: 8px !important;\n"
        "        box-shadow: rgba(83, 58, 253, 0.22) 0px 2px 6px;\n"
        "    }\n"
        '    .stButton > button[kind="primary"] {\n'
        "        background-color: #533afd !important;\n"
        "        color: #ffffff !important;\n"
        "        border: none !important;\n"
        "        border-radius: 9999px !important;\n"
        "        padding: 8px 18px !important;\n"
        "        font-weight: 600 !important;\n"
        "        box-shadow: rgba(83, 58, 253, 0.2) 0px 2px 5px !impo"
        "rtant;\n"
        "    }\n"
        '    .stButton > button[kind="primary"]:hover {\n'
        "        background-color: #4434d4 !important;\n"
        "    }\n"
        "    .stDownloadButton > button {\n"
        "        border-radius: 9999px !important;\n"
        "        border: 1px solid #533afd !important;\n"
        "        color: #533afd !important;\n"
        "        font-weight: 600 !important;\n"
        "        background: #ffffff !important;\n"
        "    }\n"
        "    </style>\n"
        "    "
    ),
    unsafe_allow_html=True,
)


def _action_pill_html(code: str, name: str = "") -> str:
    label = escape(f"{code} · {name}" if name else code)
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
        '<div class="risk-cell-wrap"><span style="font-family:Fira Co'
        "de,monospace;font-weight:600;color:"
        f"{color}"
        ';min-width:44px;">'
        f"{pct:.1f}"
        '%</span><div class="risk-bar-bg"><div class="risk-bar-fill" '
        'style="width:'
        f"{pct:.0f}"
        "%;background:"
        f"{color}"
        ';"></div></div></div>'
    )


def render_package_economics_html_table(summary_df: pd.DataFrame) -> str:
    rows_html = []
    for _, r in summary_df.iterrows():
        code = str(r["action_code"])
        name = escape(str(r["action_name"]))
        n_cust = int(r["n_customers"])
        pct = float(r["percentage"])
        cost_val = float(r["cost_sum"])
        ben_val = float(r["benefit_sum"])
        net_val = float(r["net_value_sum"])
        net_style = "color:#059669;font-weight:700;" if net_val > 0 else "color:#64748d;"
        net_str = f"+${net_val:,.0f}" if net_val > 0 else f"${net_val:,.0f}"
        bar_color = "#533afd" if code != "A0" else "#94a3b8"
        rows_html.append(
            (
                "<tr><td>"
                f"{_action_pill_html(code)}"
                "</td><td style='font-weight:500;color:#273951;'>"
                f"{name}"
                "</td><td class='num'>"
                f"{n_cust:,}"
                "</td><td><div class='risk-cell-wrap'><span style='font-famil"
                "y:Fira Code,monospace;font-size:0.78rem;min-width:42px;'>"
                f"{pct:.1f}"
                "%</span><div class='risk-bar-bg'><div class='risk-bar-fill' "
                "style='width:"
                f"{min(100.0, pct):.0f}"
                "%;background:"
                f"{bar_color}"
                ";'></div></div></div></td><td class='num'>$"
                f"{cost_val:,.0f}"
                "</td><td class='num'>$"
                f"{ben_val:,.0f}"
                "</td><td class='num' style='"
                f"{net_style}"
                "'>"
                f"{net_str}"
                "</td></tr>"
            )
        )
    return (
        '<div class="stripe-table-wrap"><table class="stripe-table">'
        "<thead><tr>"
        '<th>Mã</th><th>Tên Gói Chiến Lược</th><th class="num">Số Khách</th>'
        '<th>Tỷ Trọng Tệp</th><th class="num">Tổng Chi Phí (C)</th>'
        '<th class="num">Lợi Ích Gộp (B)</th><th class="num">Lợi Nhuận Ròng (ΔEV)</th>'
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def render_priority_customers_html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    rows_html = []
    sub = df.head(max_rows)
    for idx, r in enumerate(sub.itertuples(), start=1):
        cid = escape(str(r.customerID))
        code = str(r.action_code)
        name = str(r.action_name)
        p_before = float(cast(float, r.churn_prob))
        p_after = float(getattr(r, "churn_prob_after", p_before))
        p_star = getattr(r, "breakeven_prob", None)
        p_star_str = f"{float(p_star):.1%}" if (p_star is not None and pd.notna(p_star)) else "—"
        v_val = float(getattr(r, "proxy_value", 0.0))
        c_val = float(cast(float, r.cost))
        b_val = float(cast(float, r.benefit))
        ev_val = float(cast(float, r.net_value))
        ev_badge = (
            f'<span class="badge-emerald">+${ev_val:,.2f}</span>'
            if ev_val > 0
            else f'<span class="badge-slate">${ev_val:,.2f}</span>'
        )
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
        (
            '<div class="stripe-table-wrap"><table class="stripe-table"><'
            'thead><tr><th class="num">Hạng</th><th>Mã Khách Hàng</th><th'
            '>Gói Đề Xuất</th><th>Rủi Ro Churn (p)</th><th class="num">Hò'
            'a Vốn (p*)</th><th class="num">Sau Can Thiệp</th><th class="'
            'num">Giá Trị (V)</th><th class="num">Chi Phí (C)</th><th cla'
            'ss="num">Lợi Ích (B)</th><th class="num">Lợi Nhuận (ΔEV)</th'
            "></tr></thead><tbody>"
        )
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def render_action_matrix_html_table(cust_eval: pd.DataFrame, best_code: str) -> str:
    rows_html = []
    for _, r in cust_eval.iterrows():
        code = str(r["action_code"])
        name = escape(str(r["action_name"]))
        c_val = float(r["cost"])
        b_val = float(r["benefit"])
        ev_val = float(r["net_value"])
        be_val = r.get("breakeven_prob", None)
        be_str = f"{float(be_val):.1%}" if (be_val is not None and pd.notna(be_val)) else "—"
        is_opt = code == best_code
        row_cls = " class='optimal-row'" if is_opt else ""
        opt_tag = ' <span class="badge-emerald">TỐI ƯU</span>' if is_opt else ""
        ev_color = "#059669" if ev_val > 0 else ("#ea2261" if ev_val < 0 else "#64748d")
        rows_html.append(
            (
                "<tr"
                f"{row_cls}"
                "><td>"
                f"{_action_pill_html(code)}"
                f"{opt_tag}"
                "</td><td style='font-weight:500;'>"
                f"{name}"
                "</td><td class='num'>$"
                f"{c_val:,.2f}"
                "</td><td class='num'>$"
                f"{b_val:,.2f}"
                "</td><td class='num' style='color:"
                f"{ev_color}"
                ";font-weight:700;'>"
                f"{('+$' if ev_val >= 0 else '-$')}"
                f"{abs(ev_val):,.2f}"
                "</td><td class='num'>"
                f"{be_str}"
                "</td></tr>"
            )
        )
    return (
        (
            '<div class="stripe-table-wrap"><table class="stripe-table"><'
            'thead><tr><th>Mã</th><th>Phương Án Hợp Lệ</th><th class="num'
            '">Chi Phí (C)</th><th class="num">Lợi Ích (B)</th><th class='
            '"num">Giá Trị Ròng (ΔEV)</th><th class="num">Hòa Vốn (p*)</t'
            "h></tr></thead><tbody>"
        )
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def render_knapsack_allocations_html_table(interv_df: pd.DataFrame, max_rows: int = 10) -> str:
    rows_html = []
    has_prob = "churn_prob" in interv_df.columns
    for idx, r in enumerate(interv_df.head(max_rows).to_dict("records"), start=1):
        cid = escape(str(r.get("customerID", r.get("customer_id", "N/A"))))
        code = str(r.get("action_code", "A0"))
        name = str(r.get("action_name", ""))
        p_val = float(r.get("churn_prob", 0.0)) if has_prob else None
        c_val = float(r.get("cost", 0.0))
        b_val = float(r.get("benefit", 0.0))
        ev_val = float(r.get("net_value", 0.0))
        eff = float(r.get("efficiency_ratio", (ev_val / c_val) if c_val > 0 else 0.0))
        prob_cell = (
            f"<td>{_risk_bar_html(p_val)}</td>"
            if p_val is not None
            else "<td style='color:#64748d;'>—</td>"
        )
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
        "<thead><tr>"
        '<th class="num">#</th><th>Mã Khách Hàng</th><th>Gói Phân Bổ</th>'
        '<th>Xác Suất Churn</th><th class="num">Chi Phí (C)</th><th class="num">Lợi Ích (B)</th>'
        '<th class="num">Lợi Nhuận (ΔEV)</th><th class="num">Hiệu Suất (ΔEV/C)</th>'
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def _archive_read(reader, *args, **kwargs):
    try:
        return reader(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- Corrupt archives must not expose raw DB/customer values.
        st.error("Không thể đọc archive an toàn. Không xóa hoặc tạo lại database tự động.")
        st.stop()


# --- LLM + RAG V2 SUPPORT SERVICE; EXPLICIT SUBMIT, PERSISTED SNAPSHOT ---
view = prediction_controls()
if view is None:
    st.stop()
customers_df, predictions_df = view.customers, view.predictions
active_dataset_name = view.run.source_name
active_dataset_id = hashlib.sha256(customers_df.to_json(orient="records").encode()).hexdigest()
active_model_version = (
    view.run.predictor_ref
)  # Configuration reference, not actual provider model version.
active_run_id = view.run.run_id
result_mode = execution_label(view.run.execution_mode)
predictor_info = f"LLM + RAG v2 · {result_mode} provider"

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
        c1_contact = st.number_input(
            "A1: Chi phí liên hệ ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=2.0,
            step=0.5,
            key="c1_c",
        )
        c1_offer = st.number_input(
            "A1: Chi phí ưu đãi ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=15.0,
            step=1.0,
            key="c1_o",
        )
        c1_u = st.slider("A1: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05, key="c1_u")
        c1_e = st.slider("A1: Mức giảm churn (e)", 0.1, 1.0, 0.25, 0.05, key="c1_e")

    with st.expander("A2: Tư vấn Hợp đồng & Thanh toán", expanded=False):
        c2_contact = st.number_input(
            "A2: Chi phí liên hệ ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=2.0,
            step=0.5,
            key="c2_c",
        )
        c2_offer = st.number_input(
            "A2: Chi phí ưu đãi ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=20.0,
            step=1.0,
            key="c2_o",
        )
        c2_u = st.slider("A2: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.35, 0.05, key="c2_u")
        c2_e = st.slider("A2: Mức giảm churn (e)", 0.1, 1.0, 0.40, 0.05, key="c2_e")

    with st.expander("A3: Ưu đãi Giảm Cước Dài Hạn", expanded=False):
        c3_contact = st.number_input(
            "A3: Chi phí liên hệ ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=3.0,
            step=0.5,
            key="c3_c",
        )
        c3_offer = st.number_input(
            "A3: Chi phí voucher ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=40.0,
            step=2.0,
            key="c3_o",
        )
        c3_u = st.slider("A3: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05, key="c3_u")
        c3_e = st.slider("A3: Mức giảm churn (e)", 0.1, 1.0, 0.50, 0.05, key="c3_e")

    st.markdown("---")
    st.markdown("### Dữ Liệu & Dịch Vụ Dự Đoán")
    st.caption(f"Tập dữ liệu đã chấm: {len(customers_df)} dòng")
    st.text(active_dataset_name)
    st.caption(predictor_info)
    st.text(f"Reference version: {view.run.reference_version}")
    st.caption("Snapshot đã lưu; thay tham số kinh tế, bộ lọc hoặc tab không gọi lại LLM.")

# Dựng lại danh mục hành động
current_actions = (
    ActionConfig("A0", "Không can thiệp", 0, 0, 0, 0, is_no_action=True),
    ActionConfig(
        "A1",
        "Hỗ trợ kỹ thuật & bảo mật chủ động",
        c1_contact,
        c1_offer,
        c1_u,
        c1_e,
        all_of=(Condition("InternetService", "ne", "No"),),
        any_of=(
            Condition("TechSupport", "eq", "No"),
            Condition("OnlineSecurity", "eq", "No"),
        ),
    ),
    ActionConfig(
        "A2",
        "Tư vấn hợp đồng & thanh toán tự động",
        c2_contact,
        c2_offer,
        c2_u,
        c2_e,
        any_of=(
            Condition("Contract", "eq", "Month-to-month"),
            Condition("PaymentMethod", "in", MANUAL_PAYMENTS),
        ),
    ),
    ActionConfig(
        "A3",
        "Ưu đãi giảm cước cam kết dài hạn",
        c3_contact,
        c3_offer,
        c3_u,
        c3_e,
        all_of=(
            Condition("Contract", "eq", "Month-to-month"),
            Condition("MonthlyCharges", "ge", 50.0),
        ),
    ),
)


try:
    if not math.isfinite(
        math.fsum(float(v) * horizon * base_margin for v in customers_df["MonthlyCharges"])
    ):
        raise ValueError("Non-finite economics")
except (ValueError, OverflowError):
    st.error("Giá trị kinh tế vượt giới hạn số hữu hạn. Chưa thể mô phỏng.")
    st.stop()

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
    (
        "\n"
        '    <div class="exec-header" style="margin-bottom: 8px;">\n'
        '        <div class="exec-header-left">\n'
        '            <div class="exec-eyebrow">TELECOM RETENTION DECI'
        "SION SUPPORT SYSTEM</div>\n"
        '            <div class="exec-title">Hệ Thống Hỗ Trợ Quyết Đị'
        "nh &amp; Giữ Chân Khách Hàng</div>\n"
        '            <div class="exec-subtitle">\n'
        "                Dashboard Cường · Dịch vụ hỗ trợ LLM + RAG v"
        "2 ("
        f"{result_mode}"
        " provider) · Mô phỏng kinh tế theo giả định, chưa hiệu chuẩn"
        ".\n"
        "            </div>\n"
        "        </div>\n"
        "    </div>\n"
        '    <div style="display:flex; align-items:center; justify-co'
        "ntent:space-between; flex-wrap:wrap; gap:10px; background:#f"
        "fffff; border:1px solid #e3e8ee; border-radius:10px; padding"
        ":8px 16px; margin-bottom:14px; font-size:0.8rem; color:#2739"
        '51;">\n'
        '        <div style="display:flex; align-items:center; gap:10'
        'px; flex-wrap:wrap;">\n'
        '            <span style="font-weight:700; color:#0d253d; tex'
        "t-transform:uppercase; font-size:0.7rem; letter-spacing:0.05"
        'em;">Kịch bản kinh tế đang áp dụng:</span>\n'
        '            <span class="badge-indigo">Biên lợi nhuận m = '
        f"{base_margin:.0%}"
        "</span>\n"
        '            <span class="badge-indigo">Kỳ hạn đánh giá H = '
        f"{horizon}"
        " tháng</span>\n"
        "        </div>\n"
        '        <div style="display:flex; align-items:center; gap:8p'
        'x;">\n'
        '            <span class="badge-emerald">'
        f"{SVG_CHECK}"
        " LLM + RAG v2 · "
        f"{result_mode}"
        " provider</span>\n"
        "        </div>\n"
        "    </div>\n"
        "    "
    ),
    unsafe_allow_html=True,
)

# --- 4 TABS ĐIỀU HƯỚNG (KHÔNG EMOJI LÒE LOẸT) ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "01. Tổng Quan Kinh Tế",
        "02. Hành Vi Churn & Giải Thích LLM",
        "03. Hồ Sơ Khách Hàng 360°",
        "04. Tối Ưu Ngân Sách & Chiến Dịch",
    ]
)


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
    roi_label = f"{roi:+.1f}%" if total_cost > 0 else "N/A (chi phí bằng 0)"

    cost_to_benefit_pct = (
        min(100.0, (total_cost / total_benefit * 100.0)) if total_benefit > 0 else 0.0
    )
    net_to_benefit_pct = (
        min(100.0, (total_net_value / total_benefit * 100.0)) if total_benefit > 0 else 0.0
    )
    st.markdown(
        (
            "\n"
            '        <div class="kpi-grid">\n'
            '            <div class="modern-card card-accent-blue">\n'
            '                <div class="card-label">\n'
            "                    <span>Khách Hàng Hoạt Động</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#eef0ff; color:#533afd;">'
            f"{SVG_USERS}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">'
            f"{n_total:,}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:100%; background:#533afd;"><'
            "/div></div>\n"
            '                <div class="card-footer">\n'
            "                    <span>Toàn bộ danh mục quản lý</span>\n"
            '                    <span class="badge-indigo">100% Active</'
            "span>\n"
            "                </div>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-indigo">\n'
            '                <div class="card-label">\n'
            "                    <span>Quyết Định Can Thiệp</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#eef0ff; color:#4434d4;">'
            f"{SVG_TARGET}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">'
            f"{n_intervened:,}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{intervention_rate * 100:.1f}"
            '%; background:#4434d4;"></div></div>\n'
            '                <div class="card-footer">\n'
            "                    <span>Tỷ lệ khách có ΔEV &gt; 0</span>\n"
            '                    <span class="badge-indigo">'
            f"{intervention_rate:.1%}"
            "</span>\n"
            "                </div>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-amber">\n'
            '                <div class="card-label">\n'
            "                    <span>Ngân Sách Đầu Tư (C)</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#fffbeb; color:#d97706;">'
            f"{SVG_WALLET}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">$'
            f"{total_cost:,.0f}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{cost_to_benefit_pct:.1f}"
            '%; background:#d97706;"></div></div>\n'
            '                <div class="card-footer">\n'
            "                    <span>Chi phí TB / khách</span>\n"
            '                    <span class="badge-amber">$'
            f"{avg_cost_per_user:,.1f}"
            "/KH</span>\n"
            "                </div>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-emerald">\n'
            '                <div class="card-label">\n'
            "                    <span>Lợi Nhuận Ròng (ΔEV)</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#ecfdf5; color:#059669;">'
            f"{SVG_TRENDING}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value" style="color: #05966'
            '9;">+$'
            f"{total_net_value:,.0f}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{net_to_benefit_pct:.1f}"
            '%; background:#059669;"></div></div>\n'
            '                <div class="card-footer">\n'
            "                    <span>Gross: <strong>$"
            f"{total_benefit:,.0f}"
            "</strong></span>\n"
            '                    <span class="badge-emerald">ROI mô phỏng'
            " "
            f"{roi_label}"
            "</span>\n"
            "                </div>\n"
            "            </div>\n"
            "        </div>\n"
            "        "
        ),
        unsafe_allow_html=True,
    )

    action_counts = recommendations["action_code"].value_counts().reset_index()
    action_counts.columns = ["action_code", "n_customers"]

    action_metrics = (
        recommendations.groupby("action_code")
        .agg(
            cost_sum=("cost", "sum"),
            benefit_sum=("benefit", "sum"),
            net_value_sum=("net_value", "sum"),
        )
        .reset_index()
    )

    summary_df = action_counts.merge(action_metrics, on="action_code", how="left")
    action_names = {a.code: a.name for a in current_actions}
    summary_df["action_name"] = summary_df["action_code"].map(action_names)
    summary_df["percentage"] = (summary_df["n_customers"] / n_total) * 100 if n_total > 0 else 0

    col_chart_left, col_chart_right = st.columns([1, 1.2])

    with col_chart_left, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Cơ Cấu Phân Bổ Hành Động Can '
                'Thiệp</span><span class="panel-tag">SHARE OF WALLET</span></'
                "div>"
            ),
            unsafe_allow_html=True,
        )

        colors_map = {
            "A0": "#94a3b8",
            "A1": "#1e40af",
            "A2": "#0284c7",
            "A3": "#059669",
        }
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        f"{row.action_code}: {row.action_name}" for _, row in summary_df.iterrows()
                    ],
                    values=summary_df["n_customers"],
                    hole=0.60,
                    marker={
                        "colors": [
                            colors_map.get(code, "#4f46e5") for code in summary_df["action_code"]
                        ]
                    },
                    textinfo="percent",
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{label}</b><br>Số lượng: %{value:,} khách<br>Tỷ lệ: %{pe"
                        "rcent}<extra></extra>"
                    ),
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.08,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 11},
            },
            height=340,
            margin={"l": 10, "r": 10, "t": 10, "b": 40},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                {
                    ("text"): (
                        "<b>"
                        f"{n_total:,}"
                        "</b><br><span style='font-size:11px;color:#64748b;'>Khách hà"
                        "ng</span>"
                    ),
                    "x": 0.5,
                    "y": 0.5,
                    "font_size": 15,
                    "showarrow": False,
                }
            ],
        )
        st.plotly_chart(fig_donut, width="stretch")

    with col_chart_right, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Cân Đối Dòng Tiền & Hiệu Quả '
                'Can Thiệp</span><span class="panel-tag">CASHFLOW BRIDGE</spa'
                "n></div>"
            ),
            unsafe_allow_html=True,
        )

        total_contact_cost = sum(
            len(recommendations[recommendations["action_code"] == a.code]) * a.contact_cost
            for a in current_actions
            if not a.is_no_action
        )
        total_offer_cost = total_cost - total_contact_cost

        flow_labels = [
            "Lợi ích gộp (Gross Benefit)",
            "Chi phí liên hệ CSKH",
            "Chi phí gói ưu đãi/voucher",
            "Lợi nhuận ròng (Net ΔEV)",
        ]
        flow_values = [
            total_benefit,
            total_contact_cost,
            total_offer_cost,
            total_net_value,
        ]
        flow_colors = ["#1e40af", "#e11d48", "#d97706", "#059669"]
        flow_texts = [
            f"+${total_benefit:,.0f}",
            f"-${total_contact_cost:,.0f}",
            f"-${total_offer_cost:,.0f}",
            f"+${total_net_value:,.0f}",
        ]

        max_flow_x = max(flow_values) if flow_values else 1000.0
        fig_cashflow = go.Figure(
            go.Bar(
                y=flow_labels,
                x=flow_values,
                orientation="h",
                text=flow_texts,
                textposition="outside",
                textangle=0,
                cliponaxis=False,
                textfont={
                    "family": "Fira Code, monospace",
                    "size": 12,
                    "color": "#0d253d",
                },
                marker={"color": flow_colors, "cornerradius": 4},
                hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
            )
        )
        fig_cashflow.update_layout(
            height=310,
            margin={"l": 10, "r": 75, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={
                "title": "Giá trị tài chính ($)",
                "gridcolor": "rgba(148,163,184,0.22)",
                "range": [0, max_flow_x * 1.24],
            },
            yaxis={"autorange": "reversed", "tickfont": {"size": 12}},
        )
        st.plotly_chart(fig_cashflow, width="stretch")
        st.caption(
            (
                "Lợi ích gộp bảo vệ được sau khi trừ chi phí tiếp cận và ưu đ"
                "ãi mang lại thặng dư kinh tế ròng (ΔEV > 0)."
            )
        )

    with st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>So Sánh Hiệu Quả Tài Chính Từ'
                'ng Gói Can Thiệp (A1, A2, A3)</span><span class="panel-tag">'
                "PACKAGE ECONOMICS</span></div>"
            ),
            unsafe_allow_html=True,
        )

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
                        textposition="outside",
                        textangle=0,
                        cliponaxis=False,
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng (ΔEV)",
                        x=interv_actions["action_code"],
                        y=interv_actions["net_value_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in interv_actions["net_value_sum"]],
                        textposition="outside",
                        textangle=0,
                        cliponaxis=False,
                    ),
                ]
            )
            fig_grouped.update_layout(
                barmode="group",
                height=260,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                yaxis={"gridcolor": "rgba(148,163,184,0.22)"},
                xaxis={"tickangle": 0},
            )
            st.plotly_chart(fig_grouped, width="stretch")

        with col_table:
            st.markdown(render_package_economics_html_table(summary_df), unsafe_allow_html=True)

    with st.expander(
        "Xem & Tải Báo Cáo Tóm Tắt Điều Hành (Executive Summary Report)", expanded=False
    ):
        action_candidates = summary_df[summary_df["action_code"] != "A0"]
        top_action = (
            action_candidates.loc[cast(int, action_candidates["net_value_sum"].idxmax())].to_dict()
            if not action_candidates.empty
            else {
                "action_code": "A0",
                "action_name": "Không can thiệp",
                "net_value_sum": 0.0,
            }
        )
        a0_row = summary_df[summary_df["action_code"] == "A0"]
        a0_count = int(a0_row["n_customers"].values[0]) if not a0_row.empty else 0
        report_mode_note = (
            "MOCK — đầu ra mô phỏng, không phải dự đoán thực tế."
            if result_mode == "MOCK"
            else "LIVE API — ước lượng chưa hiệu chuẩn, không phải kết quả đã kiểm chứng."
        )
        report_text = (
            "**"
            f"{report_mode_note}"
            "**\n"
            "**Run:** "
            f"{active_run_id}"
            " | **Reference:** "
            f"{view.run.reference_version}"
            "\n"
            "\n"
            "# BÁO CÁO TÓM TẮT CHIẾN LƯỢC GIỮ CHÂN KHÁCH HÀNG (EXECUTIVE "
            "SUMMARY)\n"
            "**Dự án:** Customer Churn & Retention Decision Support Syste"
            "m\n"
            "**Kịch bản cơ sở:** Biên lợi nhuận m = "
            f"{base_margin:.0%}"
            " | Kỳ hạn H = "
            f"{horizon}"
            " tháng\n"
            "\n"
            "---\n"
            "\n"
            "### 1. KẾT QUẢ ĐIỀU HÀNH TỔNG THỂ\n"
            "- **Quy mô tệp khách hàng hoạt động:** "
            f"{n_total:,}"
            " khách hàng\n"
            "- **Quy mô can thiệp được duyệt (ΔEV > 0):** "
            f"{n_intervened:,}"
            " khách hàng ("
            f"{intervention_rate:.1%}"
            " tổng tệp)\n"
            "- **Tổng ngân sách đề xuất đầu tư:** $"
            f"{total_cost:,.2f}"
            " (Chi phí trung bình: $"
            f"{avg_cost_per_user:,.2f}"
            "/khách)\n"
            "- **Tổng lợi ích giữ chân kỳ vọng (Gross Benefit):** $"
            f"{total_benefit:,.2f}"
            "\n"
            "- **Lợi nhuận ròng kỳ vọng (Net Economic Value - ΔEV):** +$"
            f"{total_net_value:,.2f}"
            "\n"
            "- **Tỷ suất sinh lời trên chi phí (ROI mô phỏng):** "
            f"{roi_label}"
            "\n"
            "\n"
            "### 2. PHÂN BỔ THEO GÓI HÀNH ĐỘNG\n"
            "- **Gói hiệu quả tài chính cao nhất:** "
            f"{top_action['action_code']}"
            " ("
            f"{top_action['action_name']}"
            ") mang lại +$"
            f"{top_action['net_value_sum']:,.2f}"
            " lợi nhuận ròng.\n"
            "- **Phương án không can thiệp (A0):** "
            f"{a0_count:,}"
            " khách hàng do chi phí can thiệp lớn hơn lợi ích bảo vệ được"
            " (ΔEV ≤ 0).\n"
        )
        st.markdown(
            (
                "\n"
                '            <div style="background:#ffffff; border:1px solid'
                " #e3e8ee; border-left:4px solid #533afd; border-radius:10px;"
                ' padding:18px 22px; margin-bottom:12px;">\n'
                '                <div style="display:flex; justify-content:sp'
                "ace-between; align-items:center; flex-wrap:wrap; gap:8px; bo"
                "rder-bottom:1px solid #e3e8ee; padding-bottom:10px; margin-b"
                'ottom:14px;">\n'
                "                    <div>\n"
                '                        <div style="font-size:0.68rem; font-'
                "weight:700; text-transform:uppercase; letter-spacing:0.08em;"
                ' color:#533afd;">EXECUTIVE SUMMARY REPORT</div>\n'
                '                        <div style="font-size:1.05rem; font-'
                'weight:700; color:#0d253d;">Báo Cáo Tóm Tắt Chiến Lược Giữ C'
                "hân Khách Hàng</div>\n"
                "                    </div>\n"
                '                    <div style="display:flex; gap:6px;">\n'
                '                        <span class="badge-indigo">Biên lợi '
                "nhuận m = "
                f"{base_margin:.0%}"
                "</span>\n"
                '                        <span class="badge-slate">Kỳ hạn H ='
                " "
                f"{horizon}"
                " tháng</span>\n"
                "                    </div>\n"
                "                </div>\n"
                '                <div style="display:grid; grid-template-colu'
                "mns:1fr 1fr; gap:16px; font-size:0.84rem; line-height:1.65; "
                'color:#273951;">\n'
                '                    <div style="background:#f6f9fc; border:1'
                'px solid #e3e8ee; border-radius:8px; padding:12px 16px;">\n'
                '                        <div style="font-weight:700; font-si'
                "ze:0.86rem; color:#0d253d; margin-bottom:6px; text-transform"
                ':uppercase; letter-spacing:0.03em;">1. Kết Quả Điều Hành Tổn'
                "g Thể</div>\n"
                "                        <div>• <b>Quy mô tệp khách hàng hoạt"
                ' động:</b> <span style="font-family:Fira Code,monospace;font'
                '-weight:600;">'
                f"{n_total:,}"
                "</span> khách hàng</div>\n"
                "                        <div>• <b>Quy mô can thiệp được duyệ"
                't (ΔEV &gt; 0):</b> <span style="font-family:Fira Code,monos'
                'pace;font-weight:600;color:#4434d4;">'
                f"{n_intervened:,}"
                "</span> khách ("
                f"{intervention_rate:.1%}"
                " tổng tệp)</div>\n"
                "                        <div>• <b>Tổng ngân sách đề xuất đầu"
                ' tư:</b> <span style="font-family:Fira Code,monospace;font-w'
                'eight:600;color:#d97706;">&#36;'
                f"{total_cost:,.2f}"
                '</span> (TB: <span style="font-family:Fira Code,monospace;">'
                "&#36;"
                f"{avg_cost_per_user:,.2f}"
                "</span>/khách)</div>\n"
                "                        <div>• <b>Lợi ích giữ chân kỳ vọng ("
                'Gross B):</b> <span style="font-family:Fira Code,monospace;f'
                'ont-weight:600;">&#36;'
                f"{total_benefit:,.2f}"
                "</span></div>\n"
                "                        <div>• <b>Lợi nhuận ròng kỳ vọng (Ne"
                't ΔEV):</b> <span style="font-family:Fira Code,monospace;fon'
                't-weight:700;color:#059669;">+&#36;'
                f"{total_net_value:,.2f}"
                '</span> (ROI mô phỏng <span class="badge-emerald">'
                f"{roi_label}"
                "</span>)</div>\n"
                "                    </div>\n"
                '                    <div style="background:#f6f9fc; border:1'
                'px solid #e3e8ee; border-radius:8px; padding:12px 16px;">\n'
                '                        <div style="font-weight:700; font-si'
                "ze:0.86rem; color:#0d253d; margin-bottom:6px; text-transform"
                ':uppercase; letter-spacing:0.03em;">2. Phân Bổ Theo Gói Hành'
                " Động</div>\n"
                '                        <div style="margin-bottom:6px;">• <b'
                '>Gói hiệu quả tài chính cao nhất:</b> <span class="badge-eme'
                'rald">'
                f"{top_action['action_code']}"
                "</span> <b>"
                f"{top_action['action_name']}"
                '</b> mang lại lợi nhuận ròng <span style="font-family:Fira C'
                'ode,monospace;font-weight:700;color:#059669;">+&#36;'
                f"{top_action['net_value_sum']:,.2f}"
                "</span>.</div>\n"
                "                        <div>• <b>Phương án không can thiệp "
                '(A0):</b> <span style="font-family:Fira Code,monospace;font-'
                'weight:600;">'
                f"{a0_count:,}"
                "</span> khách hàng được giữ nguyên trạng thái do chi phí can"
                " thiệp lớn hơn lợi ích bảo vệ được (ΔEV ≤ 0), giúp tiết kiệm"
                " ngân sách.</div>\n"
                "                    </div>\n"
                "                </div>\n"
                "            </div>\n"
                "            "
            ),
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Tải Văn Bản Báo Cáo (.MD)",
            data=report_text.encode("utf-8"),
            file_name=f"{result_mode}_executive_retention_summary.md",
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
        (
            "\n"
            '        <div class="kpi-grid">\n'
            '            <div class="modern-card card-accent-blue">\n'
            '                <div class="card-label">\n'
            "                    <span>Tổng Tệp Khảo Sát</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#eef0ff; color:#533afd;">'
            f"{SVG_USERS}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">'
            f"{eda_total:,}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:100%; background:#533afd;"><'
            "/div></div>\n"
            '                <div class="card-footer"><span>Toàn bộ mẫu d'
            'ữ liệu Telco</span><span class="badge-indigo">100%</span></d'
            "iv>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-ruby">\n'
            '                <div class="card-label">\n'
            "                    <span>Churn Đã Quan Sát Trong Snapshot</"
            "span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#fff1f2; color:#ea2261;">'
            f"{SVG_ALERT}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value" style="color: #ea226'
            '1;">'
            f"{churn_pct:.1f}"
            "%</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{min(100.0, churn_pct):.1f}"
            '%; background:#ea2261;"></div></div>\n'
            '                <div class="card-footer"><span>Số lượng rời '
            'bỏ</span><span class="badge-rose">'
            f"{churn_count:,}"
            " KH</span></div>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-indigo">\n'
            '                <div class="card-label">\n'
            "                    <span>Cước Trung Bình Tháng</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#eef0ff; color:#4434d4;">'
            f"{SVG_WALLET}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">$'
            f"{avg_mc:.1f}"
            "</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{min(100.0, avg_mc / 1.2):.1f}"
            '%; background:#4434d4;"></div></div>\n'
            '                <div class="card-footer"><span>ARPU / Monthl'
            'yCharges</span><span class="badge-indigo">USD/Tháng</span></'
            "div>\n"
            "            </div>\n"
            '            <div class="modern-card card-accent-emerald">\n'
            '                <div class="card-label">\n'
            "                    <span>Thời Gian Gắn Bó TB</span>\n"
            '                    <div class="card-icon-wrap" style="backg'
            'round:#ecfdf5; color:#059669;">'
            f"{SVG_CLOCK}"
            "</div>\n"
            "                </div>\n"
            '                <div class="card-value">'
            f"{avg_tenure:.1f}"
            "M</div>\n"
            '                <div class="kpi-progress-track"><div class="'
            'kpi-progress-fill" style="width:'
            f"{min(100.0, avg_tenure / 72.0 * 100.0):.1f}"
            '%; background:#059669;"></div></div>\n'
            '                <div class="card-footer"><span>Chu kỳ gắn bó'
            '</span><span class="badge-emerald">~'
            f"{avg_tenure / 12:.1f}"
            " năm</span></div>\n"
            "            </div>\n"
            "        </div>\n"
            "        "
        ),
        unsafe_allow_html=True,
    )

    eda_row1_c1, eda_row1_c2 = st.columns(2)

    with eda_row1_c1, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Tỷ Lệ Churn Theo Loại Hợp Đồn'
                'g (Contract)</span><span class="panel-tag">CONTRACT COHORT</'
                "span></div>"
            ),
            unsafe_allow_html=True,
        )

        contract_churn = (
            customers_df.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        contract_churn.columns = ["Contract", "ChurnRate"]
        contract_order = ["Month-to-month", "One year", "Two year"]
        contract_churn = contract_churn.sort_values(
            by="Contract",
            key=lambda x: x.map({v: i for i, v in enumerate(contract_order)}),
        )

        fig_contract = go.Figure(
            go.Bar(
                x=contract_churn["Contract"],
                y=contract_churn["ChurnRate"],
                marker_color=["#e11d48", "#d97706", "#059669"],
                text=[f"{v:.1f}%" for v in contract_churn["ChurnRate"]],
                textposition="outside",
                textangle=0,
                cliponaxis=False,
            )
        )
        fig_contract.update_layout(
            height=260,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={
                "title": "Tỷ lệ rời mạng (%)",
                "gridcolor": "rgba(148,163,184,0.22)",
            },
            xaxis={"title": "Loại hợp đồng", "tickangle": 0},
        )
        st.plotly_chart(fig_contract, width="stretch")
        _cr = dict(zip(contract_churn["Contract"], contract_churn["ChurnRate"], strict=False))
        _m2m, _two = _cr.get("Month-to-month"), _cr.get("Two year")
        if _m2m is not None and _two:
            st.caption(
                (
                    "Nhận xét: Hợp đồng Month-to-month có tỷ lệ churn "
                    f"{_m2m:.1f}"
                    "%, cao gấp **"
                    f"{_m2m / _two:.1f}"
                    " lần** hợp đồng 2 năm ("
                    f"{_two:.1f}"
                    "%). Đây là cơ sở thiết kế gói A3."
                )
            )

    with eda_row1_c2, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Tỷ Lệ Churn Theo Dịch Vụ Inte'
                'rnet</span><span class="panel-tag">SERVICE TIER</span></div>'
            ),
            unsafe_allow_html=True,
        )

        internet_churn = (
            customers_df.groupby("InternetService")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        internet_churn.columns = ["InternetService", "ChurnRate"]

        fig_internet = go.Figure(
            go.Bar(
                x=internet_churn["InternetService"],
                y=internet_churn["ChurnRate"],
                marker_color=["#1e40af", "#e11d48", "#94a3b8"],
                text=[f"{v:.1f}%" for v in internet_churn["ChurnRate"]],
                textposition="outside",
                textangle=0,
                cliponaxis=False,
            )
        )
        fig_internet.update_layout(
            height=260,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={
                "title": "Tỷ lệ rời mạng (%)",
                "gridcolor": "rgba(148,163,184,0.22)",
            },
            xaxis={"title": "Dịch vụ Internet", "tickangle": 0},
        )
        st.plotly_chart(fig_internet, width="stretch")
        _ir = dict(
            zip(internet_churn["InternetService"], internet_churn["ChurnRate"], strict=False)
        )
        _fiber, _dsl = _ir.get("Fiber optic"), _ir.get("DSL")
        if _fiber is not None and _dsl:
            st.caption(
                (
                    "Nhận xét: Nhóm Fiber optic có tỷ lệ churn "
                    f"{_fiber:.1f}"
                    "%, gấp **"
                    f"{_fiber / _dsl:.1f}"
                    " lần** DSL ("
                    f"{_dsl:.1f}"
                    "%)."
                )
            )

    eda_row2_c1, eda_row2_c2 = st.columns(2)

    with eda_row2_c1, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Tác Động Của Hỗ Trợ Kỹ Thuật '
                '(Tech Support)</span><span class="panel-tag">SUPPORT IMPACT<'
                "/span></div>"
            ),
            unsafe_allow_html=True,
        )

        tech_churn = (
            customers_df.groupby("TechSupport")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        tech_churn.columns = ["TechSupport", "ChurnRate"]

        fig_tech = go.Figure(
            go.Bar(
                x=tech_churn["TechSupport"],
                y=tech_churn["ChurnRate"],
                marker_color=["#e11d48", "#94a3b8", "#059669"],
                text=[f"{v:.1f}%" for v in tech_churn["ChurnRate"]],
                textposition="outside",
                textangle=0,
                cliponaxis=False,
            )
        )
        fig_tech.update_layout(
            height=250,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={
                "title": "Tỷ lệ rời mạng (%)",
                "gridcolor": "rgba(148,163,184,0.22)",
            },
            xaxis={"title": "Trạng thái Tech Support", "tickangle": 0},
        )
        st.plotly_chart(fig_tech, width="stretch")
        _tr = dict(zip(tech_churn["TechSupport"], tech_churn["ChurnRate"], strict=False))
        _no_ts, _yes_ts = _tr.get("No"), _tr.get("Yes")
        if _no_ts is not None and _yes_ts is not None:
            st.caption(
                (
                    "Nhận xét: Khách thiếu Tech Support rời mạng "
                    f"{_no_ts:.1f}"
                    "%, so với "
                    f"{_yes_ts:.1f}"
                    "% ở nhóm có hỗ trợ. Đây là căn cứ thiết kế gói A1."
                )
            )

    with eda_row2_c2, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Tác Động Của Phương Thức Than'
                'h Toán</span><span class="panel-tag">PAYMENT CHANNEL</span><'
                "/div>"
            ),
            unsafe_allow_html=True,
        )

        pay_churn = (
            customers_df.groupby("PaymentMethod")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        pay_churn.columns = ["PaymentMethod", "ChurnRate"]
        pay_churn["ShortName"] = pay_churn["PaymentMethod"].map(
            {
                "Electronic check": "Electronic check (Thủ công)",
                "Mailed check": "Mailed check (Thủ công)",
                "Bank transfer (automatic)": "Bank transfer (Tự động)",
                "Credit card (automatic)": "Credit card (Tự động)",
            }
        )

        fig_pay = go.Figure(
            go.Bar(
                y=pay_churn["ShortName"],
                x=pay_churn["ChurnRate"],
                orientation="h",
                marker_color=["#059669", "#059669", "#e11d48", "#d97706"],
                text=[f"{v:.1f}%" for v in pay_churn["ChurnRate"]],
                textposition="outside",
                textangle=0,
                cliponaxis=False,
            )
        )
        fig_pay.update_layout(
            height=250,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={
                "title": "Tỷ lệ rời mạng (%)",
                "gridcolor": "rgba(148,163,184,0.22)",
            },
        )
        st.plotly_chart(fig_pay, width="stretch")
        _pr = dict(zip(pay_churn["PaymentMethod"], pay_churn["ChurnRate"], strict=False))
        _echeck = _pr.get("Electronic check")
        _auto = [v for k, v in _pr.items() if "automatic" in str(k)]
        if _echeck is not None and _auto:
            st.caption(
                (
                    "Nhận xét: Thanh toán Electronic check rời mạng "
                    f"{_echeck:.1f}"
                    "%, trong khi thanh toán tự động chỉ "
                    f"{min(_auto):.1f}"
                    "–"
                    f"{max(_auto):.1f}"
                    "%. Đây là căn cứ thiết kế gói A2."
                )
            )

    with st.container(border=True):
        st.markdown("### Giải thích từ dịch vụ LLM (mode được ghi trên từng kết quả)")
        st.info(
            (
                "TreeSHAP không áp dụng cho LLM và đã vô hiệu hóa riêng phần "
                "này. Reason là giải thích do provider trả về, không phải fea"
                "ture attribution hoặc quan hệ nhân quả."
            )
        )
        st.dataframe(
            safe_export_frame(
                predictions_df[
                    [
                        "customer_id",
                        "churn_prediction",
                        "churn_prob",
                        "reason",
                        "execution_mode",
                    ]
                ]
            ),
            hide_index=True,
        )


# ==============================================================================
# TAB 3: HỒ SƠ KHÁCH HÀNG 360° & ĐỀ XUẤT CAN THIỆP
# ==============================================================================
with tab3:
    with st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Bộ Lọc & Danh Sách Đề Xuất Ca'
                'n Thiệp</span><span class="panel-tag">CUSTOMER SEGMENTATION<'
                "/span></div>"
            ),
            unsafe_allow_html=True,
        )

        f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1, 1, 1])
        with f_col1:
            search_query = st.text_input(
                "Tìm kiếm Mã Khách Hàng (customerID):", placeholder="Ví dụ: DEMO-001"
            )
        with f_col2:
            action_filter = st.selectbox(
                "Gói can thiệp:",
                options=[
                    "Tất cả",
                    "Chỉ can thiệp (A1, A2, A3)",
                    "A0: Không can thiệp",
                    "A1: Hỗ trợ kỹ thuật",
                    "A2: Hợp đồng & TT",
                    "A3: Giảm cước",
                ],
                index=1,
            )
        with f_col3:
            risk_filter = st.selectbox(
                "Mức rủi ro Churn:",
                options=[
                    "Tất cả",
                    "Rất cao (> 70%)",
                    "Trung bình (40% - 70%)",
                    "Thấp (< 40%)",
                ],
                index=0,
            )
        with f_col4:
            contract_filter = st.selectbox(
                "Loại hợp đồng:",
                options=["Tất cả", "Month-to-month", "One year", "Two year"],
                index=0,
            )

        rec_with_features = recommendations.merge(
            customers_df[
                [
                    "customerID",
                    "Contract",
                    "MonthlyCharges",
                    "tenure",
                    "InternetService",
                    "TechSupport",
                    "OnlineSecurity",
                    "PaymentMethod",
                ]
            ],
            on="customerID",
            how="left",
        )

        filtered_df = rec_with_features.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["customerID"].str.contains(
                    search_query.strip(), case=False, regex=False
                )
            ]
        if action_filter == "Chỉ can thiệp (A1, A2, A3)":
            filtered_df = filtered_df[filtered_df["action_code"] != "A0"]
        elif action_filter.startswith("A"):
            code_sel = action_filter.split(":")[0]
            filtered_df = filtered_df[filtered_df["action_code"] == code_sel]

        if risk_filter == "Rất cao (> 70%)":
            filtered_df = filtered_df[filtered_df["churn_prob"] > 0.70]
        elif risk_filter == "Trung bình (40% - 70%)":
            filtered_df = filtered_df[
                (filtered_df["churn_prob"] >= 0.40) & (filtered_df["churn_prob"] <= 0.70)
            ]
        elif risk_filter == "Thấp (< 40%)":
            filtered_df = filtered_df[filtered_df["churn_prob"] < 0.40]

        if contract_filter != "Tất cả":
            filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]

        st.caption(f"Tìm thấy **{len(filtered_df):,}** khách hàng thỏa mãn tiêu chí lọc.")

        st.markdown(
            render_priority_customers_html_table(filtered_df, max_rows=10),
            unsafe_allow_html=True,
        )
        with st.expander(
            "Mở bảng dữ liệu tương tác toàn phần (Sắp xếp / Tra cứu toàn bộ danh sách)",
            expanded=False,
        ):
            filtered_display = (
                filtered_df[
                    [
                        "rank",
                        "customerID",
                        "action_code",
                        "action_name",
                        "churn_prob",
                        "proxy_value",
                        "cost",
                        "benefit",
                        "net_value",
                        "breakeven_prob",
                        "churn_prob_after",
                    ]
                ]
                .copy()
                .reset_index(drop=True)
            )
            filtered_display["rank"] = range(1, len(filtered_display) + 1)
            st.dataframe(
                safe_export_frame(filtered_display).style.format(
                    {
                        "churn_prob": "{:.1%}",
                        "churn_prob_after": "{:.1%}",
                        "breakeven_prob": "{:.1%}",
                        "proxy_value": "${:.2f}",
                        "cost": "${:.2f}",
                        "benefit": "${:.2f}",
                        "net_value": "${:+.2f}",
                    }
                ),
                width="stretch",
                height=220,
            )

        csv_data = (
            safe_export_frame(
                filtered_df[
                    [
                        "rank",
                        "customerID",
                        "action_code",
                        "action_name",
                        "churn_prob",
                        "proxy_value",
                        "cost",
                        "benefit",
                        "net_value",
                        "breakeven_prob",
                        "churn_prob_after",
                    ]
                ]
            )
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="Tải Danh Sách Đang Lọc (.CSV)",
            data=csv_data,
            file_name=f"{result_mode}_retention_recommendations_filtered.csv",
            mime="text/csv",
        )

    if not filtered_df.empty:
        cust_list = filtered_df["customerID"].tolist()
        cust_options = [
            (
                f"{cid}"
                "  |  Gói: "
                f"{row.action_code}"
                "  |  ΔEV: $"
                f"{row.net_value:+.2f}"
                "  |  Churn: "
                f"{row.churn_prob:.1%}"
            )
            for cid, row in zip(cust_list, filtered_df.itertuples(), strict=False)
        ]

        selected_idx = st.selectbox(
            "Chọn khách hàng để phân tích chuyên sâu (Customer 360 & LLM reason):",
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

        with r1_col1, st.container(border=True):
            st.markdown(
                (
                    '<div class="panel-title"><span>Hồ Sơ Khách Hàng: '
                    f"{escape(str(selected_cust_id))}"
                    '</span><span class="panel-tag">PROFILE</span></div>'
                ),
                unsafe_allow_html=True,
            )

            churn_val = float(cust_rec["churn_prob"]) * 100
            churn_after_val = float(cust_rec["churn_prob_after"]) * 100

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_val,
                    number={"suffix": "%", "valueformat": ".1f"},
                    delta={
                        "reference": churn_after_val,
                        "increasing": {"color": "#e11d48"},
                        "decreasing": {"color": "#059669"},
                        "position": "bottom",
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#cbd5e1",
                        },
                        "bar": {
                            "color": "#e11d48"
                            if churn_val > 65
                            else ("#d97706" if churn_val > 35 else "#059669")
                        },
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#e2e8f0",
                        "steps": [
                            {"range": [0, 35], "color": "#ecfdf5"},
                            {"range": [35, 65], "color": "#fffbeb"},
                            {"range": [65, 100], "color": "#fff1f2"},
                        ],
                        "threshold": {
                            "line": {"color": "#1e40af", "width": 3},
                            "thickness": 0.75,
                            "value": churn_after_val,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=195,
                margin={"l": 15, "r": 15, "t": 15, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    {
                        "text": "Vạch xanh: Rủi ro sau can thiệp",
                        "x": 0.5,
                        "y": -0.15,
                        "showarrow": False,
                        "font": {"size": 11, "color": "#1e40af"},
                    }
                ],
            )
            st.plotly_chart(fig_gauge, width="stretch")

            contract_badge = (
                '<span class="badge-rose">Month-to-month</span>'
                if cust_info["Contract"] == "Month-to-month"
                else f'<span class="badge-emerald">{cust_info["Contract"]}</span>'
            )
            tech_badge = (
                '<span class="badge-rose">Chưa có</span>'
                if cust_info["TechSupport"] == "No"
                else '<span class="badge-emerald">Đã kích hoạt</span>'
            )
            sec_badge = (
                '<span class="badge-rose">Chưa có</span>'
                if cust_info["OnlineSecurity"] == "No"
                else '<span class="badge-emerald">Đã kích hoạt</span>'
            )
            pay_badge = (
                '<span class="badge-amber">Thủ công</span>'
                if cust_info["PaymentMethod"] in MANUAL_PAYMENTS
                else '<span class="badge-indigo">Tự động</span>'
            )

            st.markdown(
                (
                    "\n"
                    '                <div class="profile-box">\n'
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Loại Hợp Đồng</span> '
                    f"{contract_badge}"
                    "</div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Thời Gian Gắn Bó</span> <span class="profile-val'
                    '">'
                    f"{cust_info['tenure']}"
                    " tháng</span></div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Cước Phí Tháng</span> <span class="profile-val" '
                    'style="color:#1e40af;">$'
                    f"{cust_info['MonthlyCharges']:.2f}"
                    "</span></div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Dịch Vụ Internet</span> <span class="profile-val'
                    '">'
                    f"{cust_info['InternetService']}"
                    "</span></div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Hỗ Trợ Kỹ Thuật</span> '
                    f"{tech_badge}"
                    "</div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Bảo Mật Online</span> '
                    f"{sec_badge}"
                    "</div>\n"
                    '                    <div class="profile-item"><span class="p'
                    'rofile-key">Thanh Toán</span> '
                    f"{pay_badge}"
                    "</div>\n"
                    "                </div>\n"
                    "                "
                ),
                unsafe_allow_html=True,
            )

        with r1_col2, st.container(border=True):
            st.markdown(
                (
                    '<div class="panel-title"><span>Quyết Định Tối Ưu & Điểm Hòa '
                    'Vốn</span><span class="panel-tag">DECISION TICKET</span></di'
                    "v>"
                ),
                unsafe_allow_html=True,
            )

            best_code = cust_rec["action_code"]
            best_name = cust_rec["action_name"]
            best_ev = cust_rec["net_value"]
            proxy_val = cust_rec["proxy_value"]
            breakeven = cust_rec["breakeven_prob"]

            if best_code != "A0":
                st.markdown(
                    (
                        "\n"
                        '                    <div class="recommend-box">\n'
                        '                        <div class="recommend-title">Gói Đề '
                        "Xuất: "
                        f"{best_code}"
                        " — "
                        f"{best_name}"
                        "</div>\n"
                        '                        <div style="display: grid; grid-temp'
                        'late-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">\n'
                        '                            <div style="background: #ffffff;'
                        " padding: 10px 12px; border-radius: 8px; border: 1px solid #"
                        'e2e8f0;">\n'
                        '                                <div style="font-size: 0.72r'
                        'em; color: #64748b; font-weight: 700;">LỢI NHUẬN RÒNG (ΔEV)<'
                        "/div>\n"
                        '                                <div style="font-size: 1.25r'
                        "em; font-weight: 700; color: #059669; font-family: 'Fira Cod"
                        "e', monospace;\">+$"
                        f"{best_ev:.2f}"
                        "</div>\n"
                        "                            </div>\n"
                        '                            <div style="background: #ffffff;'
                        " padding: 10px 12px; border-radius: 8px; border: 1px solid #"
                        'e2e8f0;">\n'
                        '                                <div style="font-size: 0.72r'
                        'em; color: #64748b; font-weight: 700;">GIÁ TRỊ BẢO VỆ (V)</d'
                        "iv>\n"
                        '                                <div style="font-size: 1.25r'
                        "em; font-weight: 700; color: #1e40af; font-family: 'Fira Cod"
                        "e', monospace;\">$"
                        f"{proxy_val:.2f}"
                        "</div>\n"
                        "                            </div>\n"
                        "                        </div>\n"
                        '                        <div style="padding: 12px 14px; back'
                        "ground: #ffffff; border-radius: 8px; border: 1px solid #e2e8"
                        'f0; line-height: 1.65; font-size: 0.85rem; color: #334155;">'
                        "\n"
                        '                            <div style="font-weight: 700; co'
                        'lor: #0f172a; margin-bottom: 4px;">Căn cứ kinh tế & điều kiệ'
                        "n hòa vốn:</div>\n"
                        "                            <div>• Ngưỡng rủi ro hòa vốn tối"
                        " thiểu: <b style=\"font-family:'Fira Code',monospace;\">p* = "
                        f"{breakeven:.1%}"
                        "</b></div>\n"
                        "                            <div>• Rủi ro rời bỏ hiện tại: <"
                        "b style=\"font-family:'Fira Code',monospace;\">p = "
                        f"{churn_val:.1f}"
                        "%</b></div>\n"
                        '                            <div style="margin-top: 6px; pad'
                        "ding-top: 6px; border-top: 1px solid #f1f5f9; color: #047857"
                        '; font-weight: 600;">\n'
                        "                                Vì p ("
                        f"{churn_val:.1f}"
                        "%) &gt; p* ("
                        f"{breakeven:.1%}"
                        "), phương án can thiệp mang lại kỳ vọng lời ròng +$"
                        f"{best_ev:.2f}"
                        ".\n"
                        "                            </div>\n"
                        "                        </div>\n"
                        "                    </div>\n"
                        "                    "
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    (
                        "\n"
                        '                    <div class="recommend-box" style="border'
                        '-left-color:#94a3b8;">\n'
                        '                        <div class="recommend-title" style="'
                        'color:#475569;">Đề Xuất: Gói A0 — Không Can Thiệp</div>\n'
                        '                        <div style="color:#64748b; font-size'
                        ': 0.86rem; line-height: 1.6;">\n'
                        "                            Tất cả phương án có chi phí đều "
                        "cho lợi nhuận ròng âm (ΔEV ≤ 0) hoặc rủi ro rời bỏ thấp hơn "
                        "ngưỡng hòa vốn (p &lt; p*). Giữ nguyên trạng thái để bảo toà"
                        "n ngân sách.\n"
                        "                        </div>\n"
                        "                    </div>\n"
                        "                    "
                    ),
                    unsafe_allow_html=True,
                )

        with r1_col3, st.container(border=True):
            st.markdown("### Reason từ LLM service")
            reason_row = predictions_df[predictions_df["customer_id"] == selected_cust_id].iloc[0]
            st.text(
                (
                    "churn_prediction: "
                    f"{int(reason_row['churn_prediction'])}"
                    " · churn_prob: "
                    f"{float(reason_row['churn_prob']):.6f}"
                    " · Churn đã quan sát: "
                    f"{cust_info['Churn']}"
                )
            )
            st.text(reason_row["reason"])
            st.caption(
                f"Provider: {reason_row['provider']} · Reference: {reason_row['reference_version']}"
            )
            st.info(
                (
                    "Không có Local SHAP cho LLM. Xác suất và hiệu quả can thiệp "
                    "chưa được hiệu chuẩn/kiểm chứng."
                )
            )

        r2_col1, r2_col2 = st.columns([1.35, 1])

        with r2_col1, st.container(border=True):
            st.markdown(
                (
                    '<div class="panel-title"><span>So Sánh Giá Trị Ròng (ΔEV) Từ'
                    'ng Phương Án Hợp Lệ</span><span class="panel-tag">ACTION MAT'
                    "RIX</span></div>"
                ),
                unsafe_allow_html=True,
            )

            fig_action_comp = go.Figure()
            colors_comp = [
                "#059669"
                if code == best_code and best_code != "A0"
                else ("#94a3b8" if code == "A0" else "#1e40af")
                for code in cust_eval["action_code"]
            ]

            fig_action_comp.add_trace(
                go.Bar(
                    x=cust_eval["action_code"],
                    y=cust_eval["net_value"],
                    text=[f"${v:+.2f}" for v in cust_eval["net_value"]],
                    textposition="outside",
                    textangle=0,
                    cliponaxis=False,
                    marker_color=colors_comp,
                    hovertemplate="<b>%{x}</b><br>Giá trị ròng (ΔEV): %{y:$.2f}<extra></extra>",
                )
            )
            fig_action_comp.update_layout(
                height=200,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis={
                    "title": "Lợi nhuận ròng ΔEV ($)",
                    "gridcolor": "rgba(148,163,184,0.22)",
                },
                xaxis={"title": "Mã hành động", "tickangle": 0},
            )
            st.plotly_chart(fig_action_comp, width="stretch")

            st.markdown(
                render_action_matrix_html_table(cust_eval, best_code),
                unsafe_allow_html=True,
            )

        with r2_col2, st.container(border=True):
            st.markdown(
                (
                    '<div class="panel-title"><span>Luận Cứ Ra Quyết Định & Phê D'
                    'uyệt</span><span class="panel-tag">EXECUTION</span></div>'
                ),
                unsafe_allow_html=True,
            )

            if best_code != "A0":
                be_val = cust_rec.get("breakeven_prob", None)
                be_text = f"{be_val:.1%}" if (be_val is not None and pd.notna(be_val)) else "N/A"
                st.markdown(
                    (
                        "\n"
                        '                    <div style="background: #f8fafc; border:'
                        " 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px;"
                        " font-size: 0.86rem; line-height: 1.65; color: #334155; marg"
                        'in-bottom: 12px;">\n'
                        '                        <div style="margin-bottom: 8px;">• <'
                        "b>Lý do chọn gói "
                        f"{best_code}"
                        ":</b> Mang lại thặng dư lợi nhuận kỳ vọng ròng cao nhất (+$"
                        f"{best_ev:.2f}"
                        ") sau khi trừ chi phí liên hệ và ưu đãi ($"
                        f"{cust_rec['cost']:.1f}"
                        ").</div>\n"
                        '                        <div style="margin-bottom: 8px;">• <'
                        "b>Điều kiện hòa vốn (p* = "
                        f"{be_text}"
                        "):</b> Rủi ro rời bỏ hiện tại là "
                        f"{churn_val:.1f}"
                        "%, vượt ngưỡng an toàn "
                        f"{be_text}"
                        ".</div>\n"
                        "                        <div>• <b>Khuyến nghị:</b> Phê duyệt"
                        " gói can thiệp để bảo toàn dòng tiền cước <b>$"
                        f"{cust_info['MonthlyCharges']:.1f}"
                        "/tháng</b>.</div>\n"
                        "                    </div>\n"
                        "                    "
                    ),
                    unsafe_allow_html=True,
                )
                st.button(
                    "Phê Duyệt & Ghi Nhận Gói Can Thiệp",
                    key="btn_apply",
                    type="primary",
                    width="stretch",
                    disabled=True,
                )
                st.info(
                    (
                        "Chưa có dịch vụ thực thi can thiệp. Đề xuất chưa được áp dụn"
                        "g; có thể lưu chiến dịch mô phỏng ở tab 04."
                    )
                )
            else:
                st.info(
                    (
                        "Phương án tối ưu là **A0 (Không can thiệp)** vì mọi phương á"
                        "n có chi phí đều mang lại lợi nhuận ròng âm (ΔEV ≤ 0)."
                    )
                )


# ==============================================================================
# TAB 4: TỐI ƯU NGÂN SÁCH & QUẢN LÝ CHIẾN DỊCH
# ==============================================================================
with tab4:
    st.info(
        (
            "Lưu ý thuật toán: Khi ngân sách bị giới hạn, hệ thống sử dụn"
            "g heuristic tham lam ưu tiên gói có **tỷ suất ΔEV trên mỗi U"
            "SD chi phí** cao nhất (nên có thể chọn gói tiết kiệm hơn như"
            " A2 thay vì A3 ở tab Hồ sơ 360° để phục vụ được nhiều khách "
            "hàng hơn). Đây là nghiệm xấp xỉ thực tế."
        )
    )

    current_total_cost = float(interventions["cost"].sum()) if not interventions.empty else 30000.0
    sim_col_left, sim_col_right = st.columns([1, 1.45])

    with sim_col_left, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Thiết Lập Hạn Mức Ngân Sách</'
                'span><span class="panel-tag">KNAPSACK SOLVER</span></div>'
            ),
            unsafe_allow_html=True,
        )

        e2e_budget = st.slider(
            "Hạn mức ngân sách B (USD):",
            min_value=0,
            max_value=max(int(current_total_cost) + 5000, 50000),
            value=15000,
            step=1000,
            key="e2e_budget_slider",
            help=(
                "Thuật toán ưu tiên khách hàng và gói can thiệp có tỷ số ΔEV "
                "/ Cost cao nhất cho tới khi chạm trần ngân sách."
            ),
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
        roi_e2e_label = f"{roi_e2e:+.1f}%" if spent_e2e > 0 else "N/A (chi phí bằng 0)"
        st.markdown(
            (
                "\n"
                '            <div class="profile-box" style="background:#ffff'
                'ff; border:1px solid #e3e8ee; padding:16px;">\n'
                '                <div style="display:flex; justify-content:sp'
                'ace-between; align-items:center; margin-bottom:6px;">\n'
                '                    <span style="font-size:0.74rem; font-wei'
                'ght:700; text-transform:uppercase; color:#64748d;">Giải Ngân'
                " Ngân Sách Knapsack</span>\n"
                '                    <span class="badge-indigo">'
                f"{budget_utilization:.1f}"
                "% Hạn mức</span>\n"
                "                </div>\n"
                "                <div style=\"font-family:'Fira Code',monospac"
                "e; font-size:1.35rem; font-weight:700; color:#0d253d; margin"
                '-bottom:8px;">\n'
                "                    $"
                f"{spent_e2e:,.2f}"
                ' <span style="font-size:0.88rem; color:#64748d; font-weight:'
                '500;">/ $'
                f"{e2e_budget:,}"
                "</span>\n"
                "                </div>\n"
                '                <div class="kpi-progress-track" style="heigh'
                't:7px; margin-bottom:14px;">\n'
                '                    <div class="kpi-progress-fill" style="wi'
                "dth:"
                f"{min(100.0, budget_utilization):.1f}"
                '%; background:#533afd;"></div>\n'
                "                </div>\n"
                '                <div class="profile-item">\n'
                '                    <span class="profile-key">Lợi nhuận ròng'
                " thu về (ΔEV)</span>\n"
                '                    <span class="badge-emerald" style="font-'
                'size:0.85rem;">+$'
                f"{earned_e2e:,.2f}"
                "</span>\n"
                "                </div>\n"
                '                <div class="profile-item">\n'
                '                    <span class="profile-key">Lợi ích gộp bả'
                "o vệ (Gross B)</span>\n"
                '                    <span class="profile-val">$'
                f"{benefit_e2e:,.2f}"
                "</span>\n"
                "                </div>\n"
                '                <div class="profile-item">\n'
                '                    <span class="profile-key">Tỷ suất sinh l'
                "ời (ROI mô phỏng)</span>\n"
                '                    <span class="profile-val" style="color:#'
                '059669;">'
                f"{roi_e2e_label}"
                "</span>\n"
                "                </div>\n"
                '                <div class="profile-item">\n'
                '                    <span class="profile-key">Số khách hàng '
                "được duyệt</span>\n"
                '                    <span class="profile-val">'
                f"{len(interv_e2e):,}"
                " khách hàng</span>\n"
                "                </div>\n"
                "            </div>\n"
                "            "
            ),
            unsafe_allow_html=True,
        )

    with sim_col_right, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Phân Bổ Ngân Sách & Danh Sách'
                ' Ưu Tiên Hàng Đầu</span><span class="panel-tag">TOP ALLOCATI'
                "ONS</span></div>"
            ),
            unsafe_allow_html=True,
        )

        if not interv_e2e.empty:
            sim_action_breakdown = (
                interv_e2e.groupby("action_code")
                .agg(
                    n_cust=("customerID", "count"),
                    cost_sum=("cost", "sum"),
                    net_sum=("net_value", "sum"),
                )
                .reset_index()
            )

            fig_sim = go.Figure(
                data=[
                    go.Bar(
                        name="Ngân Sách Tiêu ($)",
                        x=sim_action_breakdown["action_code"],
                        y=sim_action_breakdown["cost_sum"],
                        marker_color="#d97706",
                        text=[f"${v:,.0f}" for v in sim_action_breakdown["cost_sum"]],
                        textposition="outside",
                        textangle=0,
                        cliponaxis=False,
                    ),
                    go.Bar(
                        name="Lợi Nhuận Ròng Thu Về ($)",
                        x=sim_action_breakdown["action_code"],
                        y=sim_action_breakdown["net_sum"],
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in sim_action_breakdown["net_sum"]],
                        textposition="outside",
                        textangle=0,
                        cliponaxis=False,
                    ),
                ]
            )
            fig_sim.update_layout(
                barmode="group",
                height=210,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"title": "Giá trị ($)", "gridcolor": "rgba(148,163,184,0.22)"},
                xaxis={"title": "Gói can thiệp", "tickangle": 0},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
            )
            st.plotly_chart(fig_sim, width="stretch")

            st.markdown(
                render_knapsack_allocations_html_table(interv_e2e, max_rows=8),
                unsafe_allow_html=True,
            )
        else:
            st.info("Hạn mức ngân sách quá nhỏ để thực hiện bất kỳ hành động can thiệp nào.")

    with st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Chốt Kịch Bản Ngân Sách & Xuấ'
                't Chiến Dịch Thực Thi (.CSV)</span><span class="panel-tag">C'
                "AMPAIGN PERSISTENCE</span></div>"
            ),
            unsafe_allow_html=True,
        )

        camp_c1, camp_c2, camp_c3 = st.columns([1.4, 1, 1.1])
        with camp_c1:
            e2e_camp_name = st.text_input(
                "Tên chiến dịch:",
                value=(
                    "Chiến dịch ngân sách $"
                    f"{e2e_budget:,}"
                    " - "
                    f"{pd.Timestamp.now().strftime('%d/%m/%Y')}"
                ),
                key="e2e_camp_name_input",
            )
        with camp_c2:
            e2e_camp_ds_ref = st.text_input(
                "Tập dữ liệu áp dụng:",
                value=active_dataset_name,
                disabled=True,
                key="e2e_camp_ds_ref",
            )
        with camp_c3:
            st.write("")
            if st.button(
                "Lưu Chiến Dịch Vào Hệ Thống",
                type="primary",
                key="btn_save_e2e_camp",
                width="stretch",
            ):
                camp_results = {
                    "summary": {
                        "total_cost": spent_e2e,
                        "total_benefit": benefit_e2e,
                        "total_net_value": earned_e2e,
                        "n_intervened": len(interv_e2e),
                        "n_total": len(alloc_e2e),
                    },
                    "allocations": json.loads(alloc_e2e.to_json(orient="records")),
                }
                try:
                    new_saved_camp = create_campaign(
                        name=e2e_camp_name,
                        dataset_id=active_dataset_id,
                        model_version=active_model_version,
                        scoring_run_id=active_run_id,
                        budget=float(e2e_budget),
                        assumptions={
                            "margin": base_margin,
                            "horizon_months": horizon,
                            "actions": [asdict(a) for a in current_actions],
                            "prediction_source": "llm-rag-v2",
                            "execution_mode": view.run.execution_mode,
                            "probability_calibrated": False,
                            "reference_version": view.run.reference_version,
                            "reference_sha256": view.run.reference_sha256,
                            "source_run_dataset_digest": view.run.dataset_digest,
                            "model_version_semantics": (
                                "predictor configuration reference; actual model version unav"
                                "ailable"
                            ),
                        },
                        results=camp_results,
                        status=CampaignStatus.SIMULATED,
                    )
                    st.session_state["last_saved_camp_id"] = new_saved_camp.campaign_id
                    st.success("Đã lưu chiến dịch vào cơ sở dữ liệu.")
                except Exception:  # noqa: BLE001 -- Preserve snapshots and hide database internals.
                    st.error(
                        (
                            "Không thể lưu chiến dịch. Dữ liệu đã lưu được giữ nguyên; vu"
                            "i lòng thử lại."
                        )
                    )

        last_id = st.session_state.get("last_saved_camp_id")
        if last_id:
            csv_bytes = _archive_read(export_campaign_csv, last_id, only_interventions=True)
            campaign_mode = _archive_read(get_campaign, last_id).assumptions.get("execution_mode")
            st.download_button(
                label=(
                    "Tải Ngay File CSV Danh Sách Khách Hàng Được Chọn (customer_i"
                    "d, action_code, cost, benefit, net_value)"
                ),
                data=csv_bytes,
                file_name=(
                    f"{execution_label(campaign_mode)}_campaign_{last_id[:8]}_interventions.csv"
                ),
                mime="text/csv",
                key="e2e_csv_dl_btn",
                on_click=mark_exported,
                args=(last_id,),
            )

    with st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Lịch Sử Chiến Dịch Đã Lưu & T'
                'ải Báo Cáo CSV</span><span class="panel-tag">IMMUTABLE ARCHI'
                "VE</span></div>"
            ),
            unsafe_allow_html=True,
        )

        all_campaigns = _archive_read(list_campaigns)
        if not all_campaigns:
            st.info(
                (
                    "Hiện chưa có chiến dịch nào được lưu. Hãy bấm 'Lưu Chiến Dịc"
                    "h Vào Hệ Thống' ở trên để khởi tạo."
                )
            )
        else:
            camp_options = {
                c.campaign_id: (
                    f"{c.name}"
                    " ["
                    f"{c.status.value}"
                    "] — Ngân sách: $"
                    f"{c.budget:,.0f}"
                    " — Ngày: "
                    f"{c.created_at[:10]}"
                )
                for c in all_campaigns
            }
            selected_camp_id = st.selectbox(
                "Chọn chiến dịch đã lưu để tra cứu hoặc tải file CSV:",
                options=list(camp_options.keys()),
                format_func=lambda x: camp_options[x],
            )
            if selected_camp_id:
                selected_camp = _archive_read(get_campaign, selected_camp_id)
                summary_data = selected_camp.results.get("summary", {})

                c_budget = float(selected_camp.budget)
                c_cost = float(summary_data.get("total_cost", 0.0))
                c_net = float(summary_data.get("total_net_value", 0.0))
                c_n = int(summary_data.get("n_intervened", 0))
                st.markdown(
                    (
                        "\n"
                        '                    <div class="kpi-grid" style="margin-top:'
                        '8px; margin-bottom:12px;">\n'
                        '                        <div class="modern-card card-accent-'
                        'blue" style="padding:14px 16px;">\n'
                        '                            <div class="card-label"><span>Hạ'
                        "n Mức Ngân Sách</span></div>\n"
                        '                            <div class="card-value" style="f'
                        'ont-size:1.3rem;">$'
                        f"{c_budget:,.2f}"
                        "</div>\n"
                        "                        </div>\n"
                        '                        <div class="modern-card card-accent-'
                        'amber" style="padding:14px 16px;">\n'
                        '                            <div class="card-label"><span>Ch'
                        "i Phí Thực Tiêu</span></div>\n"
                        '                            <div class="card-value" style="f'
                        'ont-size:1.3rem;">$'
                        f"{c_cost:,.2f}"
                        "</div>\n"
                        "                        </div>\n"
                        '                        <div class="modern-card card-accent-'
                        'emerald" style="padding:14px 16px;">\n'
                        '                            <div class="card-label"><span>Lợ'
                        "i Nhuận Ròng (ΔEV)</span></div>\n"
                        '                            <div class="card-value" style="f'
                        'ont-size:1.3rem; color:#059669;">+$'
                        f"{c_net:,.2f}"
                        "</div>\n"
                        "                        </div>\n"
                        '                        <div class="modern-card card-accent-'
                        'indigo" style="padding:14px 16px;">\n'
                        '                            <div class="card-label"><span>Tr'
                        "ạng Thái & Quy Mô</span></div>\n"
                        '                            <div style="margin-top:4px;"><sp'
                        'an class="badge-indigo">'
                        f"{selected_camp.status.value}"
                        '</span> <span class="badge-emerald">'
                        f"{c_n:,}"
                        " KH</span></div>\n"
                        "                        </div>\n"
                        "                    </div>\n"
                        "                    "
                    ),
                    unsafe_allow_html=True,
                )

                camp_df = _archive_read(export_campaign_dataframe, selected_camp.campaign_id)
                interv_df = (
                    camp_df[camp_df["action_code"] != "A0"]
                    if not camp_df.empty and "action_code" in camp_df.columns
                    else camp_df
                )
                if not interv_df.empty:
                    st.markdown(
                        render_knapsack_allocations_html_table(interv_df, max_rows=8),
                        unsafe_allow_html=True,
                    )
                history_csv_data = _archive_read(
                    export_campaign_csv,
                    selected_camp.campaign_id,
                    only_interventions=True,
                )
                st.download_button(
                    label="Xuất File CSV Chiến Dịch Đã Chọn",
                    data=history_csv_data,
                    file_name=f"{execution_label(selected_camp.assumptions.get('execution_mode'))}_campaign_{selected_camp.campaign_id[:8]}.csv",
                    mime="text/csv",
                    key="history_csv_dl_btn",
                    on_click=mark_exported,
                    args=(selected_camp.campaign_id,),
                )

    sens_df = sensitivity_by_margin(
        customers=customers_df,
        predictions=predictions_df,
        margins=(0.20, 0.30, 0.40, 0.50),
        actions=current_actions,
        horizon_months=horizon,
    )

    sens_col1, sens_col2 = st.columns([1.2, 1])

    with sens_col1, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Độ Nhạy Dòng Tiền Khi Biên Lợ'
                'i Nhuận (m) Biến Động</span><span class="panel-tag">MARGIN S'
                "ENSITIVITY</span></div>"
            ),
            unsafe_allow_html=True,
        )

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_net_value"],
                name="Lợi Nhuận Ròng (ΔEV)",
                line={"color": "#059669", "width": 3},
                mode="lines+markers+text",
                text=[f"${v:,.0f}" for v in sens_df["total_net_value"]],
                textposition="top center",
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_cost"],
                name="Tổng Chi Phí",
                line={"color": "#d97706", "width": 2, "dash": "dash"},
                mode="lines+markers",
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=[f"{m:.0%}" for m in sens_df["margin"]],
                y=sens_df["total_benefit"],
                name="Lợi Ích Gộp",
                line={"color": "#1e40af", "width": 2},
                mode="lines+markers",
            )
        )
        fig_sens.update_layout(
            height=280,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={"title": "Giá trị ($)", "gridcolor": "rgba(148,163,184,0.22)"},
            xaxis={"title": "Biên Lợi Nhuận (m)", "tickangle": 0},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_sens, width="stretch")

    with sens_col2, st.container(border=True):
        st.markdown(
            (
                '<div class="panel-title"><span>Cơ Cấu Gói Can Thiệp Khi Marg'
                'in Tăng</span><span class="panel-tag">ACTION MIX</span></div'
                ">"
            ),
            unsafe_allow_html=True,
        )

        action_cols = [c for c in sens_df.columns if c.startswith("n_A")]
        fig_stacked = go.Figure()
        colors_dict = {"n_A1": "#1e40af", "n_A2": "#0284c7", "n_A3": "#059669"}
        names_dict = {
            "n_A1": "A1 (Kỹ thuật)",
            "n_A2": "A2 (Hợp đồng)",
            "n_A3": "A3 (Giảm cước)",
        }

        for ac in action_cols:
            fig_stacked.add_trace(
                go.Bar(
                    name=names_dict.get(ac, ac),
                    x=[f"{m:.0%}" for m in sens_df["margin"]],
                    y=sens_df[ac],
                    marker_color=colors_dict.get(ac, "#4f46e5"),
                    text=sens_df[ac],
                    textposition="outside",
                    textangle=0,
                    cliponaxis=False,
                )
            )

        fig_stacked.update_layout(
            barmode="stack",
            height=280,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={
                "title": "Số lượng khách can thiệp",
                "gridcolor": "rgba(148,163,184,0.22)",
            },
            xaxis={"title": "Biên Lợi Nhuận (m)", "tickangle": 0},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_stacked, width="stretch")
