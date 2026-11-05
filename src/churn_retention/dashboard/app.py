"""Main retention workspace: official Streamlit navigation, existing services."""

import base64

import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state
from churn_retention.dashboard.i18n import TRANSLATIONS, t
from churn_retention.dashboard.theme_improved import apply_improved_theme as apply_theme
from churn_retention.entrypoints.streamlit import demo_security

try:
    demo_security.require_safe_runtime()
except ValueError:
    st.error(t("app_safe_runtime_error"))
    st.stop()

st.set_page_config(
    page_title="Trung tâm điều hành giữ chân khách hàng",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

# Resolve the active language once so navigation titles (which must be string
# values at the time st.navigation is called) are localised correctly.
_lang = st.session_state.get("ui_lang", "vi")
_tr = TRANSLATIONS.get(_lang, TRANSLATIONS["vi"])

brand_name = _tr.get("app_brand_name", "Giữ chân khách hàng")
brand_sub = _tr.get("app_brand_sub", "Trung tâm điều hành")
dark = st.session_state.get("ui_dark_mode", False)
brand_fill = "#f1f5f9" if dark else "#1a202c"
sub_fill = "#94a3b8" if dark else "#718096"

logo = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="252" height="40" viewBox="0 0 252 40">'
    '<rect x="0" y="0" width="40" height="40" rx="12" fill="#4f46e5"/>'
    '<path d="M12 29l6 6 11-15M12 20h7" fill="none" stroke="white" '
    'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    f'<text x="53" y="19" font-family="Inter,-apple-system,BlinkMacSystemFont,sans-serif" font-size="18"'
    f' font-weight="600" fill="{brand_fill}">{brand_name}</text>'
    f'<text x="53" y="35" font-family="Inter,-apple-system,BlinkMacSystemFont,sans-serif" font-size="12"'
    f' fill="{sub_fill}">{brand_sub}</text></svg>'
)
st.logo("data:image/svg+xml;base64," + base64.b64encode(logo.encode()).decode(), size="large")

page = st.navigation(
    [
        st.Page("views/overview.py", title=_tr["nav_overview"], icon=":material/dashboard:", default=True),
        st.Page(
            "views/customers.py",
            title=_tr["nav_customers"],
            icon=":material/group:",
            url_path="khach-hang",
        ),
        st.Page(
            "views/analysis.py",
            title=_tr["nav_analysis"],
            icon=":material/insights:",
            url_path="phan-tich",
        ),
        st.Page(
            "views/plan.py",
            title=_tr["nav_plan"],
            icon=":material/account_balance_wallet:",
            url_path="ke-hoach",
        ),
        st.Page(
            "views/campaigns.py",
            title=_tr["nav_campaigns"],
            icon=":material/folder_open:",
            url_path="chien-dich",
        ),
    ],
    position="sidebar",
    expanded=True,
)

try:
    ctx = state.context()
    with st.sidebar:
        # ── Dataset / results info ────────────────────────────────────────
        st.caption(t("sidebar_data_label"))
        if ctx.data:
            st.write(ctx.data.source)
            st.caption(f"{len(ctx.data.frame)} {t('sidebar_customers_suffix')}")
        else:
            st.caption(t("sidebar_no_data"))
        if ctx.view:
            st.caption(t("sidebar_view_label"))
            st.write(ctx.view.run.source_name)
            ui.badge(
                ui.mode_label(ctx.view.run.execution_mode),
                "warning" if ctx.view.run.execution_mode == "mock" else "success",
            )
        else:
            ui.badge(t("sidebar_mock_badge"), "warning")
        st.caption(t("sidebar_sim_note"))
        if ctx.data or ctx.view:
            st.divider()
            if st.button(
                "🔄 " + t("sidebar_reset_btn"),
                key="reset_workspace_btn",
                use_container_width=True,
            ):
                state.clear_session()
                st.rerun()

    # ── Top-Right Header Bar: Language (Tiếng Việt / English) & Dark Mode ────
    with st.container(key="top_bar_container"):
        col_lang, col_dark = st.columns([2.15, 1.05])
        with col_lang:
            current_lang = st.session_state.get("ui_lang", "vi")
            chosen_lang = st.segmented_control(
                "Language",
                options=["vi", "en"],
                format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English",
                default=current_lang,
                key="top_lang_control",
                label_visibility="collapsed",
            )
            if chosen_lang and chosen_lang != current_lang:
                st.session_state["ui_lang"] = chosen_lang
                st.rerun()
        with col_dark:
            is_dark = st.session_state.get("ui_dark_mode", False)
            chosen_theme = st.segmented_control(
                "Theme",
                options=["light", "dark"],
                format_func=lambda x: (
                    ("☀️ Sáng" if x == "light" else "🌙 Tối")
                    if current_lang == "vi"
                    else ("☀️ Light" if x == "light" else "🌙 Dark")
                ),
                default="dark" if is_dark else "light",
                key="top_dark_control",
                label_visibility="collapsed",
            )
            if chosen_theme and (chosen_theme == "dark") != is_dark:
                st.session_state["ui_dark_mode"] = chosen_theme == "dark"
                st.rerun()

    if ctx.error:
        st.error(ctx.error)
    page.run()
except Exception:  # noqa: BLE001 -- Never expose raw archive, customer or provider errors.
    st.error(t("app_workspace_error"))
