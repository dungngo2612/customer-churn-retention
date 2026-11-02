"""Explicitly registered simulation page; no inference on render/rerun."""

import streamlit as st

from churn_retention.entrypoints.streamlit.llm_retention_presenter import (
    render_retention,
)

st.set_page_config(page_title="Mô phỏng Retention / Budget", page_icon="🧮")
render_retention()
