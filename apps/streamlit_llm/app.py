"""Independent LLM/RAG app with an explicit Predict/simulation page allowlist."""

from pathlib import Path

import streamlit as st

predict_page = Path(__file__).resolve().parents[1] / "streamlit/pages/2_Predict.py"

# An explicit page registry disables automatic pages/ discovery for this app.
# Only Predict and the Retention/Budget simulation are registered.
st.navigation(
    [
        st.Page(
            predict_page,
            title="Dự đoán churn bằng LLM",
            icon="🔎",
            default=True,
        ),
        st.Page(
            Path(__file__).with_name("retention_page.py"),
            title="Mô phỏng Retention / Budget",
            icon="🧮",
            url_path="retention",
        ),
    ],
    position="sidebar",
).run()
