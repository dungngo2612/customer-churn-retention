"""Dataset upload, catalog, detail, and integrity screen.

Run with: streamlit run apps/streamlit/datasets_page.py
"""

import pandas as pd
import streamlit as st

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.session import SessionLocal
from churn_retention.modules.datasets import use_cases

st.set_page_config(page_title="Datasets", page_icon="???")
st.title("Dataset management")
st.caption("Upload snapshots, browse stored datasets, and verify their files.")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
if st.button("Upload", disabled=uploaded is None):
    with SessionLocal() as session:
        result = use_cases.process_dataset_upload(
            uploaded.getvalue(),
            uploaded.name,
            DatasetRepository(session),
        )
    if result["status"] == "valid":
        st.success(f"Dataset created: {result['dataset_id']} ({result['total_rows']} rows)")
        st.rerun()
    elif result["status"] == "invalid":
        st.error("Dataset validation failed.")
        st.dataframe(pd.DataFrame(result["errors"]), hide_index=True)
    else:
        st.error(result["message"])

with SessionLocal() as session:
    repository = DatasetRepository(session)
    datasets = use_cases.list_datasets(repository)
    if datasets:
        st.subheader("Datasets")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "dataset_id": item["dataset_id"],
                        "filename": item["filename"],
                        "rows": item["total_rows"],
                        "columns": item["total_columns"],
                        "has_labels": item["has_labels"],
                        "created_at": item["created_at"],
                    }
                    for item in datasets
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        selected_id = st.selectbox(
            "Dataset details",
            options=[item["dataset_id"] for item in datasets],
        )
        detail = use_cases.get_dataset_detail(selected_id, repository)
        st.json(detail)
        if st.button("Check data quality"):
            quality = use_cases.check_dataset_quality(selected_id, repository)
            if quality and quality["is_valid"]:
                st.success("Dataset files, IDs, row counts, schema, and checksum are valid.")
            else:
                st.error("Dataset quality check failed.")
                if quality:
                    st.dataframe(pd.DataFrame(quality["errors"]), hide_index=True)
    else:
        st.info("No datasets uploaded yet.")
