import os
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import streamlit as st

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.modules.scoring.repository import InMemoryScoringRunRepository
from churn_retention.modules.scoring.use_cases import (
    DatasetNotFoundError,
    InvalidDatasetError,
    score_dataset,
)


st.set_page_config(page_title="Chấm điểm churn", page_icon="📊")
st.title("Chấm điểm churn")
st.caption("Chạy scoring trực tiếp trên dataset CSV và model bundle đã lưu.")

dataset_path = st.text_input(
    "Đường dẫn dataset CSV",
    placeholder="data/scoring/customers.csv",
)
bundle_path = st.text_input(
    "Đường dẫn model bundle",
    value=os.getenv("CHURN_MODEL_BUNDLE", ""),
    placeholder="artifacts/bundles/<model-version>",
)

if st.button("Chấm điểm", type="primary"):
    if not dataset_path.strip():
        st.error("Vui lòng nhập đường dẫn dataset.")
    elif not bundle_path.strip():
        st.error("Vui lòng nhập đường dẫn model bundle.")
    else:
        try:
            predictor = BundlePredictor()
            predictor.load(bundle_path.strip())
            batch = score_dataset(
                dataset_path.strip(),
                lambda path: pd.read_csv(path),
                predictor,
                lambda: str(uuid4()),
                lambda: datetime.now(timezone.utc),
            )

            repository = InMemoryScoringRunRepository()
            repository.save(batch)
            saved_batch = repository.get(batch.scoring_run_id)
            if saved_batch is None:
                raise RuntimeError("Không đọc lại được kết quả scoring vừa lưu.")

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "customer_id": prediction.customer_id,
                            "churn_probability": prediction.churn_probability,
                            "model_version": prediction.model_version,
                        }
                        for prediction in saved_batch.predictions
                    ],
                    columns=["customer_id", "churn_probability", "model_version"],
                ),
                hide_index=True,
            )
        except DatasetNotFoundError as error:
            st.error(f"Không tìm thấy dataset: {error}")
        except InvalidDatasetError as error:
            st.error(f"Dataset không hợp lệ: {error}")
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            st.error(f"Không đọc được dataset hoặc model bundle: {error}")
        except Exception as error:
            st.error(f"Chấm điểm thất bại: {error}")
