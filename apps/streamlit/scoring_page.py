import os
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import streamlit as st

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.entrypoints.streamlit.explanation_views import (
    SCORING_CONTEXT_KEY,
    scoring_context,
)
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
            st.session_state.pop(SCORING_CONTEXT_KEY, None)
            predictor = BundlePredictor()
            predictor.load(bundle_path.strip())
            loaded_frames: dict[str, pd.DataFrame] = {}

            def load_scoring_data(path: str) -> pd.DataFrame:
                frame = pd.read_csv(path)
                loaded_frames["dataset"] = frame
                return frame

            batch = score_dataset(
                dataset_path.strip(),
                load_scoring_data,
                predictor,
                lambda: str(uuid4()),
                lambda: datetime.now(UTC),
            )

            repository = InMemoryScoringRunRepository()
            repository.save(batch)
            saved_batch = repository.get(batch.scoring_run_id)
            if saved_batch is None:
                raise RuntimeError("Không đọc lại được kết quả scoring vừa lưu.")

            st.session_state[SCORING_CONTEXT_KEY] = scoring_context(
                predictor.loaded_bundle, loaded_frames["dataset"], bundle_path.strip()
            )
            st.page_link("pages/1_Explain.py", label="Giải thích mô hình và khách hàng")

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
        # Keep unexpected failures in the existing user-visible error message.
        except Exception as error:  # noqa: BLE001
            st.error(f"Chấm điểm thất bại: {error}")
