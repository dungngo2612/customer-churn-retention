"""Run through: streamlit run apps/streamlit/scoring_page.py."""

import streamlit as st

from churn_retention.contracts.http.explanations import ModelExplanation
from churn_retention.entrypoints.streamlit.explanation_views import (
    SCORING_CONTEXT_KEY,
    customer_ids,
    global_importance_frame,
    load_train_background,
    local_contribution_frame,
    select_customer,
)
from churn_retention.ml.explainability import explain_global, explain_one

st.set_page_config(page_title="Giải thích SHAP", page_icon="🔎")
st.title("Giải thích mô hình và khách hàng")
st.info("SHAP mô tả mô hình, không phải nhân quả")
st.caption(
    'output_type = "log_odds". Đóng góp dương tăng log-odds churn; '
    "đóng góp âm giảm log-odds churn. Đơn vị đóng góp là log-odds."
)

context = st.session_state.get(SCORING_CONTEXT_KEY)
if context is None:
    st.warning("Hãy chấm điểm dataset bằng model bundle trước khi mở trang này.")
    st.stop()

st.write(f"Model đang dùng khi scoring: {context.bundle.model_version}")
st.caption(f"Bundle: {context.bundle_path}")
background_path = st.text_input(
    "Background chỉ từ train.csv", value="data/processed/train.csv"
)
st.caption("Lấy tối đa 200 dòng train, seed 42. Global importance mô tả mẫu nền này.")

try:
    customer_id = st.selectbox(
        "Khách hàng trong dataset vừa scoring", customer_ids(context.dataset)
    )
    if st.button("Tính SHAP", type="primary"):
        background = load_train_background(background_path)
        global_result = explain_global(context.bundle, background)
        row = select_customer(context.dataset, customer_id)
        local = ModelExplanation.model_validate(
            explain_one(context.bundle, row, background)
        )

        st.subheader("Toàn cục: mean |SHAP| theo feature gốc")
        st.bar_chart(
            global_importance_frame(global_result),
            y="mean_abs_shap",
            horizontal=True,
            sort=False,
        )
        st.caption(
            "Cộng các cột one-hot theo từng dòng trước khi lấy trung bình trị tuyệt đối."
        )
        st.subheader(f"Khách hàng: {customer_id}")
        columns = st.columns(3)
        columns[0].metric("Base value (log-odds)", f"{local.base_value:.6f}")
        columns[1].metric("Log-odds dự đoán", f"{local.log_odds:.6f}")
        columns[2].metric("Xác suất churn", f"{local.probability:.2%}")
        frame = local_contribution_frame(local)
        st.bar_chart(
            frame.set_index("feature"),
            y="contribution_log_odds",
            horizontal=True,
            sort=False,
        )
        st.dataframe(frame, hide_index=True)
        total = local.base_value + sum(
            item.contribution for item in local.contributions
        )
        st.caption(
            f"Base + tổng contribution = {total:.9f}; log-odds = {local.log_odds:.9f}."
        )
        with st.expander("Chi tiết cột sau preprocessing / one-hot"):
            st.dataframe(
                [item.model_dump() for item in local.detailed_contributions],
                hide_index=True,
            )
        st.download_button(
            "Tải giải thích JSON",
            local.model_dump_json(by_alias=True, indent=2),
            file_name="explanation.json",
            mime="application/json",
        )
except (ValueError, TypeError, FileNotFoundError, NotImplementedError) as error:
    st.error(str(error))
