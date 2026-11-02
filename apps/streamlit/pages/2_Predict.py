"""Submit one validated customer to the existing LLM/RAG predictor."""

import streamlit as st

from churn_retention.contracts.reference import ReferenceArtifactError
from churn_retention.entrypoints.streamlit.llm_input import (
    ERROR_KEY,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    FINGERPRINT_KEY,
    RESOURCES_KEY,
    RESULT_INPUT_KEY,
    RESULT_KEY,
    VALIDATED_INPUT_KEY,
    load_form_resources,
    reference_artifact_path,
    reference_artifact_sha256,
    reference_fingerprint,
    validate_customer,
    value_label,
)
from churn_retention.entrypoints.streamlit.llm_prediction_record import (
    capture_submission,
    clear_record,
)
from churn_retention.entrypoints.streamlit.llm_prediction_service import (
    LLMPredictionServiceError,
    predict_customer,
)

st.set_page_config(page_title="Dự đoán churn bằng LLM", page_icon="🔎")
st.title("Dự đoán khách hàng rời bỏ bằng LLM")
st.caption(
    "Hệ thống sử dụng dữ liệu khách hàng lịch sử, truy xuất các trường hợp tương tự "
    "và mô hình ngôn ngữ lớn để đưa ra dự đoán."
)


def clear_submission() -> None:
    clear_record(st.session_state)
    for key in (VALIDATED_INPUT_KEY, ERROR_KEY, RESULT_KEY, RESULT_INPUT_KEY):
        st.session_state.pop(key, None)


try:
    reference_path = reference_artifact_path()
    fingerprint = (reference_fingerprint(reference_path), reference_artifact_sha256())
    if (
        RESOURCES_KEY not in st.session_state
        or st.session_state.get(FINGERPRINT_KEY) != fingerprint
    ):
        clear_submission()
        st.session_state.pop(RESOURCES_KEY, None)
        st.session_state.pop(FINGERPRINT_KEY, None)
        loaded = load_form_resources(reference_path)
        if (
            reference_fingerprint(reference_path),
            reference_artifact_sha256(),
        ) != fingerprint:
            raise ReferenceArtifactError("Reference changed while loading.")
        st.session_state[RESOURCES_KEY] = loaded
        st.session_state[FINGERPRINT_KEY] = fingerprint
except Exception:  # noqa: BLE001 -- Never render artifact/preprocessor error contents.
    # Artifact/preprocessor errors may contain local paths or customer data.
    clear_submission()
    st.session_state.pop(RESOURCES_KEY, None)
    st.session_state.pop(FINGERPRINT_KEY, None)
    st.error(
        "Không thể nạp dữ liệu tham chiếu hợp lệ. Vui lòng liên hệ người quản trị."
    )
    st.stop()

resources = st.session_state[RESOURCES_KEY]
customer: dict[str, object] = {}
with st.form("llm_customer_input"):
    for group_label, features in FEATURE_GROUPS:
        st.subheader(group_label)
        for name in features:
            label = FEATURE_LABELS[name]
            key = f"llm_input.feature.{name}"
            if name in resources.choices:
                customer[name] = st.selectbox(
                    label,
                    resources.choices[name],
                    index=None,
                    placeholder="Chọn một giá trị",
                    format_func=value_label,
                    key=key,
                )
            elif name == "SeniorCitizen":
                customer[name] = st.selectbox(
                    label,
                    (0, 1),
                    index=None,
                    placeholder="Chọn một giá trị",
                    format_func=lambda value: "Có" if value == 1 else "Không",
                    key=key,
                )
            elif name == "tenure":
                customer[name] = st.number_input(
                    label, min_value=0, value=None, step=1, key=key
                )
            else:
                customer[name] = st.number_input(
                    label,
                    min_value=0.0,
                    value=None,
                    step=0.01,
                    key=key,
                    help="Có thể để trống khi tenure bằng 0; không tự tính từ chi phí hàng tháng."
                    if name == "TotalCharges"
                    else None,
                )
    submitted = st.form_submit_button("Dự đoán bằng LLM", type="primary")

if submitted:
    clear_submission()
    try:
        snapshot = validate_customer(resources, customer)
    except ValueError as error:
        st.session_state[ERROR_KEY] = str(error)
    except Exception:  # noqa: BLE001 -- Hide unexpected preprocessing/validation details.
        st.session_state[ERROR_KEY] = (
            "Không thể xác thực dữ liệu. Vui lòng kiểm tra lại hoặc liên hệ người quản trị."
        )
    else:
        st.session_state[VALIDATED_INPUT_KEY] = snapshot
        try:
            with st.spinner("Đang truy xuất dữ liệu tham chiếu và dự đoán bằng LLM…"):
                result = predict_customer(resources.store, snapshot)
            capture_submission(
                st.session_state, snapshot, result, fingerprint[1], fingerprint
            )
            st.session_state[RESULT_KEY] = result
            st.session_state[RESULT_INPUT_KEY] = dict(customer)
        except LLMPredictionServiceError as error:
            st.session_state[ERROR_KEY] = str(error)
        except Exception:  # noqa: BLE001 -- Final UI boundary must not expose raw errors.
            st.session_state[ERROR_KEY] = (
                "Chưa thể hoàn tất dự đoán bằng LLM. Vui lòng thử lại sau."
            )

if ERROR_KEY in st.session_state:
    st.error(st.session_state[ERROR_KEY])
elif RESULT_KEY in st.session_state:
    if customer != st.session_state.get(RESULT_INPUT_KEY):
        st.info(
            "Dữ liệu trong form đã thay đổi. Hãy gửi lại để có dự đoán cho dữ liệu mới."
        )
    else:
        result = st.session_state[RESULT_KEY]
        st.caption(
            "Kết quả của lần gửi dữ liệu gần nhất. Thay đổi trong form chưa tạo dự đoán mới."
        )
        st.subheader("Kết quả dự đoán")
        st.success(
            "Khách hàng có khả năng rời bỏ"
            if result.churn_prediction == 1
            else "Khách hàng có khả năng tiếp tục sử dụng"
        )
        st.metric("Xác suất do LLM ước lượng", f"{result.churn_probability:.0%}")
        st.warning(
            "Đây là ước lượng do LLM cung cấp, chưa được hiệu chuẩn thành xác suất thống kê."
        )
        st.subheader("Giải thích")
        st.text(result.reason)
        provider = {"Groq": "Groq/Qwen", "Gemini": "Gemini"}.get(
            result.provider, "Không xác định"
        )
        st.caption(f"Provider: {provider}")
        st.caption(f"Số khách hàng tham chiếu: {result.reference_count}")
        st.text(f"Reference version: {result.reference_version}")
