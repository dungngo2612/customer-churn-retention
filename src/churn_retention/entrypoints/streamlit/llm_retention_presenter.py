"""Downstream, session-only simulation UI; this module never imports an LLM service."""

import streamlit as st

from churn_retention.contracts.llm_prediction_record import PredictionRecord
from churn_retention.entrypoints.streamlit.llm_prediction_record import (
    RECORD_CONTEXT_KEY,
    RECORD_KEY,
)
from churn_retention.modules.retention.demo import demo_inputs
from churn_retention.modules.retention.llm_adapter import (
    CustomerContext,
    RetentionScenario,
    simulate,
)

DEMO_KEY = "llm_retention.demo_inputs"


def _accepted_record() -> PredictionRecord | None:
    record = st.session_state.get(RECORD_KEY)
    if not isinstance(record, PredictionRecord) or record.origin != "llm":
        st.info(
            "Chưa có kết quả LLM đã chấp nhận trong phiên này. Có thể dùng dữ liệu giả để thử."
        )
        return None
    # Imported only for live results. Demo needs no artifact or legacy deployment.
    from churn_retention.entrypoints.streamlit.llm_input import (
        reference_artifact_path,
        reference_artifact_sha256,
        reference_fingerprint,
    )

    try:
        current = (
            reference_fingerprint(reference_artifact_path()),
            reference_artifact_sha256(),
        )
    except Exception:  # noqa: BLE001 -- Do not expose artifact paths/customer details.
        st.error(
            "Không thể xác minh ngữ cảnh tham chiếu hiện tại. Chưa thể mô phỏng kết quả này."
        )
        return None
    if (
        current != st.session_state.get(RECORD_CONTEXT_KEY)
        or record.reference_sha256 != current[1]
    ):
        st.error(
            "Tham chiếu đã thay đổi. Kết quả cũ không được dùng cho mô phỏng hiện hành."
        )
        return None
    return record


def render_retention() -> None:
    st.title("Mô phỏng Retention / Budget")
    st.warning(
        "Mô phỏng theo giả định: xác suất LLM chưa hiệu chuẩn; hiệu quả can thiệp chưa được kiểm chứng. "
        "Giá trị kinh tế không phải ROI đã xác nhận. Phân bổ dùng greedy heuristic."
    )
    source = st.radio(
        "Nguồn dữ liệu",
        ("Dữ liệu giả (demo)", "Kết quả LLM vừa gửi"),
        key="llm_retention.source",
    )
    if source == "Dữ liệu giả (demo)":
        st.info(
            "DỮ LIỆU GIẢ: 4 khách hàng và xác suất viết tay. Không gọi LLM, không đọc dataset."
        )
        if DEMO_KEY not in st.session_state:
            st.session_state[DEMO_KEY] = demo_inputs()
        inputs = st.session_state[DEMO_KEY]
    else:
        record = _accepted_record()
        if record is None:
            return
        st.caption(
            f"Kết quả đã gửi lúc {record.created_at.isoformat()} · Run {record.record_id}"
        )
        st.text(f"Reference version: {record.prediction.reference_version}")
        st.caption(
            "Không dùng bản nháp trong form Predict và không tạo lượt dự đoán mới."
        )
        if st.session_state.get("llm_retention.context_record_id") != record.record_id:
            for key in ("llm_retention.customer_ref", "llm_retention.observed_active"):
                st.session_state.pop(key, None)
            st.session_state["llm_retention.context_record_id"] = record.record_id
        customer_ref = st.text_input(
            "Mã khách hàng cục bộ cho mô phỏng",
            value="local-customer",
            key="llm_retention.customer_ref",
        )
        status = st.selectbox(
            "Trạng thái hoạt động từ ngữ cảnh nghiệp vụ (không phải nhãn LLM)",
            ("Đang hoạt động", "Đã rời bỏ"),
            index=None,
            key="llm_retention.observed_active",
        )
        if status is None:
            st.info(
                "Cần xác nhận trạng thái hoạt động; không suy ra từ churn_prediction."
            )
            return
        try:
            inputs = (
                (
                    record,
                    CustomerContext(
                        customer_ref=customer_ref,
                        observed_active=status == "Đang hoạt động",
                    ),
                ),
            )
        except ValueError:
            st.error("Mã khách hàng mô phỏng không hợp lệ.")
            return
    budget = st.number_input(
        "Ngân sách giả định",
        min_value=0.0,
        value=20.0,
        step=1.0,
        key="llm_retention.budget",
    )
    margin = st.number_input(
        "Biên lợi nhuận giả định",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="llm_retention.margin",
    )
    horizon = st.number_input(
        "Kỳ hạn (tháng)",
        min_value=1,
        max_value=120,
        value=12,
        step=1,
        key="llm_retention.horizon",
    )
    multiplier = st.number_input(
        "Hệ số chi phí liên hệ giả định",
        min_value=0.0,
        max_value=1000.0,
        value=1.0,
        step=1.0,
        key="llm_retention.contact_multiplier",
    )
    try:
        scenario = RetentionScenario(
            budget=budget,
            margin=margin,
            horizon_months=horizon,
            contact_cost_multiplier=multiplier,
        )
        result = simulate(inputs, scenario)
    except Exception:  # noqa: BLE001 -- Keep raw data/exception contents out of the UI.
        st.error("Không thể mô phỏng từ dữ liệu hoặc cấu hình hiện tại.")
        return
    allocations = result.allocations
    st.caption(
        f"Đủ điều kiện: {len(allocations)}/{result.input_count}. Cần đang hoạt động và tenure > 0."
    )
    st.metric("Chi phí phân bổ giả định", f"{allocations['cost'].sum():.2f}")
    st.metric("Giá trị ròng mô phỏng", f"{allocations['net_value'].sum():.2f}")
    if allocations.empty:
        st.info("Không có khách hàng đủ điều kiện mô phỏng.")
    elif (allocations["action_code"] == "A0").all():
        st.info(
            "Tất cả khách hàng chọn A0: không can thiệp theo các giả định hiện tại."
        )
    st.subheader("Phân bổ trong ngân sách")
    st.dataframe(allocations, hide_index=True)
    st.subheader("Khuyến nghị chưa giới hạn ngân sách")
    st.dataframe(result.recommendations, hide_index=True)
    st.subheader("Giả định hành động")
    st.dataframe(
        [
            {
                "action": a.code,
                "name": a.name,
                "contact_cost": a.contact_cost,
                "offer_cost": a.offer_cost,
                "acceptance_rate": a.acceptance_rate,
                "efficacy": a.efficacy,
                "expected_cost": a.expected_cost,
            }
            for a in scenario.actions()
        ],
        hide_index=True,
    )
