"""Customer profile shared by the input list and accepted analysis results."""

import pandas as pd
import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state
from churn_retention.dashboard.planning import recommendations
from churn_retention.entrypoints.streamlit.dashboard_llm_ui import DashboardView
from churn_retention.modules.retention.use_cases import evaluate_actions

LABELS = {
    "gender": "Giới tính",
    "SeniorCitizen": "Khách cao tuổi",
    "Partner": "Bạn đời",
    "Dependents": "Người phụ thuộc",
    "tenure": "Thời gian sử dụng (tháng)",
    "PhoneService": "Dịch vụ điện thoại",
    "MultipleLines": "Nhiều đường dây",
    "InternetService": "Dịch vụ Internet",
    "OnlineSecurity": "Bảo mật trực tuyến",
    "OnlineBackup": "Sao lưu trực tuyến",
    "DeviceProtection": "Bảo vệ thiết bị",
    "TechSupport": "Hỗ trợ kỹ thuật",
    "StreamingTV": "Truyền hình trực tuyến",
    "StreamingMovies": "Phim trực tuyến",
    "Contract": "Hợp đồng",
    "PaperlessBilling": "Hóa đơn điện tử",
    "PaymentMethod": "Thanh toán",
    "MonthlyCharges": "Cước tháng (USD)",
    "TotalCharges": "Tổng cước (USD)",
}
VALUES = {
    "No": "Không",
    "Yes": "Có",
    "Month-to-month": "Hàng tháng",
    "One year": "Một năm",
    "Two year": "Hai năm",
    "Male": "Nam",
    "Female": "Nữ",
    "Electronic check": "Séc điện tử",
    "Mailed check": "Séc giấy",
    "Bank transfer (automatic)": "Chuyển khoản tự động",
    "Credit card (automatic)": "Thẻ tín dụng tự động",
    "No internet service": "Không dùng Internet",
    "No phone service": "Không dùng điện thoại",
}


def translated(value: object) -> str:
    return VALUES.get(str(value), str(value))


def render(frame: pd.DataFrame, key: str, view: DashboardView | None = None) -> None:
    if frame.empty:
        return
    ids = frame.customerID.tolist()
    selected = st.session_state.get(state.CUSTOMER)
    cid = st.selectbox(
        "Mở hồ sơ khách hàng",
        ids,
        index=ids.index(selected) if selected in ids else None,
        placeholder="Chọn mã khách hàng trong danh sách",
        key=f"{key}.profile",
    )
    if cid is None:
        return
    st.session_state[state.CUSTOMER] = cid
    row = frame.set_index("customerID").loc[cid]
    with ui.panel("customer-profile"):
        ui.section("Hồ sơ khách hàng", "Thông tin nguồn và kết quả phân tích được tách riêng.")
        st.text(cid)
        prediction = None
        if view is not None:
            matches = view.predictions[view.predictions.customer_id == cid]
            if not matches.empty:
                prediction = matches.iloc[0]
        ui.summary(
            [
                (
                    "Điểm rời bỏ",
                    ui.percent(prediction.churn_prob) if prediction is not None else "—",
                    ui.risk_label(prediction.churn_prob)
                    if prediction is not None
                    else "Chưa phân tích",
                ),
                ("Cước hàng tháng", ui.money(row.MonthlyCharges), "USD / tháng"),
                (
                    "Trạng thái quan sát",
                    "Đã rời bỏ" if row.Churn == "Yes" else "Đang hoạt động",
                    f"{int(row.tenure)} tháng sử dụng",
                ),
            ]
        )
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Hợp đồng & dịch vụ**")
            for name in ("Contract", "InternetService", "TechSupport", "PaymentMethod"):
                st.write(f"{LABELS[name]}: {translated(row[name])}")
            with st.expander("Xem đủ 19 thuộc tính"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Thuộc tính": label, "Giá trị": translated(row[name])}
                            for name, label in LABELS.items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
        with right:
            st.markdown("**Lý do & bằng chứng tham chiếu**")
            if prediction is None or view is None:
                st.info("Khách hàng này chưa có kết quả thuộc bộ dữ liệu đang xem.")
                st.page_link("views/analysis.py", label="Đến phân tích rời bỏ →")
            else:
                st.text(prediction.reason)
                st.caption(
                    f"Dựa trên {int(prediction.reference_count)} khách hàng tham chiếu. "
                    "Điểm LLM chưa được hiệu chuẩn."
                )
                with st.expander("Nguồn kết quả"):
                    st.write(f"Chế độ: {ui.mode_label(view.run.execution_mode)}")
                    st.write(f"Nhà cung cấp: {prediction.provider}")
                    st.text(f"Reference: {prediction.reference_version}")
                    st.caption(
                        (
                            "Thông tin đang có: số lượng và phiên bản tham chiếu. Nội dung "
                            "từng hồ sơ tham chiếu chưa có trong kết quả đã lưu."
                        )
                    )
        if prediction is not None and view is not None:
            settings = st.session_state[state.SETTINGS]
            recs = recommendations(view, settings)
            rec = recs[recs.customerID == cid]
            if rec.empty:
                st.info(
                    (
                        "Khách đã rời bỏ hoặc chưa phát sinh thời gian sử dụng; không "
                        "thuộc diện can thiệp."
                    )
                )
            else:
                recommendation = rec.iloc[0]
                st.markdown(f"**Đề xuất hiện tại: {recommendation.action_name}**")
                st.caption(
                    f"Chi phí {ui.markdown_money(recommendation.cost)} · Lợi ích ròng mô phỏng "
                    f"{ui.markdown_money(recommendation.net_value)} · Ngưỡng hòa vốn "
                    f"{ui.percent(recommendation.breakeven_prob)}"
                )
                choices = evaluate_actions(
                    view.customers,
                    view.predictions,
                    settings.actions,
                    settings.margin,
                    settings.horizon,
                )
                choices = choices[choices.customerID == cid]
                with st.expander("So sánh các phương án hợp lệ"):
                    shown = choices[
                        ["action_code", "action_name", "cost", "benefit", "net_value"]
                    ].rename(
                        columns={
                            "action_code": "Gói",
                            "action_name": "Phương án",
                            "cost": "Chi phí (USD)",
                            "benefit": "Lợi ích (USD)",
                            "net_value": "Lợi ích ròng (USD)",
                        }
                    )
                    st.dataframe(
                        shown,
                        hide_index=True,
                        width="stretch",
                        column_config=ui.money_columns(shown),
                    )
                st.caption(
                    "Đề xuất chưa được thực thi. Có thể lập kế hoạch và lưu chiến dịch mô phỏng."
                )
                st.page_link("views/plan.py", label="Lập kế hoạch giữ chân →")
