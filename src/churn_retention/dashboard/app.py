"""Streamlit Dashboard for Customer Retention & Economic Optimization (Person 3).

Chạy dashboard bằng lệnh:
    streamlit run src/churn_retention/dashboard/app.py
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Đảm bảo Python nhận diện được package src/
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from churn_retention.adapters.predictor.fake import FakePredictor
from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    Condition,
    MANUAL_PAYMENTS,
    default_actions,
    evaluate_actions,
    recommend,
    sensitivity_by_margin,
)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ thống Thiết kế Giữ chân Khách hàng",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Bảng Điều Khiển Tối Ưu Hóa Chiến Lược Giữ Chân Khách Hàng")
st.caption("Khoa học Dịch vụ — Tối ưu hóa can thiệp kinh tế dựa trên Proxy CLV và Mô hình dự đoán Churn")

# --- SIDEBAR: CẤU HÌNH KINH TẾ & THAM SỐ CAN THIỆP ---
st.sidebar.header("⚙️ Cấu hình Kinh tế (Kịch bản cơ sở)")
base_margin = st.sidebar.slider(
    "Biên lợi nhuận đóng góp giả định (m)",
    min_value=0.10,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Dùng để tính Proxy CLV: V = MonthlyCharges * H * m",
)
horizon = st.sidebar.selectbox("Kỳ hạn đánh giá H (tháng)", options=[6, 12, 24], index=1)

st.sidebar.subheader("🛠️ Tùy chỉnh Hành động Giữ chân")
expander_actions = st.sidebar.expander("Chỉnh sửa Chi phí & Hiệu quả", expanded=False)
with expander_actions:
    st.markdown("**A1: Kỹ thuật & Bảo mật**")
    c1_contact = st.number_input("A1: Chi phí liên hệ ($)", value=2.0, step=0.5)
    c1_offer = st.number_input("A1: Chi phí ưu đãi ($)", value=15.0, step=1.0)
    c1_u = st.slider("A1: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05)
    c1_e = st.slider("A1: Mức giảm churn (e)", 0.1, 1.0, 0.25, 0.05)

    st.markdown("**A2: Hợp đồng & Thanh toán**")
    c2_contact = st.number_input("A2: Chi phí liên hệ ($)", value=2.0, step=0.5)
    c2_offer = st.number_input("A2: Chi phí ưu đãi ($)", value=20.0, step=1.0)
    c2_u = st.slider("A2: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.35, 0.05)
    c2_e = st.slider("A2: Mức giảm churn (e)", 0.1, 1.0, 0.40, 0.05)

    st.markdown("**A3: Giảm cước mục tiêu**")
    c3_contact = st.number_input("A3: Chi phí liên hệ ($)", value=3.0, step=0.5)
    c3_offer = st.number_input("A3: Chi phí voucher ($)", value=40.0, step=2.0)
    c3_u = st.slider("A3: Tỷ lệ chấp nhận (u)", 0.1, 1.0, 0.40, 0.05)
    c3_e = st.slider("A3: Mức giảm churn (e)", 0.1, 1.0, 0.50, 0.05)

# Dựng lại danh mục hành động theo tham số người dùng nhập
current_actions = (
    ActionConfig("A0", "Không can thiệp", 0, 0, 0, 0, is_no_action=True),
    ActionConfig(
        "A1", "Hỗ trợ kỹ thuật & bảo mật chủ động",
        c1_contact, c1_offer, c1_u, c1_e,
        all_of=(Condition("InternetService", "ne", "No"),),
        any_of=(Condition("TechSupport", "eq", "No"), Condition("OnlineSecurity", "eq", "No")),
    ),
    ActionConfig(
        "A2", "Tư vấn hợp đồng & thanh toán tự động",
        c2_contact, c2_offer, c2_u, c2_e,
        any_of=(Condition("Contract", "eq", "Month-to-month"), Condition("PaymentMethod", "in", MANUAL_PAYMENTS)),
    ),
    ActionConfig(
        "A3", "Ưu đãi giảm cước cam kết dài hạn",
        c3_contact, c3_offer, c3_u, c3_e,
        all_of=(Condition("Contract", "eq", "Month-to-month"), Condition("MonthlyCharges", "ge", 50.0)),
    ),
)


# --- NẠP DỮ LIỆU & CHẠY DỰ ĐOÁN (FAKE PREDICTOR CỦA NGƯỜI 1) ---
@st.cache_data
def load_data_and_predict():
    # Tìm file dataset
    possible_paths = [
        ROOT_DIR / "data" / "raw" / "telco_churn.csv",
        ROOT_DIR / "data" / "telco_churn.csv",
        ROOT_DIR / "telco_churn.csv",
    ]
    data_path = next((p for p in possible_paths if p.exists()), None)

    if data_path is None:
        # Tạo DataFrame giả lập nếu chưa copy file csv vào
        st.warning("⚠️ Chưa tìm thấy file `telco_churn.csv`. Đang sử dụng dữ liệu mô phỏng.")
        customers = pd.DataFrame({
            "customerID": ["CUST_A", "CUST_B", "CUST_C"],
            "MonthlyCharges": [80.0, 25.0, 65.0],
            "Contract": ["Month-to-month", "Month-to-month", "One year"],
            "InternetService": ["Fiber optic", "DSL", "DSL"],
            "TechSupport": ["No", "No", "Yes"],
            "OnlineSecurity": ["No", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "Churn": ["No", "No", "No"],
            "tenure": [12, 5, 24],
        })
    else:
        customers = pd.read_csv(data_path)

    # Sử dụng FakePredictor chuẩn của Người 1 để sinh xác suất
    predictor = FakePredictor(seed=42)
    probs = predictor.predict_batch(
        features=customers,
        customer_ids=customers["customerID"].tolist()
    )
    predictions = pd.DataFrame({
        "customer_id": customers["customerID"].tolist(),
        "churn_prob": probs,
    })

    return customers, predictions


customers_df, predictions_df = load_data_and_predict()

# --- CHẠY KHUYẾN NGHỊ VÀ TỔNG HỢP ---
recommendations = recommend(
    customers=customers_df,
    predictions=predictions_df,
    actions=current_actions,
    margin=base_margin,
    horizon_months=horizon,
)

# Lọc các khách hàng có can thiệp thực sự (loại A0)
interventions = recommendations[recommendations["action_code"] != "A0"]

# --- GIAO DIỆN HIỂN THỊ CHÍNH ---
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Kinh tế", "📋 Danh sách Đề xuất Can thiệp", "📈 Phân tích Độ nhạy (Sensitivity)"])

with tab1:
    st.subheader("Chỉ số Hiệu quả Chiến dịch Giữ chân")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Khách hàng đang hoạt động", f"{len(recommendations):,}")
    col2.metric("Số khách hàng được can thiệp (ENV > 0)", f"{len(interventions):,}")
    col3.metric("Tổng chi phí can thiệp dự kiến", f"${interventions['cost'].sum():,.2f}")
    col4.metric(
        "Tổng Lợi nhuận Ròng Kỳ vọng (ΔEV)",
        f"${interventions['net_value'].sum():,.2f}",
        delta=f"+{interventions['benefit'].sum():,.2f} Gross",
    )

    st.divider()
    st.subheader("Cơ cấu Hành động Được Chọn Tối Ưu")
    counts = recommendations["action_code"].value_counts().reset_index()
    counts.columns = ["Mã hành động", "Số lượng khách hàng"]
    action_names = {a.code: a.name for a in current_actions}
    counts["Tên hành động"] = counts["Mã hành động"].map(action_names)

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.dataframe(counts[["Mã hành động", "Tên hành động", "Số lượng khách hàng"]], use_container_width=True)
    with c_right:
        st.bar_chart(data=counts.set_index("Mã hành động")["Số lượng khách hàng"])

with tab2:
    st.subheader("Chi tiết Đề xuất Giữ chân Từng Khách hàng (Sắp xếp theo ΔEV giảm dần)")
    st.caption("Quy tắc: Chỉ can thiệp khi ΔEV > 0. Mỗi khách hàng chọn 1 hành động duy nhất có lợi ích ròng cao nhất.")

    # Tìm kiếm theo customerID
    search_id = st.text_input("🔍 Tìm kiếm theo Mã Khách Hàng (customerID):")
    display_df = recommendations.copy()
    if search_id:
        display_df = display_df[display_df["customerID"].str.contains(search_id, case=False)]

    st.dataframe(
        display_df.style.format({
            "churn_prob": "{:.2%}",
            "churn_prob_after": "{:.2%}",
            "breakeven_prob": "{:.2%}",
            "proxy_value": "${:.2f}",
            "cost": "${:.2f}",
            "benefit": "${:.2f}",
            "net_value": "${:+.2f}",
        }),
        use_container_width=True,
    )

with tab3:
    st.subheader("Phân tích Độ nhạy theo Biên lợi nhuận (Sensitivity Analysis)")
    st.markdown(
        "Do biên lợi nhuận $m$ là giả định kịch bản, bảng dưới đây mô phỏng sự thay đổi của tổng giá trị ròng "
        "và cơ cấu hành động khi $m$ thay đổi từ 20% đến 50%:"
    )

    sens_df = sensitivity_by_margin(
        customers=customers_df,
        predictions=predictions_df,
        margins=(0.20, 0.30, 0.40, 0.50),
        actions=current_actions,
        horizon_months=horizon,
    )

    st.dataframe(
        sens_df.style.format({
            "margin": "{:.0%}",
            "total_cost": "${:,.2f}",
            "total_benefit": "${:,.2f}",
            "total_net_value": "${:,.2f}",
        }),
        use_container_width=True,
    )

    st.line_chart(
        data=sens_df.set_index("margin")[["total_net_value", "total_cost", "total_benefit"]]
    )