"""Main retention workspace: official Streamlit navigation, existing services."""

import base64
import os

import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state
from churn_retention.dashboard.theme import apply_theme
from churn_retention.entrypoints.streamlit.demo_security import require_safe_runtime

try:
    require_safe_runtime()
except ValueError:
    st.error("Ứng dụng chỉ hoạt động qua launcher cục bộ được hỗ trợ.")
    st.stop()

st.set_page_config(
    page_title="Trung tâm điều hành giữ chân khách hàng",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()
logo = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="252" height="40" viewBox="0 0 252 40">'
    '<rect x="0" y="0" width="40" height="40" rx="12" fill="#3949ab"/>'
    '<path d="M12 29l6 6 11-15M12 20h7" fill="none" stroke="white" '
    'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    '<text x="53" y="19" font-family="Arial,sans-serif" font-size="19"'
    ' font-weight="700" fill="#17243b">Giữ chân khách hàng</text>'
    '<text x="53" y="35" font-family="Arial,sans-serif" font-size="11"'
    ' fill="#526079">Trung tâm điều hành</text></svg>'
)
st.logo("data:image/svg+xml;base64," + base64.b64encode(logo.encode()).decode(), size="large")
page = st.navigation(
    [
        st.Page("views/overview.py", title="Tổng quan", icon=":material/dashboard:", default=True),
        st.Page(
            "views/customers.py",
            title="Khách hàng & dữ liệu",
            icon=":material/group:",
            url_path="khach-hang",
        ),
        st.Page(
            "views/analysis.py",
            title="Phân tích rời bỏ",
            icon=":material/insights:",
            url_path="phan-tich",
        ),
        st.Page(
            "views/plan.py",
            title="Kế hoạch giữ chân",
            icon=":material/account_balance_wallet:",
            url_path="ke-hoach",
        ),
        st.Page(
            "views/campaigns.py",
            title="Chiến dịch",
            icon=":material/folder_open:",
            url_path="chien-dich",
        ),
    ],
    position="sidebar",
    expanded=True,
)

try:
    ctx = state.context()
    with st.sidebar:
        st.caption("Dữ liệu đang chọn")
        if ctx.data:
            st.write(ctx.data.source)
            st.caption(f"{len(ctx.data.frame)} khách hàng · Đã kiểm tra")
        else:
            st.caption("Chưa có dữ liệu")
        if ctx.view:
            st.caption("Kết quả đang xem")
            st.write(ctx.view.run.source_name)
            ui.badge(
                ui.mode_label(ctx.view.run.execution_mode),
                "warning" if ctx.view.run.execution_mode == "mock" else "success",
            )
        else:
            ui.badge("Mô phỏng · không gọi API", "warning")
        with st.expander("Cấu hình & trạng thái"):
            st.write("Tham chiếu: Sẵn sàng" if ctx.form else "Tham chiếu: Chưa sẵn sàng")
            st.caption("LLM/RAG · Qwen qua Groq, dự phòng Gemini. Giữ nguyên model và định tuyến.")
            if ctx.form:
                st.text(f"Reference: {ctx.form.store.reference_version}")
            live = (
                os.environ.get("CHURN_DASHBOARD_ALLOW_LIVE") == "1"
                and os.environ.get("CHURN_DASHBOARD_PREVIEW") != "1"
            )
            st.write("API thật được bật trên máy" if live else "API thật chưa bật")
            if live:
                st.write(
                    "Khóa provider: Sẵn sàng"
                    if all(os.environ.get(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY"))
                    else "Khóa provider: Chưa đủ"
                )
            st.caption(
                (
                    "Môi trường cục bộ cho một quản trị viên; kết quả và chiến dịch "
                    "dùng chung giữa các cửa sổ trình duyệt."
                )
            )
        st.caption("Kinh tế và can thiệp là mô phỏng. Chưa gửi ưu đãi cho khách hàng.")
    if ctx.error:
        st.error(ctx.error)
    page.run()
except Exception:  # noqa: BLE001 -- Never expose raw archive, customer or provider errors.
    st.error(
        (
            "Không thể mở không gian làm việc an toàn. Dữ liệu đã lưu được giữ"
            " nguyên; không tự chạy lại phân tích."
        )
    )
