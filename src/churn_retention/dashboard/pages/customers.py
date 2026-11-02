"""Explicit dataset acceptance, searchable customer list and profiles."""

import streamlit as st

from churn_retention.contracts.application_security import safe_source_name
from churn_retention.dashboard import components as ui
from churn_retention.dashboard import customer_detail, state
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.entrypoints.streamlit.dashboard_llm_service import read_csv, validate_frame

ctx = state.context()
ui.header(
    "Không gian dữ liệu",
    "Khách hàng & dữ liệu",
    "Kiểm tra nguồn dữ liệu trước khi phân tích. Tệp được nhập không tự tạo dự đoán.",
)
import_tab, list_tab = st.tabs(["Nhập dữ liệu", "Danh sách khách hàng"])

with import_tab:
    with st.container(border=True):
        ui.section(
            "Nhập danh sách khách hàng",
            "CSV UTF-8 · tối đa 2 MB, 1.000 dòng · mã khách hàng duy nhất",
        )
        uploaded = st.file_uploader(
            "Chọn tệp dữ liệu khách hàng", type=["csv"], key="cuong_llm.csv"
        )
        with st.expander("Định dạng tệp & dữ liệu mẫu"):
            st.write(
                (
                    "Tệp cần đúng 19 thuộc tính, customerID và Churn đã quan sát. "
                    "Không đưa nhãn dự đoán vào Churn. Chưa hỗ trợ sửa trực tiếp; hãy "
                    "sửa tệp rồi nhập lại."
                )
            )
            st.download_button(
                "Tải tệp CSV mẫu (dữ liệu giả)",
                services.synthetic_frame().to_csv(index=False),
                file_name="khach_hang_mau.csv",
                mime="text/csv",
            )
        if uploaded is not None and ctx.form is not None:
            try:
                frame = read_csv(uploaded.getvalue(), ctx.form)
                st.session_state.pop(state.INPUT_ERROR, None)
                st.success(f"Đã kiểm tra: {len(frame)} dòng hợp lệ. Chưa chạy phân tích.")
                ui.table(
                    frame.rename(
                        columns={
                            "customerID": "Mã khách hàng",
                            "MonthlyCharges": "Cước tháng (USD)",
                            "Churn": "Churn quan sát",
                        }
                    ),
                    "upload.preview",
                )
                if st.button("Sử dụng dữ liệu này", type="primary", key="data.accept"):
                    state.set_input(state.InputData(frame, safe_source_name(uploaded.name)))
                    st.success("Đã chọn dữ liệu. Kết quả trước đó vẫn được giữ nguyên.")
                    st.rerun()
            except ValueError as error:
                st.session_state[state.INPUT_ERROR] = str(error)
                st.error(str(error))
                st.caption(
                    (
                        "Sửa các giá trị/cột được báo trong tệp rồi nhập lại. Dữ liệu đã "
                        "chọn trước đó không bị thay thế."
                    )
                )
        elif ctx.form is None:
            st.error(ctx.error)
    st.write("")
    with st.container(border=True):
        ui.section(
            "Khám phá bằng dữ liệu mẫu",
            f"{len(services.synthetic_frame())} hồ sơ tổng hợp. "
            "Đây là dữ liệu giả, dùng để trải nghiệm luồng làm việc.",
        )
        if st.button("Dùng dữ liệu mẫu", key="data.sample", disabled=ctx.form is None) and ctx.form:
            frame = validate_frame(services.synthetic_frame(), ctx.form)
            state.set_input(
                state.InputData(frame, f"Dữ liệu mẫu · {len(frame)} khách hàng giả", sample=True)
            )
            st.rerun()
    if ctx.data is not None:
        st.info(f"Đang chọn: {ctx.data.source} · {len(ctx.data.frame)} khách hàng")
        st.page_link("pages/analysis.py", label="Tiếp tục phân tích rời bỏ →")

with list_tab:
    if ctx.data is None:
        ui.empty("Chưa có danh sách khách hàng", "Chọn dữ liệu tại mục Nhập dữ liệu để bắt đầu.")
    else:
        data = ctx.data
        ui.badge(
            "Dữ liệu mẫu · không phải khách hàng thật" if data.sample else "Dữ liệu đã kiểm tra",
            "warning" if data.sample else "success",
        )
        st.caption(f"Nguồn: {data.source} · {len(data.frame)} khách hàng")
        cols = st.columns([2, 1, 1])
        search = cols[0].text_input(
            "Tìm theo mã khách hàng", key="customers.search", placeholder="Ví dụ: DEMO-001"
        )
        contract = cols[1].selectbox(
            "Hợp đồng",
            ["Tất cả", *sorted(data.frame.Contract.unique())],
            format_func=customer_detail.translated,
            key="customers.contract",
        )
        order = cols[2].selectbox(
            "Sắp xếp",
            ["Mã khách tăng dần", "Cước tháng giảm dần", "Thời gian sử dụng giảm dần"],
            key="customers.sort",
        )
        frame = data.frame[
            data.frame.customerID.str.contains(search, case=False, regex=False)
        ].copy()
        if contract != "Tất cả":
            frame = frame[frame.Contract == contract]
        sort_column = {
            "Mã khách tăng dần": "customerID",
            "Cước tháng giảm dần": "MonthlyCharges",
            "Thời gian sử dụng giảm dần": "tenure",
        }[order]
        frame = frame.sort_values(
            [sort_column, "customerID"] if sort_column != "customerID" else ["customerID"],
            ascending=[False, True] if sort_column != "customerID" else True,
            kind="stable",
        )
        shown = frame[["customerID", "Contract", "MonthlyCharges", "tenure", "Churn"]].rename(
            columns={
                "customerID": "Mã khách hàng",
                "Contract": "Hợp đồng",
                "MonthlyCharges": "Cước tháng (USD)",
                "tenure": "Sử dụng (tháng)",
                "Churn": "Churn quan sát",
            }
        )
        ui.table(
            shown,
            "customers.list",
            column_config={"Cước tháng (USD)": st.column_config.NumberColumn(format="$%.2f")},
        )
        accepted_view = ctx.view if ctx.view and state.matches_input(ctx.view.run, data) else None
        customer_detail.render(frame, "customers", accepted_view)
