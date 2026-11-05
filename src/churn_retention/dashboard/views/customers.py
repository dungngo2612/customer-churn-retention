"""Explicit dataset acceptance, searchable customer list and profiles."""

import streamlit as st

from churn_retention.contracts.application_security import safe_source_name
from churn_retention.dashboard import analytics, components as ui
from churn_retention.dashboard import customer_detail, state
from churn_retention.dashboard.i18n import t
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.entrypoints.streamlit.dashboard_llm_service import read_csv, validate_frame

ctx = state.context()
ui.header(
    t("cu_eyebrow"),
    t("cu_title"),
    t("cu_desc"),
)
import_tab, list_tab, eda_tab, seg_tab = st.tabs(
    [
        t("cu_tab_import"),
        t("cu_tab_list"),
        t("cu_tab_eda"),
        t("cu_tab_segments"),
    ]
)

with import_tab:
    with ui.panel("data-import"):
        ui.section(
            t("cu_import_title"),
            t("cu_import_desc"),
        )
        uploaded = st.file_uploader(
            t("cu_file_uploader"), type=["csv"], key="cuong_llm.csv"
        )
        with st.expander(t("cu_format_expander")):
            st.write(t("cu_format_body"))
            st.download_button(
                t("cu_download_sample"),
                services.synthetic_frame().to_csv(index=False),
                file_name="khach_hang_mau.csv",
                mime="text/csv",
            )
        if uploaded is not None and ctx.form is not None:
            try:
                frame = read_csv(uploaded.getvalue(), ctx.form)
                st.session_state.pop(state.INPUT_ERROR, None)
                source_name = safe_source_name(uploaded.name)
                if (
                    ctx.data is None
                    or ctx.data.source != source_name
                    or len(ctx.data.frame) != len(frame)
                ):
                    state.set_input(state.InputData(frame, source_name))
                st.success(t("cu_validate_ok").format(n=len(frame)))
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
                if st.button(t("cu_use_data_btn"), type="primary", key="data.accept"):
                    state.set_input(state.InputData(frame, source_name))
                    st.success(t("cu_accepted"))
                    st.rerun()
            except ValueError as error:
                st.session_state[state.INPUT_ERROR] = str(error)
                st.error(str(error))
                st.caption(t("cu_fix_caption"))
        elif ctx.form is None:
            st.error(ctx.error)
    st.write("")
    with ui.panel("data-sample"):
        ui.section(
            t("cu_sample_title"),
            f"{len(services.synthetic_frame())} {t('cu_sample_desc_suffix')}",
        )
        if st.button(t("cu_use_sample_btn"), key="data.sample", disabled=ctx.form is None) and ctx.form:
            frame = validate_frame(services.synthetic_frame(), ctx.form)
            state.set_input(
                state.InputData(frame, f"Dữ liệu mẫu · {len(frame)} khách hàng giả", sample=True)
            )
            st.rerun()
    if ctx.data is not None:
        st.info(t("cu_current_info").format(source=ctx.data.source, n=len(ctx.data.frame)))
        st.page_link("views/analysis.py", label=t("cu_link_analysis"))

with list_tab:
    if ctx.data is None:
        ui.empty(t("cu_empty_list"), t("cu_empty_list_desc"))
    else:
        data = ctx.data
        ui.badge(
            t("cu_sample_badge") if data.sample else t("cu_verified_badge"),
            "warning" if data.sample else "success",
        )
        st.caption(f"Nguồn: {data.source} · {len(data.frame)} khách hàng")
        cols = st.columns([2, 1, 1.5])
        search = ui.persist(
            cols[0].text_input,
            t("cu_search_label"),
            key="customers.search",
            placeholder=t("cu_search_placeholder"),
        )
        contract = ui.persist(
            cols[1].selectbox,
            t("cu_contract_label"),
            [t("cu_contract_all"), *sorted(data.frame.Contract.unique())],
            format_func=customer_detail.translated,
            key="customers.contract",
            choices=[t("cu_contract_all"), *sorted(data.frame.Contract.unique())],
        )
        order = ui.persist(
            cols[2].selectbox,
            t("cu_sort_label"),
            [t("cu_sort_id"), t("cu_sort_charges"), t("cu_sort_tenure")],
            key="customers.sort",
        )
        frame = data.frame[
            data.frame.customerID.str.contains(search, case=False, regex=False)
        ].copy()
        if contract != t("cu_contract_all"):
            frame = frame[frame.Contract == contract]
        sort_column = (
            "MonthlyCharges" if order == t("cu_sort_charges")
            else "tenure" if order == t("cu_sort_tenure")
            else "customerID"
        )
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
                "Churn": "Trạng thái quan sát",
            }
        )
        shown["Hợp đồng"] = shown["Hợp đồng"].map(customer_detail.translated)
        shown["Trạng thái quan sát"] = shown["Trạng thái quan sát"].map(
            {"Yes": "Đã rời bỏ", "No": "Đang hoạt động"}
        )
        ui.table(
            shown,
            "customers.list",
            column_config={"Cước tháng (USD)": st.column_config.NumberColumn(format="$%.2f")},
        )
        accepted_view = ctx.view if ctx.view and state.matches_input(ctx.view.run, data) else None
        customer_detail.render(frame, "customers", accepted_view)

with eda_tab:
    if ctx.data is None:
        ui.empty(t("cu_empty_list"), t("cu_empty_list_desc"))
    else:
        analytics.render_eda_tab(ctx.data.frame)

with seg_tab:
    if ctx.data is None:
        ui.empty(t("cu_empty_list"), t("cu_empty_list_desc"))
    else:
        analytics.render_segmentation_tab(ctx.data.frame)
