"""Executive overview; rendering never performs inference."""

import plotly.graph_objects as go
import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import state
from churn_retention.dashboard.planning import allocation, totals
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services

ctx = state.context()
ui.header("Trung tâm điều hành", "Tổng quan", "Nắm tình hình khách hàng và chọn bước tiếp theo.")

if ctx.view is None:
    ui.empty(
        "Bắt đầu từ dữ liệu khách hàng",
        (
            "Nhập tệp CSV hoặc chọn dữ liệu mẫu. Sau khi kiểm tra, chủ động "
            "chạy phân tích để xem nguy cơ rời bỏ."
        ),
    )
    st.page_link(
        "views/customers.py", label="Nhập dữ liệu khách hàng →", icon=":material/upload_file:"
    )
    if ctx.data is not None:
        st.success(f"Đã chọn {len(ctx.data.frame)} khách hàng · {ctx.data.source}")
        st.page_link("views/analysis.py", label="Tiếp tục đến phân tích →")
else:
    view = ctx.view
    ui.badge(
        ui.mode_label(view.run.execution_mode),
        "warning" if view.run.execution_mode == "mock" else "success",
    )
    st.caption(
        f"Nguồn đang xem: {view.run.source_name} · {len(view.customers)} khách có kết quả đã lưu"
    )
    high = int((view.predictions.churn_prob > 0.7).sum())
    campaigns = services.list_campaigns()
    ui.summary(
        [
            ("Đã phân tích", f"{len(view.customers):,}", "Khách trong lần phân tích đang chọn"),
            ("Nguy cơ cao", str(high), "Điểm LLM trên 70% · chưa hiệu chuẩn"),
            (
                "Đang hoạt động",
                str(int(((view.customers.Churn == "No") & (view.customers.tenure > 0)).sum())),
                "Theo dữ liệu quan sát",
            ),
            ("Chiến dịch đã lưu", str(len(campaigns)), "Kế hoạch mô phỏng · chưa thực thi"),
        ]
    )
    left, right = st.columns([1.1, 1])
    with left, st.container(border=True):
        ui.section("Phân bố nguy cơ", "Phân nhóm để xem xét; không phải xác suất đã hiệu chuẩn.")
        counts = (
            view.predictions.churn_prob.map(ui.risk_label)
            .value_counts()
            .reindex(["Thấp", "Trung bình", "Cao"], fill_value=0)
        )
        if len(view.predictions) >= 3:
            fig = go.Figure(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=[ui.GREEN, ui.AMBER, ui.RED],
                    text=counts.values,
                    textposition="auto",
                )
            )
            fig.update_yaxes(title="Số khách hàng", dtick=1)
            ui.chart(fig)
        else:
            st.dataframe(
                counts.rename_axis("Nguy cơ").reset_index(name="Số khách hàng"), hide_index=True
            )
        st.page_link("views/analysis.py", label="Xem kết quả và lý do →")
    with right, st.container(border=True):
        ui.section("Kế hoạch đang xây dựng", "Theo lần phân tích và giả định đang chọn.")
        plan = totals(allocation(view, st.session_state[state.SETTINGS]))
        st.metric("Lợi ích ròng dự kiến", ui.money(plan["total_net_value"]))
        st.write(f"{plan['n_intervened']} khách được chọn · Chi phí {ui.money(plan['total_cost'])}")
        st.caption("Mô phỏng theo giả định về tỷ lệ chấp nhận và hiệu quả can thiệp.")
        st.page_link("views/plan.py", label="Tiếp tục lập kế hoạch →")
        st.divider()
        st.page_link("views/campaigns.py", label="Mở chiến dịch đã lưu →")
    with st.expander("Báo cáo điều hành"):
        report = (
            f"# Tóm tắt giữ chân khách hàng\n\nNguồn: {view.run.source_name}\n"
            f"Chế độ: {ui.mode_label(view.run.execution_mode)}\n"
            f"Khách đã phân tích: {len(view.customers)}\nNguy cơ cao: {high}\n"
            f"Chi phí: {ui.money(plan['total_cost'])}\n"
            f"Lợi ích ròng mô phỏng: {ui.money(plan['total_net_value'])}\n\n"
            "Điểm LLM chưa hiệu chuẩn. Kế hoạch chưa được thực thi.\n"
        )
        st.download_button(
            "Tải báo cáo tổng quan (.md)",
            report,
            file_name=f"{view.run.execution_mode}_retention_summary.md",
            mime="text/markdown",
        )

st.divider()
ui.section("Một luồng làm việc, từ dữ liệu đến kế hoạch")
steps = st.columns(3)
for col, title, text, page in zip(
    steps,
    ["01 · Kiểm tra dữ liệu", "02 · Hiểu nguy cơ", "03 · Lập phương án"],
    [
        "Nhập và rà soát hồ sơ khách hàng.",
        "Phân tích từng khách hoặc lô tối đa 5.",
        "Phân bổ ngân sách và lưu chiến dịch.",
    ],
    ["customers", "analysis", "plan"],
    strict=True,
):
    with col:
        st.markdown(f"**{title}**")
        st.write(text)
        st.page_link(f"views/{page}.py", label="Mở không gian →")
