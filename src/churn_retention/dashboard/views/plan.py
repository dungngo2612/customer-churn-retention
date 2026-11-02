"""Three-step planning UI over unchanged retention and campaign use cases."""

from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn_retention.dashboard import components as ui
from churn_retention.dashboard import customer_detail, state
from churn_retention.dashboard.planning import (
    Settings,
    allocation,
    campaign_arguments,
    recommendations,
    totals,
)
from churn_retention.entrypoints.streamlit import dashboard_llm_ui as services
from churn_retention.modules.retention.use_cases import sensitivity_by_margin

STEPS = ["1 · Thiết lập", "2 · Xem phương án", "3 · Tạo chiến dịch"]
ctx = state.context()
ui.header(
    "Từ phân tích đến hành động",
    "Kế hoạch giữ chân",
    "Xây dựng phương án ngân sách và lưu chiến dịch mô phỏng. Chưa thực thi ưu đãi.",
)

if ctx.view is None:
    ui.empty(
        "Cần kết quả phân tích trước khi lập kế hoạch",
        "Mở một lần phân tích đã lưu hoặc chạy phân tích khách hàng đã chọn.",
    )
    st.page_link("views/analysis.py", label="Đến phân tích rời bỏ →")
    st.stop()

view = ctx.view
ui.badge(
    ui.mode_label(view.run.execution_mode),
    "warning" if view.run.execution_mode == "mock" else "success",
)
st.caption(
    f"Nguồn phương án: {view.run.source_name} · {len(view.customers)} khách có kết quả đã lưu"
)
if ctx.data and not state.matches_input(view.run, ctx.data):
    st.warning(
        (
            "Phương án dùng snapshot của lần phân tích đang mở; dữ liệu mới "
            "nhập chưa được dùng cho kế hoạch này."
        )
    )
if "_plan_step" not in st.session_state:
    st.session_state["_plan_step"] = st.session_state.get("retention.plan_step", STEPS[0])
step = st.radio(
    "Tiến trình lập kế hoạch",
    STEPS,
    horizontal=True,
    key="_plan_step",
    label_visibility="collapsed",
)
st.session_state["retention.plan_step"] = step
settings = st.session_state[state.SETTINGS]

if step == STEPS[0]:
    ui.section(
        "Thiết lập ngân sách & giả định",
        "Bấm Áp dụng thiết lập để giữ các giá trị khi chuyển trang. Không gọi LLM.",
    )
    with st.form("plan.settings"):
        left, right = st.columns(2)
        with left:
            budget = st.number_input(
                "Ngân sách (USD)",
                min_value=0.0,
                value=float(settings.budget),
                step=10.0,
                key="e2e_budget_input",
            )
            margin = st.slider(
                "Biên lợi nhuận giả định", 0.10, 0.70, settings.margin, 0.05, key="plan.margin"
            )
        with right:
            horizon = st.selectbox(
                "Kỳ hạn đánh giá (tháng)",
                [6, 12, 24],
                index=[6, 12, 24].index(settings.horizon),
                key="plan.horizon",
            )
            st.caption(
                (
                    "USD được giữ nguyên. Tỷ lệ chấp nhận và hiệu quả là giả định, "
                    "chưa được đo lường từ chiến dịch."
                )
            )
        actions = [settings.actions[0]]
        for i, action in enumerate(settings.actions[1:], 1):
            with st.expander(f"{action.code} · {action.name}"):
                cols = st.columns(2)
                contact = cols[0].number_input(
                    "Chi phí liên hệ (USD)",
                    0.0,
                    1000000.0,
                    float(action.contact_cost),
                    step=0.5,
                    key=f"c{i}_c",
                )
                offer = cols[1].number_input(
                    "Chi phí ưu đãi (USD)",
                    0.0,
                    1000000.0,
                    float(action.offer_cost),
                    step=1.0,
                    key=f"c{i}_o",
                )
                acceptance = cols[0].slider(
                    "Tỷ lệ chấp nhận", 0.1, 1.0, float(action.acceptance_rate), 0.05, key=f"c{i}_u"
                )
                efficacy = cols[1].slider(
                    "Mức giảm rời bỏ giả định",
                    0.1,
                    1.0,
                    float(action.efficacy),
                    0.05,
                    key=f"c{i}_e",
                )
                actions.append(
                    replace(
                        action,
                        contact_cost=contact,
                        offer_cost=offer,
                        acceptance_rate=acceptance,
                        efficacy=efficacy,
                    )
                )
        if st.form_submit_button("Áp dụng thiết lập", type="primary"):
            st.session_state[state.SETTINGS] = Settings(budget, margin, horizon, tuple(actions))
            st.success("Đã áp dụng và giữ thiết lập. Chọn bước 2 để xem phương án.")
    ui.calculation_note()
    with st.expander("Chi phí và mức giảm kỳ vọng theo gói"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Gói": a.code,
                        "Phương án": a.name,
                        "Chi phí kỳ vọng (USD)": a.expected_cost,
                        "Mức giảm hiệu dụng": ui.percent(a.effective_reduction),
                    }
                    for a in st.session_state[state.SETTINGS].actions
                ]
            ),
            hide_index=True,
            width="stretch",
        )

elif step == STEPS[1]:
    frame = allocation(view, settings)
    summary = totals(frame)
    spent, earned = summary["total_cost"], summary["total_net_value"]
    ui.section(
        "Phương án phân bổ",
        f"Biên lợi nhuận {settings.margin:.0%} · Kỳ hạn {settings.horizon} tháng · Mô phỏng",
    )
    ui.summary(
        [
            ("Ngân sách sử dụng", ui.money(spent), f"Còn lại {ui.money(settings.budget - spent)}"),
            (
                "Khách được chọn",
                str(summary["n_intervened"]),
                f"Trong {summary['n_total']} khách đủ điều kiện",
            ),
            (
                "Lợi ích ròng dự kiến",
                ui.money(earned),
                f"Lợi ích trước chi phí {ui.money(summary['total_benefit'])}",
            ),
            (
                "ROI dự kiến",
                f"{earned / spent:+.1%}" if spent > 0 else "—",
                "Mô phỏng" if spent > 0 else "Không xác định khi chi phí bằng 0",
            ),
        ]
    )
    if not summary["n_intervened"]:
        recs = recommendations(view, settings)
        if not recs.empty and (recs.action_code != "A0").any():
            st.warning(
                (
                    "Ngân sách hiện tại chưa chọn được gói có lợi ích dương. Tăng ngân"
                    " sách hoặc xem lại giả định tại bước 1."
                )
            )
        else:
            st.info(
                (
                    "Không có phương án can thiệp lợi ích dương theo điều kiện và giả "
                    "định hiện tại. Có thể lưu phương án không can thiệp."
                )
            )
    columns = {
        "customerID": "Mã khách hàng",
        "action_name": "Phương án",
        "churn_prob": "Điểm rời bỏ",
        "cost": "Chi phí (USD)",
        "benefit": "Lợi ích (USD)",
        "net_value": "Lợi ích ròng (USD)",
    }
    shown = frame[list(columns)].rename(columns=columns)
    shown["Điểm rời bỏ"] = shown["Điểm rời bỏ"].map(ui.percent)
    ui.table(
        shown,
        "plan.allocations",
        column_config={
            label: st.column_config.NumberColumn(format="$%.2f")
            for label in ["Chi phí (USD)", "Lợi ích (USD)", "Lợi ích ròng (USD)"]
        },
    )
    customer_detail.render(view.customers, "plan", view)
    st.button(
        "Tiếp tục tạo chiến dịch →", type="primary", on_click=state.plan_step, args=(STEPS[2],)
    )
    ui.calculation_note()
    with st.expander("Khảo sát ngân sách & độ nhạy"):
        st.caption(
            (
                "Các phương án so sánh dùng cùng snapshot đã lưu; không dự đoán "
                "lại và không bảo đảm nghiệm tối ưu."
            )
        )
        amounts = sorted(
            {
                0.0,
                settings.budget * 0.25,
                settings.budget * 0.5,
                settings.budget,
                float(recommendations(view, settings).cost.sum()),
            }
        )
        curve = pd.DataFrame(
            [
                {
                    "Ngân sách (USD)": amount,
                    "Lợi ích ròng (USD)": totals(
                        allocation(view, replace(settings, budget=amount))
                    )["total_net_value"],
                }
                for amount in amounts
            ]
        )
        if len(curve) > 1:
            fig = go.Figure(
                go.Scatter(
                    x=curve["Ngân sách (USD)"], y=curve["Lợi ích ròng (USD)"], mode="lines+markers"
                )
            )
            fig.update_xaxes(title="Ngân sách (USD)")
            fig.update_yaxes(title="Lợi ích ròng (USD)")
            ui.chart(fig)
        sensitivity = sensitivity_by_margin(
            view.customers,
            view.predictions,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            settings.actions,
            settings.horizon,
        )
        sensitivity = sensitivity.rename(
            columns={
                "margin": "Biên lợi nhuận",
                "n_customers": "Khách đủ điều kiện",
                "n_intervened": "Khách can thiệp",
                "total_cost": "Chi phí (USD)",
                "total_benefit": "Lợi ích (USD)",
                "total_net_value": "Lợi ích ròng (USD)",
            }
        )
        st.dataframe(sensitivity, hide_index=True, width="stretch")

else:
    args = campaign_arguments(view, settings, "")
    summary = args["results"]["summary"]
    ui.section(
        "Lưu phương án thành chiến dịch",
        (
            "Lưu snapshot bất biến của dữ liệu, kết quả và giả định. Chưa gửi "
            "ưu đãi hoặc thực thi can thiệp."
        ),
    )
    with st.container(border=True):
        st.write(f"**Nguồn:** {view.run.source_name}")
        st.write(
            f"**Ngân sách:** {ui.money(settings.budget)} · "
            f"**Khách được chọn:** {summary['n_intervened']}"
        )
        st.write(
            f"**Chi phí dự kiến:** {ui.money(summary['total_cost'])} · "
            f"**Lợi ích ròng:** {ui.money(summary['total_net_value'])}"
        )
        with st.form("plan.campaign"):
            name = st.text_input(
                "Tên chiến dịch",
                value="Kế hoạch giữ chân khách hàng",
                max_chars=200,
                key="e2e_camp_name_input",
            )
            confirmed = st.checkbox(
                "Tôi xác nhận lưu phương án mô phỏng với nguồn và ngân sách trên",
                key="plan.save_confirm",
            )
            save = st.form_submit_button("Lưu chiến dịch", type="primary", key="btn_save_e2e_camp")
        if save:
            if not confirmed or not name.strip():
                st.error("Nhập tên chiến dịch và xác nhận phương án trước khi lưu.")
            else:
                try:
                    args["name"] = name
                    campaign = services.create_campaign(**args)
                    st.session_state[state.CAMPAIGN] = campaign.campaign_id
                    st.session_state["last_saved_camp_id"] = campaign.campaign_id
                    st.success("Đã lưu chiến dịch. Phương án chưa được thực thi.")
                except Exception:  # noqa: BLE001 -- Keep accepted snapshots and hide database internals.
                    st.error(
                        (
                            "Không thể lưu chiến dịch. Dữ liệu đã lưu được giữ nguyên; bạn có "
                            "thể thử lại."
                        )
                    )
        if st.session_state.get("last_saved_camp_id"):
            st.page_link(
                "views/campaigns.py",
                label="Mở chiến dịch đã lưu →",
                icon=":material/arrow_forward:",
            )
