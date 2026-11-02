"""Small presentation components shared by the five Dashboard pages."""

import hashlib
import math
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn_retention.modules.campaigns.use_cases import safe_export_frame

INK = "#17243b"
INDIGO = "#3949ab"
GREEN = "#16734b"
AMBER = "#985709"
RED = "#b42335"


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


def percent(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


def date_label(value: str) -> str:
    return pd.Timestamp(value).tz_convert("Asia/Ho_Chi_Minh").strftime("%d/%m/%Y %H:%M")


def header(eyebrow: str, title: str, description: str) -> None:
    st.html(
        '<div class="page-heading"><div class="eyebrow">'
        f"{escape(eyebrow)}</div><h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p></div>"
    )


def section(title: str, description: str = "") -> None:
    st.subheader(title)
    if description:
        st.caption(description)


def summary(items: list[tuple[str, str, str]]) -> None:
    st.html(
        '<div class="summary-grid">'
        + "".join(
            '<div class="summary-item">'
            f'<div class="summary-label">{escape(label)}</div>'
            f'<div class="summary-value">{escape(value)}</div>'
            f'<div class="summary-note">{escape(note)}</div></div>'
            for label, value, note in items
        )
        + "</div>"
    )


def empty(title: str, description: str, *, icon: str = "○") -> None:
    st.html(
        '<div class="empty-state">'
        f'<div class="empty-icon" aria-hidden="true">{escape(icon)}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(description)}</p></div>"
    )


def badge(label: str, tone: str = "neutral") -> None:
    if tone not in {"neutral", "success", "warning", "danger"}:
        tone = "neutral"
    st.html(f'<span class="status status-{tone}">{escape(label)}</span>')


def mode_label(mode: str) -> str:
    return "Kết quả mô phỏng" if mode == "mock" else "Kết quả từ API thật"


def risk_label(probability: float) -> str:
    return "Cao" if probability > 0.7 else "Trung bình" if probability >= 0.4 else "Thấp"


def processing_label(code: str) -> str:
    return {
        "SUCCESS": "Đã lưu",
        "FAILED": "Không hoàn tất",
        "UNKNOWN": "Chưa xác định",
        "NOT_RUN": "Chưa chạy",
        "PENDING": "Đang chờ",
        "RUNNING": "Đang xử lý",
    }.get(code, code)


def page_slice(frame: pd.DataFrame, size: int, page: int) -> tuple[pd.DataFrame, int, int]:
    """Filter/sort before slicing; retain the frame's IDs and order exactly."""
    if size not in {10, 25, 50}:
        raise ValueError("Unsupported page size")
    pages = max(1, math.ceil(len(frame) / size))
    page = min(max(int(page), 1), pages)
    return frame.iloc[(page - 1) * size : page * size].copy(), page, pages


def table(
    frame: pd.DataFrame,
    key: str,
    *,
    column_config: dict | None = None,
) -> pd.DataFrame:
    """Native, horizontally scrollable table with explicit list pagination."""
    if frame.empty:
        empty("Không có kết quả phù hợp", "Thử đổi từ khóa hoặc nới bộ lọc.")
        return frame
    signature = hashlib.sha256(frame.to_json(orient="split").encode()).hexdigest()
    size_key, page_key = f"{key}.size", f"{key}.page"
    if st.session_state.get(f"{key}.filter") != signature:
        st.session_state[page_key] = 1
        st.session_state[f"{key}.filter"] = signature
    controls = st.columns([1, 1, 2], vertical_alignment="bottom")
    with controls[0]:
        size = st.selectbox("Số dòng mỗi trang", [10, 25, 50], key=size_key)
    pages = max(1, math.ceil(len(frame) / size))
    st.session_state[page_key] = min(int(st.session_state.get(page_key, 1)), pages)
    with controls[1]:
        page = st.number_input("Trang", 1, pages, step=1, key=page_key)
    visible, page, pages = page_slice(frame, size, page)
    with controls[2]:
        st.caption(f"{len(frame):,} kết quả · Trang {page}/{pages}")
    st.dataframe(
        safe_export_frame(visible),
        hide_index=True,
        width="stretch",
        height=min(560, 40 + 36 * len(visible)),
        column_config=column_config,
    )
    st.caption("Nút tải trong thanh công cụ bảng chỉ xuất trang đang xem.")
    return visible


def chart(figure: go.Figure, *, key: str | None = None) -> None:
    figure.update_layout(
        font={"family": "Arial, sans-serif", "size": 14, "color": INK},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 24, "t": 20, "b": 32},
        height=300,
        legend={"font": {"color": INK}},
        colorway=[INDIGO, GREEN, AMBER, RED],
    )
    figure.update_xaxes(color=INK, gridcolor="#e4e8f0", automargin=True)
    figure.update_yaxes(color=INK, gridcolor="#e4e8f0", automargin=True)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False}, key=key)


def calculation_note() -> None:
    with st.expander("Cách tính / Giả định"):
        st.markdown(
            "Giá trị thay thế **V = cước tháng × kỳ hạn × biên lợi nhuận**. "
            "Chi phí **C = chi phí liên hệ + tỷ lệ chấp nhận × ưu đãi**. "
            "Lợi ích **B = điểm rời bỏ × tỷ lệ chấp nhận × hiệu quả × V**; "
            "lợi ích ròng **ΔEV = B − C**. Đây là mô phỏng theo giả định, "
            "không phải CLV hay hiệu quả can thiệp đã đo lường."
        )
        st.markdown(
            "Chỉ đề xuất cho khách đang hoạt động: Churn đã quan sát = No và "
            "thời gian sử dụng > 0. Mỗi khách tối đa một gói có ΔEV dương. "
            "Phân bổ ngân sách dùng heuristic tham lam theo ΔEV/chi phí; "
            "không bảo đảm nghiệm tối ưu toàn cục. Điểm LLM chưa được hiệu chuẩn."
        )
