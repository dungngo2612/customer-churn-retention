"""Presentation orchestration of existing retention use cases; no new formula."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from churn_retention.entrypoints.streamlit.dashboard_llm_ui import DashboardView
from churn_retention.modules.campaigns import CampaignStatus
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.retention.use_cases import ActionConfig, default_actions, recommend


@dataclass(frozen=True)
class Settings:
    budget: float = 15000.0
    margin: float = 0.5
    horizon: int = 12
    actions: tuple[ActionConfig, ...] = field(default_factory=default_actions)


def recommendations(view: DashboardView, settings: Settings) -> pd.DataFrame:
    if not math.isfinite(
        math.fsum(
            float(v) * settings.horizon * settings.margin for v in view.customers.MonthlyCharges
        )
    ):
        raise ValueError("Non-finite economics")
    return recommend(
        view.customers, view.predictions, settings.actions, settings.margin, settings.horizon
    )


def allocation(view: DashboardView, settings: Settings) -> pd.DataFrame:
    recommendations(view, settings)  # Preserve the former Dashboard finite-value boundary.
    return select_actions_under_budget(
        view.customers,
        view.predictions,
        settings.budget,
        settings.actions,
        settings.margin,
        settings.horizon,
    )


def totals(frame: pd.DataFrame) -> dict:
    selected = frame[frame.action_code != "A0"]
    return {
        "total_cost": float(selected.cost.sum()),
        "total_benefit": float(selected.benefit.sum()),
        "total_net_value": float(selected.net_value.sum()),
        "n_intervened": len(selected),
        "n_total": len(frame),
    }


def campaign_arguments(view: DashboardView, settings: Settings, name: str) -> dict:
    frame = allocation(view, settings)
    return {
        "name": name,
        "dataset_id": hashlib.sha256(view.customers.to_json(orient="records").encode()).hexdigest(),
        "model_version": view.run.predictor_ref,
        "scoring_run_id": view.run.run_id,
        "budget": settings.budget,
        "assumptions": {
            "margin": settings.margin,
            "horizon_months": settings.horizon,
            "actions": [asdict(a) for a in settings.actions],
            "prediction_source": "llm-rag-v2",
            "execution_mode": view.run.execution_mode,
            "probability_calibrated": False,
            "reference_version": view.run.reference_version,
            "reference_sha256": view.run.reference_sha256,
            "source_run_dataset_digest": view.run.dataset_digest,
            "model_version_semantics": (
                "predictor configuration reference; actual model version unavailable"
            ),
        },
        "results": {
            "summary": totals(frame),
            "allocations": json.loads(frame.to_json(orient="records")),
        },
        "status": CampaignStatus.SIMULATED,
    }


def executive_report(view: DashboardView, settings: Settings) -> str:
    """Retain the former unconstrained economic report alongside the selected plan."""
    recs = recommendations(view, settings)
    proposed = totals(recs)
    budgeted = totals(allocation(view, settings))
    cost, net = proposed["total_cost"], proposed["total_net_value"]
    mode = "Mô phỏng" if view.run.execution_mode == "mock" else "API thật"
    roi = f"{net / cost:+.1%}" if cost > 0 else "Không xác định (chi phí bằng 0)"
    lines = [
        "# Báo cáo giữ chân khách hàng",
        "",
        f"Nguồn: {view.run.source_name}",
        f"Chế độ kết quả: {mode}",
        f"Lần phân tích: {view.run.run_id}",
        f"Biên lợi nhuận: {settings.margin:.0%}; kỳ hạn: {settings.horizon} tháng.",
        "",
        "## Đề xuất trước giới hạn ngân sách",
        "",
        f"Khách có kết quả: {len(view.customers)}; đủ điều kiện: {len(recs)}.",
        f"Khách đề xuất can thiệp: {proposed['n_intervened']}; "
        f"không can thiệp: {len(recs) - proposed['n_intervened']}.",
        f"Chi phí: ${cost:,.2f}; lợi ích: ${proposed['total_benefit']:,.2f}; "
        f"lợi ích ròng: ${net:,.2f}; ROI mô phỏng: {roi}.",
        "",
        "| Gói | Số khách | Chi phí (USD) | Lợi ích (USD) | Lợi ích ròng (USD) |",
        "|---|---:|---:|---:|---:|",
    ]
    for code, rows in recs.groupby("action_code", sort=True):
        lines.append(
            f"| {code} | {len(rows)} | {rows.cost.sum():.2f} | "
            f"{rows.benefit.sum():.2f} | {rows.net_value.sum():.2f} |"
        )
    lines += [
        "",
        "## Phương án theo ngân sách đang chọn",
        "",
        f"Ngân sách: ${settings.budget:,.2f}; chi phí: ${budgeted['total_cost']:,.2f}; "
        f"còn lại: ${settings.budget - budgeted['total_cost']:,.2f}.",
        f"Khách được chọn: {budgeted['n_intervened']}; "
        f"lợi ích ròng mô phỏng: ${budgeted['total_net_value']:,.2f}.",
        "",
        "Điểm LLM chưa hiệu chuẩn. Kinh tế là mô phỏng theo giả định. "
        "Phân bổ tham lam không bảo đảm tối ưu toàn cục. Chưa thực thi can thiệp.",
    ]
    return "\n".join(lines) + "\n"
