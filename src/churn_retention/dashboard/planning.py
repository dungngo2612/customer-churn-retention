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
