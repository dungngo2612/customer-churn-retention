"""Project accepted predictions into Cường's neutral economic engine."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from churn_retention.contracts.llm_prediction_record import (
    FEATURE_NAMES,
    PredictionRecord,
)
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    default_actions,
    evaluate_actions,
    recommend,
)


class CustomerContext(BaseModel):
    """Explicit business metadata; never inferred from the predicted label."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    customer_ref: str = Field(min_length=1, max_length=100, pattern=r"^\S(?:.*\S)?$")
    observed_active: StrictBool | None = None


class RetentionScenario(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, extra="forbid", allow_inf_nan=False
    )
    budget: StrictFloat = Field(default=20.0, ge=0)
    margin: StrictFloat = Field(default=0.5, gt=0, le=1)
    horizon_months: StrictInt = Field(default=12, gt=0)
    contact_cost_multiplier: StrictFloat = Field(default=1.0, ge=0)

    def actions(self) -> tuple[ActionConfig, ...]:
        return tuple(
            replace(a, contact_cost=a.contact_cost * self.contact_cost_multiplier)
            for a in default_actions()
        )


@dataclass(frozen=True)
class SimulationResult:
    options: pd.DataFrame
    recommendations: pd.DataFrame
    allocations: pd.DataFrame
    scenario: RetentionScenario
    input_count: int
    synthetic: bool


def project_inputs(
    inputs: Sequence[tuple[PredictionRecord, CustomerContext]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Churn exists ONLY in this compatibility projection of observed status.

    This does not label a dataset or alter LLM inputs. Unknown status is rejected.
    Multiple predictions for one entity require an explicit upstream choice.
    """
    customers: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    seen_customers: set[str] = set()
    seen_records: set[object] = set()
    for record, context in inputs:
        if context.observed_active is None:
            raise ValueError("Cần xác nhận trạng thái hoạt động từ ngữ cảnh nghiệp vụ.")
        if context.customer_ref in seen_customers or record.record_id in seen_records:
            raise ValueError(
                "Chỉ chọn một prediction cho mỗi khách hàng; không được trùng ID."
            )
        seen_customers.add(context.customer_ref)
        seen_records.add(record.record_id)
        customers.append(
            {
                **record.features(),
                "customerID": context.customer_ref,
                "Churn": "No" if context.observed_active else "Yes",
            }
        )
        predictions.append(
            {
                "customer_id": context.customer_ref,
                "churn_prob": record.prediction.churn_probability,
            }
        )
    return (
        pd.DataFrame(customers, columns=[*FEATURE_NAMES, "customerID", "Churn"]),
        pd.DataFrame(predictions, columns=["customer_id", "churn_prob"]),
    )


def simulate(
    inputs: Sequence[tuple[PredictionRecord, CustomerContext]],
    scenario: RetentionScenario,
) -> SimulationResult:
    if len({record.origin for record, _ in inputs}) > 1:
        raise ValueError("Không trộn dữ liệu demo và kết quả LLM trong một mô phỏng.")
    customers, predictions = project_inputs(inputs)
    actions = scenario.actions()
    if any(not math.isfinite(a.expected_cost) for a in actions):
        raise ValueError("Chi phí giả định vượt giới hạn số hữu hạn.")
    for record, _ in inputs:
        charge = record.features()["MonthlyCharges"]
        assert isinstance(charge, (int, float))
        try:
            value = charge * scenario.horizon_months * scenario.margin
        except OverflowError:
            raise ValueError("Giá trị mô phỏng vượt giới hạn số hữu hạn.") from None
        if not math.isfinite(value):
            raise ValueError("Giá trị mô phỏng vượt giới hạn số hữu hạn.")
    return SimulationResult(
        options=evaluate_actions(
            customers, predictions, actions, scenario.margin, scenario.horizon_months
        ),
        recommendations=recommend(
            customers, predictions, actions, scenario.margin, scenario.horizon_months
        ),
        allocations=select_actions_under_budget(
            customers,
            predictions,
            scenario.budget,
            actions,
            scenario.margin,
            scenario.horizon_months,
        ),
        scenario=scenario,
        input_count=len(inputs),
        synthetic=bool(inputs) and inputs[0][0].origin == "synthetic",
    )
