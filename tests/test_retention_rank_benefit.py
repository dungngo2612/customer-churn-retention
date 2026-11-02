"""Hand-calculated regression tests for retention ranking and action economics."""

import pandas as pd
import pytest

from churn_retention.modules.retention.use_cases import (
    default_actions,
    evaluate_actions,
    recommend,
)


@pytest.fixture
def retention_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "customerID": ["C-ZERO", "C-B", "C-HIGH", "C-A"],
            "MonthlyCharges": [100.0] * 4,
            "Contract": ["Month-to-month"] * 4,
            "InternetService": ["Fiber optic"] * 4,
            "TechSupport": ["No"] * 4,
            "OnlineSecurity": ["No"] * 4,
            "PaymentMethod": ["Electronic check"] * 4,
            "Churn": ["No"] * 4,
            "tenure": [12] * 4,
        }
    )
    # Deliberately use a different order so probabilities must be joined by ID.
    predictions = pd.DataFrame(
        {
            "customer_id": ["C-A", "C-HIGH", "C-ZERO", "C-B"],
            "churn_prob": [0.5, 0.9, 0.01, 0.5],
        }
    )
    return customers, predictions


def test_rank_orders_net_value_then_customer_id(
    retention_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    customers, predictions = retention_inputs

    ranked = recommend(customers, predictions, margin=0.5, horizon_months=12)

    # V = 100 * 12 * 0.5 = 600; A3 has cost 19 and reduction 0.2.
    # HIGH: 0.9 * 0.2 * 600 - 19 = 89; A/B: 0.5 * 0.2 * 600 - 19 = 41.
    # ZERO's best intervention is A1: 0.01 * 0.1 * 600 - 8 = -7.4, so use A0.
    assert ranked["customerID"].tolist() == ["C-HIGH", "C-A", "C-B", "C-ZERO"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert ranked["action_code"].tolist() == ["A3", "A3", "A3", "A0"]
    assert ranked["net_value"].tolist() == pytest.approx([89.0, 41.0, 41.0, 0.0])
    assert ranked["benefit"].tolist() == pytest.approx([108.0, 60.0, 60.0, 0.0])


@pytest.mark.parametrize(
    ("customer_id", "action_code", "expected_benefit", "expected_net_value"),
    [
        ("C-A", "A0", 0.0, 0.0),
        ("C-A", "A1", 30.0, 22.0),
        ("C-A", "A2", 42.0, 33.0),
        ("C-A", "A3", 60.0, 41.0),
        ("C-ZERO", "A0", 0.0, 0.0),
        ("C-ZERO", "A1", 0.6, -7.4),
        ("C-ZERO", "A2", 0.84, -8.16),
        ("C-ZERO", "A3", 1.2, -17.8),
    ],
)
def test_evaluate_actions_matches_hand_calculated_benefit_and_net_value(
    retention_inputs: tuple[pd.DataFrame, pd.DataFrame],
    customer_id: str,
    action_code: str,
    expected_benefit: float,
    expected_net_value: float,
) -> None:
    customers, predictions = retention_inputs

    options = evaluate_actions(customers, predictions, margin=0.5, horizon_months=12)
    option = options.set_index(["customerID", "action_code"]).loc[(customer_id, action_code)]

    # V = 600, reductions A0..A3 = 0, 0.10, 0.14, 0.20; costs = 0, 8, 9, 19.
    # At p=0.5 benefits are 0, 30, 42, 60; subtract each cost for net value.
    # At p=0.01 benefits are 0, 0.6, 0.84, 1.2; every intervention loses money.
    assert option["proxy_value"] == pytest.approx(600.0)
    assert option["benefit"] == pytest.approx(expected_benefit)
    assert option["net_value"] == pytest.approx(expected_net_value)


@pytest.mark.parametrize(
    (
        "code",
        "name",
        "contact_cost",
        "offer_cost",
        "acceptance_rate",
        "efficacy",
        "expected_cost",
        "reduction",
    ),
    [
        ("A0", "Không can thiệp", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            "A1",
            "Hỗ trợ kỹ thuật & bảo mật chủ động",
            2.0,
            15.0,
            0.40,
            0.25,
            8.0,
            0.10,
        ),
        (
            "A2",
            "Tư vấn hợp đồng & thanh toán tự động",
            2.0,
            20.0,
            0.35,
            0.40,
            9.0,
            0.14,
        ),
        (
            "A3",
            "Ưu đãi giảm cước cam kết dài hạn",
            3.0,
            40.0,
            0.40,
            0.50,
            19.0,
            0.20,
        ),
    ],
)
def test_default_actions_have_expected_configuration(
    code: str,
    name: str,
    contact_cost: float,
    offer_cost: float,
    acceptance_rate: float,
    efficacy: float,
    expected_cost: float,
    reduction: float,
) -> None:
    actions = default_actions()

    assert [action.code for action in actions] == ["A0", "A1", "A2", "A3"]
    action = next(action for action in actions if action.code == code)
    assert action.name == name
    assert action.contact_cost == pytest.approx(contact_cost)
    assert action.offer_cost == pytest.approx(offer_cost)
    assert action.acceptance_rate == pytest.approx(acceptance_rate)
    assert action.efficacy == pytest.approx(efficacy)
    assert action.expected_cost == pytest.approx(expected_cost)
    assert action.effective_reduction == pytest.approx(reduction)
    assert action.is_no_action is (code == "A0")
