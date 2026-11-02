"""Unit tests for Customer Churn Retention economic evaluation and recommendation module."""

import sys
from pathlib import Path

# Thêm thư mục src vào sys.path để test chạy độc lập mà không bắt buộc phải pip install -e .
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unittest

import pandas as pd

from churn_retention.modules.retention.policy import (
    RETENTION_CATALOG,
    RetentionPolicy,
    compute_proxy_clv,
)
from churn_retention.modules.retention.use_cases import (
    OPTION_COLUMNS,
    RANK_COLUMNS,
    default_actions,
    evaluate_actions,
    recommend,
    sensitivity_by_margin,
)


class TestRetentionPolicy(unittest.TestCase):
    """Test suite for retention policy and individual customer economic calculations."""

    def test_retention_action_spec_expected_cost_and_reduction(self):
        """Kiểm tra chi phí kỳ vọng C_k và mức giảm rủi ro r_k của hành động."""
        action_a1 = RETENTION_CATALOG["A1"]
        # C = contact_cost (2.0) + acceptance_rate (0.40) * offer_cost (15.0) = 8.0
        self.assertEqual(action_a1.expected_cost, 8.0)
        # r = acceptance_rate (0.40) * efficacy_given_accept (0.25) = 0.10
        self.assertEqual(action_a1.effective_churn_reduction, 0.10)

        action_a0 = RETENTION_CATALOG["A0"]
        self.assertEqual(action_a0.expected_cost, 0.0)
        self.assertEqual(action_a0.effective_churn_reduction, 0.0)

    def test_compute_proxy_clv(self):
        """Kiểm tra công thức tính Proxy CLV: V = MonthlyCharges * horizon * margin."""
        clv = compute_proxy_clv(
            monthly_charges=70.0, horizon_months=12, contribution_margin=0.50
        )
        # 70.0 * 12 * 0.50 = 420.0
        self.assertEqual(clv, 420.0)

    def test_retention_policy_positive_env(self):
        """Kiểm tra trường hợp can thiệp có lãi (ENV > 0)."""
        policy = RetentionPolicy(
            clv=500.0, churn_prob=0.60, cost=10.0, success_rate=0.15
        )
        # B = 0.60 * 0.15 * 500.0 = 45.0
        self.assertAlmostEqual(policy.expected_benefit(), 45.0, places=2)
        # ENV = 45.0 - 10.0 = 35.0
        self.assertAlmostEqual(policy.expected_net_value(), 35.0, places=2)
        # p* = 10.0 / (0.15 * 500.0) = 10.0 / 75.0 = 0.1333
        self.assertAlmostEqual(policy.breakeven_prob(), 0.1333, places=4)
        self.assertTrue(policy.should_intervene())

    def test_retention_policy_negative_env(self):
        """Kiểm tra trường hợp can thiệp lỗ (ENV <= 0) thì không nên can thiệp."""
        policy = RetentionPolicy(
            clv=100.0, churn_prob=0.05, cost=15.0, success_rate=0.10
        )
        # B = 0.05 * 0.10 * 100.0 = 0.5
        # ENV = 0.5 - 15.0 = -14.5
        self.assertLess(policy.expected_net_value(), 0)
        self.assertFalse(policy.should_intervene())


class TestRetentionUseCases(unittest.TestCase):
    """Test suite for batch evaluation, recommendation, and margin sensitivity."""

    def setUp(self):
        """Tạo dữ liệu mẫu gồm 2 khách hàng: 1 khách rủi ro cao, 1 khách rủi ro cực thấp."""
        self.customers = pd.DataFrame(
            [
                {
                    "customerID": "CUST-HIGH",
                    "Churn": "No",
                    "tenure": 12,
                    "MonthlyCharges": 90.0,
                    "TotalCharges": 1080.0,
                    "Contract": "Month-to-month",
                    "PaymentMethod": "Electronic check",
                    "InternetService": "Fiber optic",
                    "TechSupport": "No",
                    "OnlineSecurity": "No",
                },
                {
                    "customerID": "CUST-LOW",
                    "Churn": "No",
                    "tenure": 48,
                    "MonthlyCharges": 25.0,
                    "TotalCharges": 1200.0,
                    "Contract": "Two year",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "InternetService": "DSL",
                    "TechSupport": "Yes",
                    "OnlineSecurity": "Yes",
                },
            ]
        )

        self.predictions = pd.DataFrame(
            [
                {"customer_id": "CUST-HIGH", "churn_prob": 0.85},
                {"customer_id": "CUST-LOW", "churn_prob": 0.02},
            ]
        )

    def test_evaluate_actions_schema_and_counts(self):
        """Kiểm tra cấu trúc đầu ra của evaluate_actions."""
        actions = default_actions()
        eval_df = evaluate_actions(
            customers=self.customers,
            predictions=self.predictions,
            actions=actions,
            margin=0.50,
            horizon_months=12,
        )

        for col in OPTION_COLUMNS:
            self.assertIn(col, eval_df.columns)

        # Cả 2 khách đều phải có ít nhất hành động A0
        self.assertGreater(len(eval_df), 0)
        self.assertIn("CUST-HIGH", eval_df["customerID"].values)
        self.assertIn("CUST-LOW", eval_df["customerID"].values)

    def test_recommend_logic_and_a0_fallback(self):
        """Kiểm tra recommend: chọn gói net_value > 0 lớn nhất, hoặc về A0 nếu lỗ."""
        recs = recommend(
            customers=self.customers,
            predictions=self.predictions,
            margin=0.50,
            horizon_months=12,
        )

        for col in RANK_COLUMNS:
            self.assertIn(col, recs.columns)

        self.assertEqual(len(recs), 2)

        # Khách rủi ro cao phải được đề xuất một gói can thiệp có net_value > 0
        rec_high = recs[recs["customerID"] == "CUST-HIGH"].iloc[0]
        self.assertIn(rec_high["action_code"], ["A1", "A2", "A3"])
        self.assertGreater(rec_high["net_value"], 0)

        # Khách rủi ro cực thấp (2%) chi phí can thiệp sẽ lớn hơn lợi ích nên phải được gán về A0
        rec_low = recs[recs["customerID"] == "CUST-LOW"].iloc[0]
        self.assertEqual(rec_low["action_code"], "A0")
        self.assertEqual(rec_low["net_value"], 0.0)

    def test_sensitivity_by_margin(self):
        """Kiểm tra độ nhạy theo biên lợi nhuận: margin cao hơn thì ΔEV cao hơn."""
        sens = sensitivity_by_margin(
            customers=self.customers,
            predictions=self.predictions,
            margins=(0.20, 0.40, 0.60),
            horizon_months=12,
        )

        self.assertEqual(len(sens), 3)
        self.assertIn("total_net_value", sens.columns)

        # Biên lợi nhuận tăng từ 20% -> 60% thì tổng giá trị ròng phải tăng
        net_20 = sens.loc[sens["margin"] == 0.20, "total_net_value"].values[0]
        net_60 = sens.loc[sens["margin"] == 0.60, "total_net_value"].values[0]
        self.assertGreaterEqual(net_60, net_20)


if __name__ == "__main__":
    unittest.main()
