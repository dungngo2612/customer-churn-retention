"""Unit tests for KHDV White-Box Service Advisor (3-tier CSKH explanation & 5 extended services)."""

import sys
from pathlib import Path
import unittest
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_retention.modules.retention.service_advisor import (
    build_service_advisory,
    match_extended_services,
    service_channel_for_customer,
    summarize_extended_services,
)


class TestServiceAdvisor(unittest.TestCase):
    def test_matches_all_applicable_extended_services(self):
        customer = {
            "customerID": "CUST-HIGH-01",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 3,
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
        }
        matches = match_extended_services(customer)
        codes = [m.code for m in matches]
        self.assertEqual(codes, ["EXT-1", "EXT-2", "EXT-3", "EXT-4", "EXT-5"])
        summary = summarize_extended_services(customer)
        self.assertIn("Người cao tuổi", summary)
        self.assertIn("Cáp quang", summary)

    def test_translates_shap_drivers_and_builds_3_tier_advisory(self):
        customer = {
            "customerID": "7590-VHVEG",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
        }
        rec = {
            "customerID": "7590-VHVEG",
            "action_code": "A2",
            "action_name": "Tư vấn hợp đồng & thanh toán tự động",
            "churn_prob": 0.72,
            "net_value": 32.5,
            "breakeven_prob": 0.16,
        }
        shap_df = pd.DataFrame(
            [
                {"feature": "Contract", "value": "Month-to-month", "shap_value": 0.45},
                {"feature": "TechSupport", "value": "No", "shap_value": 0.19},
                {"feature": "tenure", "value": "24", "shap_value": -0.22},
            ]
        )

        adv = build_service_advisory(customer, rec, local_shap_df=shap_df)
        self.assertEqual(adv.customer_id, "7590-VHVEG")
        self.assertTrue(any("Month-to-month" in f for f in adv.facts))
        self.assertTrue(any("SHAP +0.45" in d for d in adv.risk_drivers_vi))
        self.assertTrue(any("SHAP -0.22" in d for d in adv.risk_drivers_vi))
        self.assertIn("Tổng đài CSKH Outbound 1-1", adv.service_channel)
        self.assertIn("7590-VHVEG", adv.call_script)
        self.assertIn("A2", adv.recommendation_summary)

    def test_a0_customer_without_shap_uses_standard_channel_and_safe_fallback(self):
        customer = {
            "customerID": "LOW-001",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 60,
            "MultipleLines": "No",
            "InternetService": "No",
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 20.0,
        }
        rec = {
            "customerID": "LOW-001",
            "action_code": "A0",
            "action_name": "Không can thiệp",
            "churn_prob": 0.08,
            "net_value": 0.0,
            "breakeven_prob": None,
        }

        self.assertIn("tiêu chuẩn", service_channel_for_customer("A0", 0.08))
        adv = build_service_advisory(customer, rec, local_shap_df=None)
        self.assertEqual(adv.extended_services, ())
        self.assertIn("chế độ mô phỏng", adv.assessment_summary)
        self.assertIn("A0", adv.call_script)


if __name__ == "__main__":
    unittest.main()
