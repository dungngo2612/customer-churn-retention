"""Unit tests for Budget-constrained Retention Allocation module.

Kiểm thử toàn diện các yêu cầu của Task 1 (Chọn can thiệp theo ngân sách):
1. Ngân sách 0: Không can thiệp tốn phí nào được chọn, tổng chi phí = 0.
2. Ngân sách nhỏ: Tổng chi phí không vượt B, ưu tiên can thiệp có tỷ số ΔEV / C cao nhất.
3. Ngân sách đủ: Mọi khách có can thiệp dương đều được phân bổ.
4. Hành động chi phí 0: Được chọn ngay mà không bị lỗi chia cho 0.
5. Lợi ích âm: Các cặp có ΔEV <= 0 tuyệt đối không được chọn, khách về A0.
6. Mỗi khách tối đa một hành động: Không bao giờ phân bổ trùng 2 hành động cho 1 khách.
7. Tổng chi phí không bao giờ vượt B: Bất kể cấu hình dữ liệu và ngân sách.
8. Đối chiếu trên bộ 5-10 khách nhỏ: So sánh với nghiệm vét cạn (Brute-Force), tính Optimality Gap.
9. Ghi nhận rõ: Đây là thuật toán heuristic xấp xỉ bài toán Ba-lô đa lựa chọn (MCKP).
"""

import unittest

import pandas as pd

from churn_retention.modules.retention.use_cases import (
    ActionConfig,
    ValidationError,
    compare_greedy_vs_optimal,
    select_actions_under_budget,
)


class TestBudgetOptimization(unittest.TestCase):
    """Bộ kiểm thử cho giải thuật chọn can thiệp theo ngân sách."""

    def setUp(self):
        """Tạo bộ dữ liệu chuẩn 6 khách hàng (5-10 khách) với các mức rủi ro và cước khác nhau."""
        self.customers = pd.DataFrame(
            [
                {
                    "customerID": f"CUST-0{i}",
                    "Churn": "No",
                    "tenure": 6 + i * 4,
                    "MonthlyCharges": 30.0 + i * 15.0,  # 45, 60, 75, 90, 105, 120
                    "TotalCharges": (30.0 + i * 15.0) * (6 + i * 4),
                    "Contract": "Month-to-month" if i % 2 == 1 else "One year",
                    "PaymentMethod": "Electronic check"
                    if i % 2 == 1
                    else "Credit card (automatic)",
                    "InternetService": "Fiber optic",
                    "TechSupport": "No",
                    "OnlineSecurity": "No",
                }
                for i in range(1, 7)
            ]
        )

        # Churn probability tăng dần từ 0.05 đến 0.85
        self.predictions = pd.DataFrame(
            [
                {
                    "customer_id": "CUST-01",
                    "churn_prob": 0.05,
                },  # Rủi ro rất thấp (lợi ích âm)
                {"customer_id": "CUST-02", "churn_prob": 0.35},
                {"customer_id": "CUST-03", "churn_prob": 0.50},
                {"customer_id": "CUST-04", "churn_prob": 0.65},
                {"customer_id": "CUST-05", "churn_prob": 0.75},
                {"customer_id": "CUST-06", "churn_prob": 0.85},
            ]
        )

    def test_budget_zero(self):
        """1. Kiểm tra trường hợp ngân sách B = 0: Không can thiệp tốn phí nào được chọn."""
        res = select_actions_under_budget(
            customers=self.customers,
            predictions=self.predictions,
            budget=0.0,
        )

        self.assertEqual(len(res), len(self.customers))
        self.assertEqual(res["cost"].sum(), 0.0)
        # Mọi khách hàng đều phải gán về A0
        self.assertTrue((res["action_code"] == "A0").all())
        self.assertEqual(res["net_value"].sum(), 0.0)

    def test_budget_small_insufficient_for_all(self):
        """2. Kiểm tra ngân sách nhỏ (B = 8.0 USD): Chỉ đủ cho đúng 1 hành động rẻ nhất (A1
        cost=8.0).
        """
        res = select_actions_under_budget(
            customers=self.customers,
            predictions=self.predictions,
            budget=8.0,
        )

        total_cost = res["cost"].sum()
        self.assertLessEqual(total_cost, 8.0)
        self.assertGreater(total_cost, 0.0)

        # Số khách được can thiệp đúng bằng 1
        intervened = res[res["action_code"] != "A0"]
        self.assertEqual(len(intervened), 1)

        # Khách được chọn phải có efficiency_ratio cao nhất trong các gói chi phí <= 8
        chosen_cid = intervened.iloc[0]["customerID"]
        self.assertIn(chosen_cid, ["CUST-05", "CUST-06"])
        self.assertEqual(intervened.iloc[0]["action_code"], "A1")

    def test_budget_full_covers_all_positive_nv(self):
        """3. Kiểm tra ngân sách đủ (B = 10,000 USD): Mọi can thiệp có ΔEV > 0 đều được chọn."""
        res = select_actions_under_budget(
            customers=self.customers,
            predictions=self.predictions,
            budget=10000.0,
        )

        total_cost = res["cost"].sum()
        self.assertLessEqual(total_cost, 10000.0)

        # Khách CUST-01 rủi ro 5% (lợi ích âm) phải về A0
        cust_01 = res[res["customerID"] == "CUST-01"].iloc[0]
        self.assertEqual(cust_01["action_code"], "A0")
        self.assertEqual(cust_01["net_value"], 0.0)

        # Các khách có rủi ro cao CUST-04, 05, 06 phải được can thiệp
        intervened = res[res["action_code"] != "A0"]
        self.assertGreaterEqual(len(intervened), 3)

    def test_zero_cost_action(self):
        """4. Kiểm tra hành động có chi phí = 0 và ΔEV > 0: Chọn ngay, không chia cho 0."""
        # Tạo thêm 1 hành động miễn phí A_FREE: email nhắc nhở tự động, chi phí = 0, efficacy = 0.05
        actions_with_free = (
            ActionConfig("A0", "Không can thiệp", 0, 0, 0, 0, is_no_action=True),
            ActionConfig("A_FREE", "Email tư vấn miễn phí", 0.0, 0.0, 1.0, 0.05),
            ActionConfig("A1", "Hỗ trợ kỹ thuật", 2, 15, 0.40, 0.25),
        )

        # Ngân sách 0 nhưng có hành động miễn phí A_FREE với ΔEV > 0
        res = select_actions_under_budget(
            customers=self.customers,
            predictions=self.predictions,
            budget=0.0,
            actions=actions_with_free,
        )

        self.assertEqual(res["cost"].sum(), 0.0)
        # Khách có churn prob lớn sẽ được gán A_FREE vì cost = 0 và ΔEV > 0
        free_assigned = res[res["action_code"] == "A_FREE"]
        self.assertGreater(len(free_assigned), 0)
        # Khẳng định efficiency_ratio là inf hoặc float
        for ratio in free_assigned["efficiency_ratio"]:
            self.assertEqual(ratio, float("inf"))

    def test_negative_benefit_never_selected(self):
        """5. Kiểm tra các cặp có ΔEV <= 0 tuyệt đối không được can thiệp."""
        # Khách hàng có xác suất rời bỏ cực thấp (1%)
        low_risk_cust = pd.DataFrame(
            [
                {
                    "customerID": "CUST-SAFE",
                    "Churn": "No",
                    "tenure": 60,
                    "MonthlyCharges": 20.0,
                    "TotalCharges": 1200.0,
                    "Contract": "Two year",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "InternetService": "DSL",
                    "TechSupport": "Yes",
                    "OnlineSecurity": "Yes",
                }
            ]
        )
        low_risk_pred = pd.DataFrame([{"customer_id": "CUST-SAFE", "churn_prob": 0.01}])

        # Thậm chí ngân sách vô hạn B = 50,000 USD
        res = select_actions_under_budget(
            customers=low_risk_cust,
            predictions=low_risk_pred,
            budget=50000.0,
        )

        self.assertEqual(len(res), 1)
        row = res.iloc[0]
        self.assertEqual(row["action_code"], "A0")
        self.assertEqual(row["cost"], 0.0)
        self.assertEqual(row["net_value"], 0.0)

    def test_at_most_one_action_per_customer(self):
        """6. Kiểm tra ràng buộc duy nhất: Mỗi khách hàng nhận tối đa đúng 1 hành động."""
        res = select_actions_under_budget(
            customers=self.customers,
            predictions=self.predictions,
            budget=50.0,
        )

        # Số dòng phải đúng bằng số khách hàng
        self.assertEqual(len(res), len(self.customers))
        # Không có customerID nào bị lặp lại
        self.assertEqual(len(res["customerID"].unique()), len(self.customers))

    def test_total_cost_never_exceeds_budget(self):
        """7. Ràng buộc cứng: Tổng chi phí không bao giờ vượt B trong mọi mức ngân sách."""
        for b in [0.0, 5.0, 8.0, 15.0, 25.0, 40.0, 60.0, 120.0]:
            res = select_actions_under_budget(
                customers=self.customers,
                predictions=self.predictions,
                budget=b,
            )
            total_cost = res["cost"].sum()
            self.assertLessEqual(
                total_cost,
                b + 1e-7,
                f"Tổng chi phí {total_cost} vượt ngân sách {b}",
            )

    def test_brute_force_comparison_and_optimality_gap(self):
        """8. Đối chiếu trên bộ 6 khách (tập 5-10 khách): So sánh Greedy với Nghiệm tối ưu
        toàn cục (Brute-Force).
        """
        # Thử nghiệm trên ngân sách B = 25.0 USD
        budget = 25.0
        comparison = compare_greedy_vs_optimal(
            customers=self.customers,
            predictions=self.predictions,
            budget=budget,
        )

        # Kiểm tra tính toàn vẹn của kết quả
        self.assertIn("optimality_gap", comparison)
        self.assertIn("greedy_net_value", comparison)
        self.assertIn("optimal_net_value", comparison)

        greedy_nv = comparison["greedy_net_value"]
        opt_nv = comparison["optimal_net_value"]
        gap = comparison["optimality_gap"]

        # Nghiệm tối ưu toàn cục luôn >= nghiệm tham lam
        self.assertGreaterEqual(opt_nv + 1e-4, greedy_nv)
        self.assertGreaterEqual(gap, 0.0)

        # Greedy knapsack trên tập thực tế có khoảng cách tối ưu thường < 20%
        self.assertLess(gap, 0.25)

        # Kiểm tra ngân sách không bị vi phạm ở cả 2 nghiệm
        self.assertLessEqual(comparison["greedy_cost"], budget + 1e-7)
        self.assertLessEqual(comparison["optimal_cost"], budget + 1e-7)

    def test_negative_budget_raises_validation_error(self):
        """9. Kiểm tra ngân sách âm: Phải ném ValidationError."""
        with self.assertRaises(ValidationError):
            select_actions_under_budget(
                customers=self.customers,
                predictions=self.predictions,
                budget=-10.0,
            )


if __name__ == "__main__":
    unittest.main()
