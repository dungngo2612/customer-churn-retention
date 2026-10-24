"""Integration tests for End-to-End Real Data Flow (Task 3: Ghép luồng ứng dụng bằng dữ liệu thật).

Kiểm thử toàn bộ chuỗi mắt xích liên hoàn:
1. Dataset Upload & Validation: Kiểm tra dữ liệu theo hợp đồng schema, phát hiện lỗi cột/dòng.
2. Scoring Job với Model Bundle thật: BundlePredictor chạy trên dữ liệu thật, không dùng mock data.
3. Explainability với TreeSHAP: Trích xuất Global Feature Importances và Local TreeSHAP cho khách hàng.
4. Phân bổ ngân sách tối ưu: select_actions_under_budget (Task 1).
5. Quản lý & Xuất chiến dịch: Tạo Campaign, lưu bất biến và xuất CSV danh sách can thiệp (Task 2).
6. Quy tắc nghiêm ngặt: Bỏ dữ liệu mẫu cứng, bắt lỗi rõ nếu thiếu bundle mô hình.
"""

import sys
from pathlib import Path
import tempfile
import unittest
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_retention.adapters.predictor.bundle_predictor import BundlePredictor
from churn_retention.adapters.database.campaigns.repository import SqliteCampaignRepository
from churn_retention.modules.datasets.validate import validate_dataset
from churn_retention.modules.retention.budget import select_actions_under_budget
from churn_retention.modules.campaigns import (
    CampaignStatus,
    create_campaign,
    export_campaign_csv,
    get_campaign,
    list_campaigns,
)
from churn_retention.ml.explainability.shap_explainer import (
    get_global_feature_importances,
    explain_single_customer_shap,
)


class TestE2EIntegration(unittest.TestCase):
    """Bộ kiểm thử liên thông toàn bộ chuỗi giá trị từ dữ liệu thật đến chiến dịch."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "e2e_test_campaigns.db"
        self.repo = SqliteCampaignRepository(db_path=self.db_path)

        # Đường dẫn file dữ liệu thật và bundle thật
        self.data_path = ROOT_DIR / "data" / "telco_churn.csv"
        self.bundle_path = ROOT_DIR / "artifacts" / "bundles" / "catboost-20261018T130222-seed42"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_e2e_dataset_validation(self):
        """1. Kiểm tra tính hợp lệ và tự động làm sạch của tập dữ liệu bằng module của Hiếu."""
        self.assertTrue(self.data_path.exists(), "File telco_churn.csv phải tồn tại trong data/")
        df = pd.read_csv(self.data_path)
        
        # Kiểm tra file raw chưa xử lý: phát hiện chính xác 11 dòng TotalCharges để trống
        raw_report = validate_dataset(df)
        self.assertFalse(raw_report.is_valid)
        self.assertEqual(len(raw_report.errors), 11)
        self.assertTrue(all("TotalCharges" in e.column for e in raw_report.errors))

        # Kiểm tra qua module làm sạch clean_and_split_features: dữ liệu sau làm sạch phải hợp lệ 100%
        from churn_retention.modules.datasets.use_cases import clean_and_split_features
        clean_df, _ = clean_and_split_features(df)
        clean_df["Churn"] = df["Churn"]
        clean_report = validate_dataset(clean_df)
        self.assertTrue(clean_report.is_valid)
        self.assertEqual(len(clean_report.errors), 0)

    def test_e2e_scoring_with_real_catboost_bundle(self):
        """2. Kiểm tra chạy Scoring Job thật bằng CatBoost bundle của Dũng trên 100 khách hàng."""
        self.assertTrue(self.bundle_path.exists(), "Model bundle CatBoost phải tồn tại trong artifacts/bundles/")
        df = pd.read_csv(self.data_path).head(100)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

        predictor = BundlePredictor()
        predictor.load(str(self.bundle_path))
        self.assertEqual(predictor.model_version, "catboost-20261018T130222-seed42")

        probs = predictor.predict_batch(features=df, customer_ids=df["customerID"].tolist())
        self.assertEqual(len(probs), 100)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_e2e_shap_explainability(self):
        """3. Kiểm tra trích xuất SHAP Global và Local TreeSHAP từ mô hình CatBoost thật."""
        predictor = BundlePredictor()
        predictor.load(str(self.bundle_path))

        # Global feature importance
        imp_df = get_global_feature_importances(predictor)
        self.assertFalse(imp_df.empty)
        self.assertIn("Contract", imp_df["feature"].values)
        self.assertIn("tenure", imp_df["feature"].values)

        # Local TreeSHAP cho 1 khách hàng
        df_sample = pd.read_csv(self.data_path).head(1)
        if "TotalCharges" in df_sample.columns:
            df_sample["TotalCharges"] = pd.to_numeric(df_sample["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

        local_shap = explain_single_customer_shap(predictor, df_sample)
        self.assertIsNotNone(local_shap)
        self.assertFalse(local_shap.empty)
        self.assertIn("feature", local_shap.columns)
        self.assertIn("shap_value", local_shap.columns)
        self.assertIn("impact", local_shap.columns)

    def test_e2e_budget_knapsack_and_campaign_workflow(self):
        """4. Kiểm tra trọn vẹn luồng: Scoring -> Phân bổ ngân sách -> Lưu chiến dịch -> Xuất CSV."""
        # 1. Đọc dữ liệu khách hàng thật (50 khách đang hoạt động)
        df_raw = pd.read_csv(self.data_path)
        if "TotalCharges" in df_raw.columns:
            df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
        active_df = df_raw[(df_raw["Churn"] == "No") & (df_raw["tenure"] > 0)].head(50).copy()

        # 2. Chấm điểm với CatBoost bundle thật
        predictor = BundlePredictor()
        predictor.load(str(self.bundle_path))
        probs = predictor.predict_batch(features=active_df, customer_ids=active_df["customerID"].tolist())
        pred_df = pd.DataFrame({"customer_id": active_df["customerID"].tolist(), "churn_prob": probs})

        # 3. Phân bổ ngân sách tối ưu $500
        budget = 500.0
        alloc_df = select_actions_under_budget(
            customers=active_df,
            predictions=pred_df,
            budget=budget,
            margin=0.50,
            horizon_months=12,
        )

        total_cost = alloc_df["cost"].sum()
        self.assertLessEqual(total_cost, budget)
        interv_df = alloc_df[alloc_df["action_code"] != "A0"]
        self.assertGreater(len(interv_df), 0)

        # 4. Lưu chiến dịch vào Database
        camp = create_campaign(
            name="E2E Pipeline Test Campaign",
            dataset_id="telco_churn.csv",
            model_version=predictor.model_version,
            scoring_run_id="run-e2e-001",
            budget=budget,
            assumptions={"margin": 0.50, "horizon_months": 12},
            results={
                "summary": {
                    "total_cost": float(total_cost),
                    "total_net_value": float(alloc_df["net_value"].sum()),
                    "n_intervened": len(interv_df),
                },
                "allocations": alloc_df.to_dict("records"),
            },
            status=CampaignStatus.SIMULATED,
            repository=self.repo,
        )

        self.assertEqual(camp.status, CampaignStatus.SIMULATED)

        # 5. Xuất file CSV danh sách can thiệp
        csv_str = export_campaign_csv(camp.campaign_id, only_interventions=True, repository=self.repo)
        self.assertIn("customer_id,action_code,action_name,cost,benefit,net_value", csv_str)
        self.assertNotIn("A0", csv_str)

        # Trạng thái trong DB phải tự động chuyển thành EXPORTED
        updated_camp = get_campaign(camp.campaign_id, repository=self.repo)
        self.assertEqual(updated_camp.status, CampaignStatus.EXPORTED)


if __name__ == "__main__":
    unittest.main()
