"""Unit tests for Campaign Management module (Task 2: Quản lý chiến dịch mô phỏng).

Kiểm thử toàn diện các yêu cầu:
1. Tạo và truy xuất chiến dịch với đầy đủ các trường:
   campaign_id, name, dataset_id, model_version, scoring_run_id, budget, assumptions, status, results, created_at.
2. Lưu trữ bền vững (Persistence): Đóng và mở lại kết nối SQLite dữ liệu vẫn nguyên vẹn.
3. Tính bất biến (Immutability): Đổi mô hình/scoring run mới tạo chiến dịch mới, chiến dịch cũ giữ nguyên.
4. Xuất danh sách CSV: customer_id, hành động, chi phí, lợi ích kỳ vọng; chuyển trạng thái sang EXPORTED.
5. Xử lý lỗi và xác thực đầu vào (ngân sách âm, ID không tồn tại).
"""

import sys
from pathlib import Path
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_retention.modules.campaigns.models import Campaign, CampaignStatus
from churn_retention.modules.campaigns.use_cases import (
    CampaignNotFoundError,
    create_campaign,
    get_campaign,
    list_campaigns,
    export_campaign_dataframe,
    export_campaign_csv,
    EXPORT_COLUMNS,
)
from churn_retention.adapters.database.campaigns.repository import (
    SqliteCampaignRepository,
)


class TestCampaignManagement(unittest.TestCase):
    """Test suite cho tính năng quản lý chiến dịch giữ chân khách hàng."""

    def setUp(self):
        """Khởi tạo SQLite database tạm thời cho mỗi ca kiểm thử."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_campaigns.db"
        self.repo = SqliteCampaignRepository(db_path=self.db_path)

        # Dữ liệu mô phỏng chuẩn
        self.sample_assumptions = {
            "margin": 0.50,
            "horizon_months": 12,
            "actions": ["A0", "A1", "A2", "A3"],
        }
        self.sample_results = {
            "summary": {
                "total_cost": 27.0,
                "total_benefit": 120.5,
                "total_net_value": 93.5,
                "n_intervened": 2,
                "n_total": 4,
            },
            "allocations": [
                {
                    "customer_id": "CUST-01",
                    "action_code": "A3",
                    "action_name": "Ưu đãi giảm cước cam kết dài hạn",
                    "cost": 19.0,
                    "benefit": 85.0,
                    "net_value": 66.0,
                },
                {
                    "customer_id": "CUST-02",
                    "action_code": "A1",
                    "action_name": "Hỗ trợ kỹ thuật & bảo mật chủ động",
                    "cost": 8.0,
                    "benefit": 35.5,
                    "net_value": 27.5,
                },
                {
                    "customer_id": "CUST-03",
                    "action_code": "A0",
                    "action_name": "Không can thiệp",
                    "cost": 0.0,
                    "benefit": 0.0,
                    "net_value": 0.0,
                },
            ],
        }

    def tearDown(self):
        """Dọn dẹp thư mục tạm."""
        self.temp_dir.cleanup()

    def test_create_and_get_campaign(self):
        """1. Kiểm tra tạo chiến dịch và truy xuất đầy đủ thông tin."""
        camp = create_campaign(
            name="Q4 Enterprise Retention Push",
            dataset_id="telco-v2026-10",
            model_version="catboost-20261018T130222-seed42",
            scoring_run_id="run-batch-001",
            budget=5000.0,
            assumptions=self.sample_assumptions,
            results=self.sample_results,
            status=CampaignStatus.SIMULATED,
            repository=self.repo,
        )

        self.assertIsNotNone(camp.campaign_id)
        self.assertEqual(camp.name, "Q4 Enterprise Retention Push")
        self.assertEqual(camp.dataset_id, "telco-v2026-10")
        self.assertEqual(camp.model_version, "catboost-20261018T130222-seed42")
        self.assertEqual(camp.scoring_run_id, "run-batch-001")
        self.assertEqual(camp.budget, 5000.0)
        self.assertEqual(camp.status, CampaignStatus.SIMULATED)
        self.assertEqual(camp.assumptions["margin"], 0.50)
        self.assertEqual(camp.results["summary"]["total_cost"], 27.0)

        # Lấy lại từ repository
        retrieved = get_campaign(camp.campaign_id, repository=self.repo)
        self.assertEqual(retrieved.campaign_id, camp.campaign_id)
        self.assertEqual(retrieved.name, camp.name)
        self.assertEqual(retrieved.budget, camp.budget)

    def test_persistence_across_app_restart(self):
        """2. Kiểm tra lưu trữ bền vững: đóng và mở lại ứng dụng vẫn đọc được dữ liệu."""
        camp = create_campaign(
            name="Persistent Campaign Test",
            dataset_id="dataset-persist-01",
            model_version="model-v1",
            scoring_run_id="run-001",
            budget=1000.0,
            assumptions=self.sample_assumptions,
            results=self.sample_results,
            repository=self.repo,
        )

        # Giả lập đóng ứng dụng: giải phóng kết nối và khởi tạo repository mới trên cùng file DB
        del self.repo
        new_repo = SqliteCampaignRepository(db_path=self.db_path)

        reopened_camp = get_campaign(camp.campaign_id, repository=new_repo)
        self.assertIsNotNone(reopened_camp)
        self.assertEqual(reopened_camp.name, "Persistent Campaign Test")
        self.assertEqual(reopened_camp.model_version, "model-v1")
        self.assertEqual(len(reopened_camp.results["allocations"]), 3)

    def test_immutability_on_new_model_version(self):
        """3. Kiểm tra tính bất biến: Đổi model mới thì tạo chiến dịch mới, không đổi chiến dịch cũ."""
        camp_v1 = create_campaign(
            name="Campaign with Model V1",
            dataset_id="dataset-001",
            model_version="catboost-seed42-v1",
            scoring_run_id="run-seed42-01",
            budget=2000.0,
            assumptions=self.sample_assumptions,
            results={"allocations": [{"customer_id": "CUST-A", "cost": 8.0, "net_value": 20.0}]},
            repository=self.repo,
        )

        # Người dùng chuyển sang mô hình mới (ví dụ XGBoost hoặc CatBoost seed 99) và chạy mô phỏng mới
        camp_v2 = create_campaign(
            name="Campaign with Model V2",
            dataset_id="dataset-001",
            model_version="xgboost-seed99-v2",
            scoring_run_id="run-seed99-02",
            budget=3000.0,
            assumptions=self.sample_assumptions,
            results={"allocations": [{"customer_id": "CUST-B", "cost": 19.0, "net_value": 50.0}]},
            repository=self.repo,
        )

        # Kiểm tra hai ID hoàn toàn độc lập
        self.assertNotEqual(camp_v1.campaign_id, camp_v2.campaign_id)

        # Kiểm tra chiến dịch cũ vẫn giữ nguyên 100% giá trị ban đầu
        original_camp_v1 = get_campaign(camp_v1.campaign_id, repository=self.repo)
        self.assertEqual(original_camp_v1.model_version, "catboost-seed42-v1")
        self.assertEqual(original_camp_v1.scoring_run_id, "run-seed42-01")
        self.assertEqual(original_camp_v1.budget, 2000.0)
        self.assertEqual(original_camp_v1.results["allocations"][0]["customer_id"], "CUST-A")

        # Danh sách phải có cả 2 chiến dịch
        all_camps = list_campaigns(repository=self.repo)
        self.assertEqual(len(all_camps), 2)

    def test_export_csv_and_status_update(self):
        """4. Kiểm tra xuất CSV danh sách khách hàng được chọn can thiệp và cập nhật EXPORTED."""
        camp = create_campaign(
            name="Exportable Campaign",
            dataset_id="dataset-001",
            model_version="model-v1",
            scoring_run_id="run-01",
            budget=5000.0,
            assumptions=self.sample_assumptions,
            results=self.sample_results,
            status=CampaignStatus.SIMULATED,
            repository=self.repo,
        )

        self.assertEqual(camp.status, CampaignStatus.SIMULATED)

        # Xuất CSV (mặc định chỉ xuất các khách được can thiệp: loại trừ A0)
        csv_str = export_campaign_csv(camp.campaign_id, only_interventions=True, repository=self.repo)

        # Kiểm tra cấu trúc CSV
        lines = csv_str.strip().split("\n")
        header = lines[0].strip().split(",")
        for col in EXPORT_COLUMNS:
            self.assertIn(col, header)

        # Có 2 khách can thiệp (CUST-01 và CUST-02), CUST-03 là A0 bị loại trừ
        self.assertEqual(len(lines), 3)  # header + 2 dòng dữ liệu
        self.assertIn("CUST-01", csv_str)
        self.assertIn("CUST-02", csv_str)
        self.assertNotIn("CUST-03", csv_str)

        # Sau khi xuất CSV, trạng thái chiến dịch trong DB phải được chuyển sang EXPORTED
        updated = get_campaign(camp.campaign_id, repository=self.repo)
        self.assertEqual(updated.status, CampaignStatus.EXPORTED)

    def test_negative_budget_raises_value_error(self):
        """5. Kiểm tra ngân sách âm: ném ngoại lệ ValueError."""
        with self.assertRaises(ValueError):
            create_campaign(
                name="Negative Budget Campaign",
                dataset_id="dataset-001",
                model_version="model-v1",
                scoring_run_id="run-01",
                budget=-500.0,
                assumptions={},
                repository=self.repo,
            )

    def test_campaign_not_found_raises_error(self):
        """6. Kiểm tra ID không tồn tại: ném CampaignNotFoundError."""
        with self.assertRaises(CampaignNotFoundError):
            get_campaign("non-existent-uuid-12345", repository=self.repo)


if __name__ == "__main__":
    unittest.main()
