"""Unit & Integration tests for Task Queue, Idempotency, and Worker Retry Policy (Task 4).

Kiểm thử toàn diện các yêu cầu:
1. Job có khóa chống trùng (Idempotency Key): Gửi lại cùng một job không tạo trùng lặp.
2. Trạng thái và lý do lỗi xem được rõ ràng (PENDING, RUNNING, RETRYING, SUCCESS, FAILED).
3. Lỗi schema (Permanent Error): Thất bại ngay lập tức (FAILED), KHÔNG retry.
4. Lỗi tạm thời (Transient Error): Retry có giới hạn (tối đa 3 lần).
5. Ràng buộc duy nhất trong DB (UNIQUE constraint): Chạy lại cùng một job hoặc chạy lại sau khi ngắt giữa chừng KHÔNG làm tăng gấp đôi predictions.
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

from churn_retention.contracts.jobs.models import JobStatus, JobType, TaskJob
from churn_retention.adapters.task_queue.queue import SqliteTaskQueue
from churn_retention.entrypoints.worker.worker import (
    IdempotentPredictionStore,
    JobWorker,
    PermanentError,
    TransientError,
)


class TestTaskQueueAndWorker(unittest.TestCase):
    """Test suite cho hạ tầng Task Queue, Chống trùng lặp và Retry Worker."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.queue_db = Path(self.temp_dir.name) / "test_queue.db"
        self.pred_db = Path(self.temp_dir.name) / "test_predictions.db"

        self.queue = SqliteTaskQueue(db_path=self.queue_db)
        self.pred_store = IdempotentPredictionStore(db_path=self.pred_db)
        self.worker = JobWorker(queue=self.queue, prediction_store=self.pred_store)

        self.data_path = ROOT_DIR / "data" / "telco_churn.csv"
        self.bundle_path = ROOT_DIR / "artifacts" / "bundles" / "catboost-20261018T130222-seed42"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_idempotency_key_prevents_duplicate_enqueue(self):
        """1. Kiểm tra gửi lại cùng một job: Không làm tăng số lượng job trong queue."""
        idempotency_key = "scoring:telco_churn.csv:catboost-seed42"
        job1 = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key=idempotency_key,
            payload={"dataset_id": "telco_churn.csv"},
        )
        enqueued1 = self.queue.enqueue(job1)
        self.assertEqual(enqueued1.job_id, job1.job_id)

        # Gửi lại lần 2 với cùng idempotency_key nhưng job_id mới
        job2 = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key=idempotency_key,
            payload={"dataset_id": "telco_churn.csv"},
        )
        enqueued2 = self.queue.enqueue(job2)

        # Hàng đợi phải trả về job1 cũ, không tạo mới
        self.assertEqual(enqueued2.job_id, job1.job_id)
        all_jobs = self.queue.list_jobs()
        self.assertEqual(len(all_jobs), 1)

    def test_schema_error_fails_immediately_no_retry(self):
        """2. Kiểm tra lỗi schema: Chuyển FAILED ngay, không được phép retry."""
        # Tạo file CSV giả bị sai schema (thiếu customerID)
        bad_csv_path = Path(self.temp_dir.name) / "bad_schema.csv"
        pd.DataFrame({"invalid_col": [1, 2, 3]}).to_csv(bad_csv_path, index=False)

        job = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key="scoring:bad_schema_01",
            payload={"dataset_path": str(bad_csv_path), "bundle_path": str(self.bundle_path)},
            max_retries=3,
        )
        self.queue.enqueue(job)

        # Worker thực thi job
        processed_job = self.worker.process_one()
        self.assertIsNotNone(processed_job)
        self.assertEqual(processed_job.status, JobStatus.FAILED)
        self.assertEqual(processed_job.error_type, "PermanentError")
        self.assertIn("Lỗi Schema", processed_job.error_reason)
        # Không được retry lần nào
        self.assertEqual(processed_job.retry_count, 0)

    def test_transient_error_retries_and_exhausts(self):
        """3. Kiểm tra lỗi tạm thời: Cho phép retry có giới hạn (3 lần) rồi mới FAILED."""
        attempt_counter = [0]

        def flaky_handler(job: TaskJob):
            attempt_counter[0] += 1
            raise TransientError(f"Database lock timeout (lần {attempt_counter[0]})")

        self.worker.register_handler(JobType.SCORING, flaky_handler)

        job = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key="scoring:transient_test",
            payload={},
            max_retries=3,
        )
        self.queue.enqueue(job)

        # Lần chạy 1 -> RETRYING (retry_count = 1)
        j1 = self.worker.process_one()
        self.assertEqual(j1.status, JobStatus.RETRYING)
        self.assertEqual(j1.retry_count, 1)

        # Lần chạy 2 -> RETRYING (retry_count = 2)
        j2 = self.worker.process_one()
        self.assertEqual(j2.status, JobStatus.RETRYING)
        self.assertEqual(j2.retry_count, 2)

        # Lần chạy 3 -> RETRYING (retry_count = 3)
        j3 = self.worker.process_one()
        self.assertEqual(j3.status, JobStatus.RETRYING)
        self.assertEqual(j3.retry_count, 3)

        # Lần chạy 4 -> Quá số lần retry -> FAILED
        j4 = self.worker.process_one()
        self.assertEqual(j4.status, JobStatus.FAILED)
        self.assertEqual(j4.error_type, "MaxRetriesExceeded")
        self.assertIn("Đã thử lại tối đa 3 lần", j4.error_reason)

    def test_unique_db_constraint_prevents_duplicate_predictions(self):
        """4. Kiểm tra ràng buộc duy nhất trong DB: Chạy lại cùng một job không làm tăng gấp đôi predictions."""
        # Tạo file CSV mẫu 10 khách
        sample_csv = Path(self.temp_dir.name) / "sample_10.csv"
        df_raw = pd.read_csv(self.data_path).head(10)
        df_raw.to_csv(sample_csv, index=False)

        scoring_run_id = "run-idempotency-test-001"
        job = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key=f"scoring:{scoring_run_id}",
            payload={
                "dataset_path": str(sample_csv),
                "bundle_path": str(self.bundle_path),
                "scoring_run_id": scoring_run_id,
            },
        )
        self.queue.enqueue(job)

        # Chạy lần 1
        res1 = self.worker.process_one()
        self.assertEqual(res1.status, JobStatus.SUCCESS)
        self.assertEqual(res1.result["inserted_new"], 10)
        self.assertEqual(self.pred_store.count_predictions(scoring_run_id), 10)

        # Giả lập gửi lại cùng job hoặc chạy lại tác vụ
        res2_inserted = self.pred_store.save_predictions_batch(
            scoring_run_id=scoring_run_id,
            predictions_df=pd.DataFrame({
                "customer_id": df_raw["customerID"].tolist(),
                "churn_prob": [0.5] * 10,
            }),
            model_version="catboost-seed42",
        )
        # Số dòng chèn mới phải là 0 vì đã tồn tại khóa (scoring_run_id, customer_id)
        self.assertEqual(res2_inserted, 0)
        # Tổng số bản ghi trong DB vẫn là đúng 10, KHÔNG bị tăng lên 20
        self.assertEqual(self.pred_store.count_predictions(scoring_run_id), 10)

    def test_worker_interrupted_and_resumed_safe(self):
        """5. Kiểm tra ngắt worker giữa chừng rồi chạy lại: Kết quả vẫn đầy đủ, không trùng lặp."""
        sample_csv = Path(self.temp_dir.name) / "sample_interrupt.csv"
        df_raw = pd.read_csv(self.data_path).head(20)
        df_raw.to_csv(sample_csv, index=False)

        scoring_run_id = "run-interrupt-test-002"

        # Giả lập worker đã ghi được 10 dòng đầu tiên thì bị crash/kill
        half_preds = pd.DataFrame({
            "customer_id": df_raw["customerID"].head(10).tolist(),
            "churn_prob": [0.6] * 10,
        })
        self.pred_store.save_predictions_batch(scoring_run_id, half_preds, "catboost-seed42")
        self.assertEqual(self.pred_store.count_predictions(scoring_run_id), 10)

        # Chạy lại worker từ đầu trên toàn bộ 20 dòng
        job = TaskJob.create(
            job_type=JobType.SCORING,
            idempotency_key=f"scoring:{scoring_run_id}",
            payload={
                "dataset_path": str(sample_csv),
                "bundle_path": str(self.bundle_path),
                "scoring_run_id": scoring_run_id,
            },
        )
        self.queue.enqueue(job)
        res = self.worker.process_one()

        self.assertEqual(res.status, JobStatus.SUCCESS)
        # Chỉ có 10 dòng còn lại được chèn mới (10 dòng đầu bị ignore do trùng khóa)
        self.assertEqual(res.result["inserted_new"], 10)
        # Tổng số trong DB là đúng 20 dòng, không thiếu và không trùng!
        self.assertEqual(self.pred_store.count_predictions(scoring_run_id), 20)


if __name__ == "__main__":
    unittest.main()
