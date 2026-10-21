from datetime import datetime, timezone

from sqlalchemy.orm import Session

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.models import DatasetMetadataModel


def test_create_and_get_dataset(db_session: Session) -> None:
    """Kiểm tra lưu bản ghi metadata và truy vấn lại theo ID."""
    repo = DatasetRepository(session=db_session)
    dataset = DatasetMetadataModel(
        dataset_id="test-dataset-001",
        filename="churn_data.csv",
        total_rows=100,
        total_columns=10,
        features_s3_uri="s3://churn-datasets/features.parquet",
        labels_s3_uri="s3://churn-datasets/labels.parquet",
        has_labels=True,
        created_at=datetime.now(timezone.utc),
    )

    created = repo.create(dataset)
    assert created.dataset_id == "test-dataset-001"

    fetched = repo.get_by_id("test-dataset-001")
    assert fetched is not None
    assert fetched.filename == "churn_data.csv"
    assert fetched.total_rows == 100


def test_list_all_datasets(db_session: Session) -> None:
    """Kiểm tra lấy toàn bộ danh sách metadata."""
    repo = DatasetRepository(session=db_session)
    d1 = DatasetMetadataModel(
        dataset_id="id-1",
        filename="file1.csv",
        total_rows=10,
        total_columns=5,
        has_labels=False,
        features_s3_uri="s3://churn-datasets/features_1.parquet",
        labels_s3_uri=None,
        created_at=datetime.now(timezone.utc),
    )
    d2 = DatasetMetadataModel(
        dataset_id="id-2",
        filename="file2.csv",
        total_rows=20,
        total_columns=5,
        has_labels=True,
        features_s3_uri="s3://churn-datasets/features_2.parquet",
        labels_s3_uri="s3://churn-datasets/labels_2.parquet",
        created_at=datetime.now(timezone.utc),
    )
    repo.create(d1)
    repo.create(d2)

    all_items = repo.list_all()
    assert len(all_items) == 2
