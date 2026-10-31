from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.models import DatasetMetadataModel


def test_postgres_migrated_schema_accepts_dataset_insert_and_read() -> None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        pytest.skip("PostgreSQL integration test requires DATABASE_URL.")

    engine = create_engine(database_url)
    dataset_id = str(uuid4())
    try:
        with Session(engine) as session:
            repository = DatasetRepository(session)
            created = repository.create(
                DatasetMetadataModel(
                    dataset_id=dataset_id,
                    filename="integration.csv",
                    total_rows=1,
                    total_columns=2,
                    has_labels=False,
                    features_path="artifacts/datasets/example/features.parquet",
                    labels_path=None,
                    schema_version="1",
                    checksum_sha256="a" * 64,
                )
            )
            assert created.dataset_id == dataset_id
            assert repository.get_by_id(dataset_id) is not None
            session.execute(
                delete(DatasetMetadataModel).where(DatasetMetadataModel.dataset_id == dataset_id)
            )
            session.commit()
    finally:
        engine.dispose()
