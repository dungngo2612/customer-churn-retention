from __future__ import annotations

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from churn_retention.adapters.database.dataset_repository import DatasetRepository
from churn_retention.adapters.database.session import get_db
from churn_retention.entrypoints.api.routes.datasets import router as datasets_router
from churn_retention.modules.datasets import use_cases


def _valid_csv() -> bytes:
    frame = pd.DataFrame(
        {
            "customerID": ["C-001", "C-002"],
            "tenure": [0, 12],
            "MonthlyCharges": [25.0, 50.0],
            "TotalCharges": ["", "600.0"],
            "Churn": ["No", "Yes"],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_upload_persists_immutable_dataset_and_quality_metadata(
    db_session, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(use_cases, "STORAGE_DIR", tmp_path / "datasets")
    repository = DatasetRepository(db_session)
    payload = _valid_csv()

    first = use_cases.process_dataset_upload(payload, "sample.csv", repository)
    second = use_cases.process_dataset_upload(payload, "sample.csv", repository)

    assert first["status"] == second["status"] == "valid"
    assert first["dataset_id"] != second["dataset_id"]

    metadata = repository.get_by_id(first["dataset_id"])
    assert metadata is not None
    assert metadata.total_rows == 2
    assert metadata.total_columns == 5
    assert metadata.has_labels is True
    assert metadata.schema_version == "1"
    assert metadata.checksum_sha256 is not None
    assert len(metadata.checksum_sha256) == 64
    assert metadata.features_path is not None
    assert metadata.labels_path is not None
    assert pd.read_parquet(metadata.features_path).shape[0] == 2
    assert pd.read_parquet(metadata.labels_path).shape[0] == 2

    listed = use_cases.list_datasets(repository)
    assert {item["dataset_id"] for item in listed} == {
        first["dataset_id"],
        second["dataset_id"],
    }
    detail = use_cases.get_dataset_detail(first["dataset_id"], repository)
    assert detail is not None
    assert detail["checksum_sha256"] == metadata.checksum_sha256

    quality = use_cases.check_dataset_quality(first["dataset_id"], repository)
    assert quality is not None
    assert quality["is_valid"] is True
    assert quality["errors"] == []


def test_quality_check_detects_tampered_stored_object(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(use_cases, "STORAGE_DIR", tmp_path / "datasets")
    repository = DatasetRepository(db_session)
    result = use_cases.process_dataset_upload(_valid_csv(), "sample.csv", repository)
    metadata = repository.get_by_id(result["dataset_id"])
    assert metadata is not None and metadata.features_path is not None

    with open(metadata.features_path, "ab") as stored:
        stored.write(b"tamper")

    quality = use_cases.check_dataset_quality(result["dataset_id"], repository)
    assert quality is not None
    assert quality["is_valid"] is False
    assert any(error["column"] == "checksum_sha256" for error in quality["errors"])


def test_upload_validation_failure_does_not_create_metadata_or_snapshot(
    db_session, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(use_cases, "STORAGE_DIR", tmp_path / "datasets")
    repository = DatasetRepository(db_session)
    invalid = (
        pd.DataFrame(
            {
                "customerID": ["C-001", "C-001"],
                "tenure": [1, 2],
                "MonthlyCharges": [20.0, 30.0],
                "TotalCharges": [20.0, 60.0],
                "Churn": ["No", "Yes"],
            }
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    result = use_cases.process_dataset_upload(invalid, "duplicate.csv", repository)

    assert result["status"] == "invalid"
    assert repository.list_all() == []
    assert not (tmp_path / "datasets").exists()


def test_dataset_api_valid_and_invalid_upload_flow(db_session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(use_cases, "STORAGE_DIR", tmp_path / "datasets")

    def override_db():
        yield db_session

    app = FastAPI()
    app.include_router(datasets_router)
    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as client:
        upload = client.post(
            "/datasets/upload",
            files={"file": ("valid.csv", _valid_csv(), "text/csv")},
        )
        assert upload.status_code == 201
        dataset_id = upload.json()["dataset_id"]

        listing = client.get("/datasets")
        assert listing.status_code == 200
        assert [item["dataset_id"] for item in listing.json()] == [dataset_id]

        detail = client.get(f"/datasets/{dataset_id}")
        assert detail.status_code == 200
        assert detail.json()["total_rows"] == 2

        quality = client.get(f"/datasets/{dataset_id}/quality")
        assert quality.status_code == 200
        assert quality.json()["is_valid"] is True

        duplicate_ids = (
            pd.DataFrame(
                {
                    "customerID": ["C-001", "C-001"],
                    "tenure": [1, 2],
                    "MonthlyCharges": [20.0, 30.0],
                    "TotalCharges": [20.0, 60.0],
                    "Churn": ["No", "Yes"],
                }
            )
            .to_csv(index=False)
            .encode("utf-8")
        )
        invalid = client.post(
            "/datasets/upload",
            files={"file": ("invalid.csv", duplicate_ids, "text/csv")},
        )
        assert invalid.status_code == 422
        assert client.get("/datasets").json()[0]["dataset_id"] == dataset_id

        wrong_extension = client.post(
            "/datasets/upload",
            files={"file": ("invalid.txt", _valid_csv(), "text/plain")},
        )
        assert wrong_extension.status_code == 400

    app.dependency_overrides.clear()
