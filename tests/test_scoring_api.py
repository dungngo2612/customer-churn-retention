from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import count

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from churn_retention.adapters.predictor.fake import FakePredictor
from churn_retention.contracts.http.scoring import PredictionBatch
from churn_retention.entrypoints.api.routes.scoring import create_scoring_router
from churn_retention.modules.scoring.repository import InMemoryScoringRunRepository


@pytest.fixture
def scoring_client() -> Iterator[TestClient]:
    """Inject demo data and FakePredictor into an app used only by tests."""
    dataset = pd.DataFrame(
        {
            "customer_id": ["demo-C03", "demo-C01", "demo-C02"],
            "demo_usage": [30.0, 10.0, 20.0],
        },
        index=[21, 4, 13],
    )

    def loader(dataset_id: str) -> pd.DataFrame:
        if dataset_id != "demo-dataset":
            raise FileNotFoundError(dataset_id)
        return dataset

    predictor = FakePredictor(seed=17)
    predictor.load("demo-artifact-unused")
    run_ids = count(1)
    app = FastAPI()
    app.include_router(
        create_scoring_router(
            loader=loader,
            predictor=predictor,
            repository=InMemoryScoringRunRepository(),
            run_id_factory=lambda: f"demo-run-{next(run_ids):03d}",
            clock=lambda: datetime(2026, 10, 17, tzinfo=timezone.utc),
        )
    )
    with TestClient(app) as client:
        yield client


def test_create_scoring_run_schema_customers_and_order(
    scoring_client: TestClient,
) -> None:
    response = scoring_client.post("/scoring/runs", json={"dataset_id": "demo-dataset"})

    assert response.status_code == 201
    batch = PredictionBatch.model_validate_json(response.text)
    customer_ids = ["demo-C03", "demo-C01", "demo-C02"]
    expected_predictor = FakePredictor(seed=17)
    expected_by_id = {
        customer_id: expected_predictor.predict_batch(
            pd.DataFrame({"demo_usage": [0.0]}), [customer_id]
        )[0]
        for customer_id in customer_ids
    }

    assert batch.scoring_run_id == "demo-run-001"
    assert batch.model_version == "fake-v1"
    assert len(batch.predictions) == len(customer_ids)
    assert [prediction.customer_id for prediction in batch.predictions] == customer_ids
    for prediction in batch.predictions:
        assert prediction.churn_probability == expected_by_id[prediction.customer_id]
        assert prediction.dataset_id == "demo-dataset"
        assert prediction.model_version == batch.model_version
        assert prediction.scoring_run_id == batch.scoring_run_id
        assert prediction.scored_at == datetime(2026, 10, 17, tzinfo=timezone.utc)


def test_get_scoring_runs_preserves_each_saved_batch(
    scoring_client: TestClient,
) -> None:
    first = scoring_client.post("/scoring/runs", json={"dataset_id": "demo-dataset"})
    second = scoring_client.post("/scoring/runs", json={"dataset_id": "demo-dataset"})

    assert first.status_code == second.status_code == 201
    assert first.json()["scoring_run_id"] != second.json()["scoring_run_id"]
    for created in (first, second):
        run_id = created.json()["scoring_run_id"]
        response = scoring_client.get(f"/scoring/runs/{run_id}")
        assert response.status_code == 200
        assert response.json() == created.json()


def test_missing_dataset_returns_404(scoring_client: TestClient) -> None:
    response = scoring_client.post("/scoring/runs", json={"dataset_id": "demo-missing"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset 'demo-missing' not found"}


def test_missing_scoring_run_returns_404(scoring_client: TestClient) -> None:
    response = scoring_client.get("/scoring/runs/demo-missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Scoring run 'demo-missing' not found"}


def test_invalid_dataset_null_customer_id_returns_422() -> None:
    """R09: Dataset with null customer_id must return 422, not 500."""
    dataset = pd.DataFrame({"customer_id": [None, "C002"], "x": [1.0, 2.0]})

    def loader(dataset_id: str) -> pd.DataFrame:
        if dataset_id == "invalid-null":
            return dataset
        raise FileNotFoundError(dataset_id)

    predictor = FakePredictor(seed=17)
    predictor.load("demo-artifact")
    app = FastAPI()
    app.include_router(
        create_scoring_router(
            loader=loader,
            predictor=predictor,
            repository=InMemoryScoringRunRepository(),
            run_id_factory=lambda: "run-1",
            clock=lambda: datetime(2026, 10, 17, tzinfo=timezone.utc),
        )
    )
    with TestClient(app) as client:
        response = client.post("/scoring/runs", json={"dataset_id": "invalid-null"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "customer_id" in detail.lower()


def test_invalid_dataset_duplicate_customer_id_returns_422() -> None:
    """R09: Dataset with duplicate customer_id must return 422, not 500."""
    dataset = pd.DataFrame({"customer_id": ["C001", "C001"], "x": [1.0, 2.0]})

    def loader(dataset_id: str) -> pd.DataFrame:
        if dataset_id == "invalid-dup":
            return dataset
        raise FileNotFoundError(dataset_id)

    predictor = FakePredictor(seed=17)
    predictor.load("demo-artifact")
    app = FastAPI()
    app.include_router(
        create_scoring_router(
            loader=loader,
            predictor=predictor,
            repository=InMemoryScoringRunRepository(),
            run_id_factory=lambda: "run-1",
            clock=lambda: datetime(2026, 10, 17, tzinfo=timezone.utc),
        )
    )
    with TestClient(app) as client:
        response = client.post("/scoring/runs", json={"dataset_id": "invalid-dup"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "customer_id" in detail.lower() or "duplicate" in detail.lower()


def test_file_not_found_still_returns_404() -> None:
    """R09: FileNotFoundError must still return 404, not 422."""

    def loader(dataset_id: str) -> pd.DataFrame:
        raise FileNotFoundError(f"No such file: {dataset_id}")

    predictor = FakePredictor(seed=17)
    predictor.load("demo-artifact")
    app = FastAPI()
    app.include_router(
        create_scoring_router(
            loader=loader,
            predictor=predictor,
            repository=InMemoryScoringRunRepository(),
            run_id_factory=lambda: "run-1",
            clock=lambda: datetime(2026, 10, 17, tzinfo=timezone.utc),
        )
    )
    with TestClient(app) as client:
        response = client.post("/scoring/runs", json={"dataset_id": "missing"})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not found" in detail.lower()


def test_create_scoring_run_requires_dataset_id(scoring_client: TestClient) -> None:
    response = scoring_client.post("/scoring/runs", json={})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "dataset_id"]
