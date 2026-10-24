import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest

from churn_retention.adapters.predictor.fake import FakePredictor
from churn_retention.modules.scoring.repository import (
    InMemoryScoringRunRepository,
    ScoringRunRepository,
)
from churn_retention.modules.scoring.use_cases import DatasetNotFoundError, score_dataset


@pytest.fixture
def demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["demo-C03", "demo-C01", "demo-C02"],
            "demo_usage": [30.0, 10.0, 20.0],
        },
        index=[21, 4, 13],
    )


def test_score_dataset_keys_predictions_and_preserves_order(
    demo_dataset: pd.DataFrame,
) -> None:
    original = demo_dataset.copy(deep=True)
    predictor = FakePredictor(seed=17)
    predictor.load("demo-artifact-unused")
    scored_at = datetime(2026, 10, 17, tzinfo=timezone.utc)

    def loader(dataset_id: str) -> pd.DataFrame:
        assert dataset_id == "demo-dataset"
        return demo_dataset

    batch = score_dataset(
        "demo-dataset", loader, predictor, lambda: "demo-run", lambda: scored_at
    )
    expected_by_id = {
        customer_id: predictor.predict_batch(
            pd.DataFrame({"demo_usage": [0.0]}), [customer_id]
        )[0]
        for customer_id in demo_dataset["customer_id"]
    }

    assert batch.scoring_run_id == "demo-run"
    assert batch.model_version == "fake-v1"
    assert [prediction.customer_id for prediction in batch.predictions] == list(
        demo_dataset["customer_id"]
    )
    for prediction in batch.predictions:
        assert prediction.churn_probability == expected_by_id[prediction.customer_id]
        assert prediction.dataset_id == "demo-dataset"
        assert prediction.scoring_run_id == batch.scoring_run_id
        assert prediction.model_version == batch.model_version
        assert prediction.scored_at == scored_at
    pd.testing.assert_frame_equal(demo_dataset, original)


@pytest.mark.parametrize(
    "dataset, message",
    [
        (pd.DataFrame({"demo_usage": [1.0]}), "must contain customer_id"),
        (pd.DataFrame({"customer_id": [None]}), "non-null strings"),
        (pd.DataFrame({"customer_id": [123]}), "non-null strings"),
        (pd.DataFrame({"customer_id": ["demo-C01", "demo-C01"]}), "must be unique"),
    ],
)
def test_score_dataset_rejects_invalid_customer_keys(
    dataset: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_dataset(
            "demo-dataset",
            lambda dataset_id: dataset,
            FakePredictor(),
            lambda: "demo-run",
            lambda: datetime.now(timezone.utc),
        )


def test_score_dataset_rejects_wrong_probability_count(
    demo_dataset: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictor = FakePredictor()

    def wrong_count(features: pd.DataFrame, customer_ids: Sequence[str]) -> list[float]:
        return [0.5]

    monkeypatch.setattr(predictor, "predict_batch", wrong_count)

    with pytest.raises(ValueError, match="returned 1 probabilities for 3 customers"):
        score_dataset(
            "demo-dataset",
            lambda dataset_id: demo_dataset,
            predictor,
            lambda: "demo-run",
            lambda: datetime.now(timezone.utc),
        )


def test_score_dataset_missing_dataset() -> None:
    def loader(dataset_id: str) -> pd.DataFrame:
        raise FileNotFoundError(dataset_id)

    with pytest.raises(DatasetNotFoundError, match="Dataset 'demo-missing' not found"):
        score_dataset(
            "demo-missing",
            loader,
            FakePredictor(),
            lambda: "demo-run",
            lambda: datetime.now(timezone.utc),
        )


def test_score_dataset_empty_dataset() -> None:
    batch = score_dataset(
        "demo-empty",
        lambda dataset_id: pd.DataFrame(columns=["customer_id", "demo_usage"]),
        FakePredictor(),
        lambda: "demo-run",
        lambda: datetime.now(timezone.utc),
    )

    assert batch.predictions == []


def test_scoring_run_repository_round_trip(demo_dataset: pd.DataFrame) -> None:
    repository: ScoringRunRepository = InMemoryScoringRunRepository()
    batch = score_dataset(
        "demo-dataset",
        lambda dataset_id: demo_dataset,
        FakePredictor(),
        lambda: "demo-run",
        lambda: datetime.now(timezone.utc),
    )
    repository.save(batch)

    assert repository.get("demo-run") == batch
    assert repository.get("demo-missing") is None
    assert InMemoryScoringRunRepository().get("demo-run") is None


def test_use_cases_does_not_import_ml_libraries(tmp_path: Path) -> None:
    script = """
import sys
import churn_retention.modules.scoring.use_cases

assert "catboost" not in sys.modules
assert "sklearn" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
