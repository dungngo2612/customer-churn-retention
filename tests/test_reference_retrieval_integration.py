"""Read existing project data; skip explicitly if local generated assets are absent."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from churn_retention.adapters.retrieval.legacy_reference_store import ReferenceStore
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.ml.training.bundle import load_bundle
from churn_retention.workflows.build_reference import build_reference

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/train.csv"
VAL = ROOT / "data/processed/val.csv"
SPLITS = ROOT / "artifacts/splits/dataset_splits.csv"
BUNDLE = ROOT / "artifacts/bundles/logistic-20261018T130159-seed42"
pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in [TRAIN, VAL, SPLITS, BUNDLE / "model.pkl"]),
    reason=(
        "Existing project CSVs and saved Logistic bundle are required"
        "; no fake project data is substituted."
    ),
)


@pytest.fixture
def project_store(tmp_path: Path) -> ReferenceStore:
    output = tmp_path / "telco-reference.zip"
    build_reference(TRAIN, BUNDLE, SPLITS, output)
    return ReferenceStore.load(output)


def test_real_reference_contains_exactly_train_and_reuses_saved_transform(
    project_store: ReferenceStore,
) -> None:
    train = pd.read_csv(TRAIN).sort_values("customerID")
    splits = pd.read_csv(SPLITS)
    source_bundle = load_bundle(BUNDLE)
    store = project_store
    assert len(store.records) == 4930
    assert store.matrix.shape == (4930, 45)
    assert len(store.feature_schema) == 19
    assert {r.customer_id for r in store.records} == set(
        splits.loc[splits.split.eq("train"), "customerID"]
    )
    assert not {r.customer_id for r in store.records} & set(
        splits.loc[splits.split.isin(["val", "test"]), "customerID"]
    )
    assert {r.customer_id: r.churn for r in store.records} == dict(
        zip(train.customerID, train.Churn, strict=True)
    )
    names = [c.name for c in store.feature_schema]
    expected = source_bundle._pipeline.named_steps["preprocessor"].transform(train[names])
    np.testing.assert_array_equal(store.matrix, expected)


def test_real_validation_query_has_train_neighbors_only_without_csv_rereads(
    project_store: ReferenceStore,
) -> None:
    row = {str(name): value for name, value in pd.read_csv(VAL).iloc[0].items()}
    retriever = SimilarCustomerRetriever(project_store)
    prepared = project_store.prepare_query(row)
    expected = sorted(
        zip(
            np.linalg.norm(project_store.matrix - prepared.vector, axis=1),
            project_store.records,
            strict=True,
        ),
        key=lambda pair: (pair[0], pair[1].customer_id),
    )[:5]
    with patch.object(pd, "read_csv", side_effect=AssertionError("No request-time CSV reads")):
        first = retriever.retrieve(row)
        second = retriever.retrieve({**row, "Churn": "ignored"})
    assert first == second
    assert len(first) == 5
    assert [r.customer_id for r in first] == [r.customer_id for _, r in expected]
    np.testing.assert_allclose([r.distance for r in first], [d for d, _ in expected], atol=1e-7)
    assert all(set(r.to_prompt_data()) == {"features", "Churn"} for r in first)
