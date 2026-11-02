"""Immutable real-v1 regression oracles; no training fixtures or API clients."""

import json
import os
import pickle
import stat
import subprocess
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from churn_retention.adapters.retrieval.legacy_reference_store import (
    ReferenceRecord,
    ReferenceStore,
)
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from scripts.capture_llm_rag_golden import (
    ARTIFACT,
    PRIVATE,
    PUBLIC,
    ROOT,
    VALIDATION,
    digest_array,
    digest_bytes,
    digest_json,
    missing_probes,
    offline_without_fitting,
    preprocessing_snapshot,
    probe_snapshot,
    query_snapshot,
    validation_queries,
)

GOLDEN: dict[str, Any] = json.loads(PUBLIC.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = GOLDEN["queries"]


@pytest.fixture(scope="module")
def store() -> Iterator[ReferenceStore]:
    if not ARTIFACT.is_file() or not VALIDATION.is_file():
        message = "Real v1 artifact and validation CSV required; no replacement data"
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    with offline_without_fitting():
        loaded = ReferenceStore.load(ARTIFACT)
        before = pickle.dumps(loaded._preprocessor)
        yield loaded
        assert pickle.dumps(loaded._preprocessor) == before


@pytest.fixture(scope="module")
def queries(store: ReferenceStore) -> dict[str, dict[str, object]]:
    return validation_queries(store, CASES)


def test_golden_source_is_the_pinned_real_v1(store: ReferenceStore) -> None:
    source = GOLDEN["source"]
    assert store.reference_version == source["reference_version"]
    assert digest_bytes(ARTIFACT.read_bytes()) == source["artifact_sha256"]
    assert digest_bytes(VALIDATION.read_bytes()) == source["validation_sha256"]
    assert store.manifest.libraries == source["libraries"]


def test_fitted_preprocessing_schema_vocabulary_and_parameters(
    store: ReferenceStore,
) -> None:
    assert preprocessing_snapshot(store) == GOLDEN["preprocessing"]


def test_every_reference_matrix_value_and_row_order_are_unchanged(
    store: ReferenceStore,
) -> None:
    matrix = GOLDEN["matrix"]
    assert list(store.matrix.shape) == matrix["shape"]
    assert str(store.matrix.dtype) == matrix["dtype"]
    assert digest_array(store.matrix) == matrix["values_sha256"]
    assert (
        digest_json([row.customer_id for row in store.records])
        == matrix["ordered_reference_ids_sha256"]
    )
    # Compare transform output, rather than trusting just a stored matrix payload.
    names = [column.name for column in store.feature_schema]
    import pandas as pd

    transformed = store._preprocessor.transform(
        pd.DataFrame([dict(row.features) for row in store.records], columns=names)
    )
    assert digest_array(transformed) == matrix["values_sha256"]


@pytest.mark.parametrize("expected", CASES, ids=[case["name"] for case in CASES])
def test_validation_query_vectors_are_unchanged(
    store: ReferenceStore,
    queries: dict[str, dict[str, object]],
    expected: dict[str, Any],
) -> None:
    prepared = store.prepare_query(queries[expected["name"]])
    assert list(prepared.vector.shape) == expected["vector_shape"]
    assert str(prepared.vector.dtype) == expected["vector_dtype"]
    assert digest_array(prepared.vector) == expected["vector_sha256"]
    assert digest_json(dict(prepared.features)) == expected["cleaned_features_sha256"]
    assert list(prepared.features) == GOLDEN["preprocessing"]["input_feature_names"]


@pytest.mark.parametrize("expected", CASES, ids=[case["name"] for case in CASES])
def test_top_five_ids_distances_and_order_are_unchanged(
    store: ReferenceStore,
    queries: dict[str, dict[str, object]],
    expected: dict[str, Any],
) -> None:
    retriever = SimilarCustomerRetriever(store, k=5)
    query = queries[expected["name"]]
    results = retriever.retrieve(query)
    assert len(results) == 5
    assert (
        digest_json([row.customer_id for row in results])
        == expected["ordered_neighbor_ids_sha256"]
    )
    # IDs/order/vectors are exact. Only kernel-dependent distances get tolerance.
    np.testing.assert_allclose(
        [row.distance for row in results], expected["distances"], rtol=1e-12, atol=1e-12
    )
    assert [(row.distance, row.customer_id) for row in results] == sorted(
        (row.distance, row.customer_id) for row in results
    )
    # Poisoned metadata is synthetic, never the validation target or real IDs.
    poisoned = retriever.retrieve(
        {**query, "Churn": "IGNORED_SYNTHETIC_LABEL", "customerID": "synthetic-query"}
    )
    assert (
        digest_json([row.customer_id for row in poisoned])
        == expected["ordered_neighbor_ids_sha256"]
    )


@pytest.mark.parametrize("expected", CASES, ids=[case["name"] for case in CASES])
def test_complete_prompt_hash_structure_labels_and_example_order(
    store: ReferenceStore,
    queries: dict[str, dict[str, object]],
    expected: dict[str, Any],
) -> None:
    actual, _ = query_snapshot(
        store, SimilarCustomerRetriever(store), queries[expected["name"]]
    )
    assert actual["prompt"] == expected["prompt"]
    assert actual["prompt"]["reference_count"] == 5


@pytest.mark.parametrize("name", list(GOLDEN["missing_values"]))
def test_missing_values_validation_and_input_order_are_unchanged(
    store: ReferenceStore, queries: dict[str, dict[str, object]], name: str
) -> None:
    actual = probe_snapshot(store, missing_probes(queries["zero_tenure"])[name])
    assert actual == GOLDEN["missing_values"][name]


@pytest.mark.parametrize("exclude_self", [False, True])
def test_equal_distances_use_id_order_at_k_boundary_and_exclude_self(
    store: ReferenceStore,
    queries: dict[str, dict[str, object]],
    exclude_self: bool,
) -> None:
    query = queries["first"]
    prepared = store.prepare_query(query)
    # Six identical vectors cross the K=5 boundary in deliberately shuffled order.
    # All fixture IDs/labels here are synthetic; no preprocessing is fitted.
    records = tuple(
        ReferenceRecord(prepared.features, "No", f"synthetic-tie-{suffix}")
        for suffix in ("f", "b", "e", "a", "d", "c")
    )
    tied_store = ReferenceStore(
        store.manifest.model_copy(update={"row_count": 6}),
        records,
        np.repeat(prepared.vector, 6, axis=0),
        store._preprocessor,
        store._categories,
    )
    results = SimilarCustomerRetriever(tied_store).retrieve(
        query, customer_id="synthetic-tie-c" if exclude_self else None
    )
    suffixes = ("a", "b", "d", "e", "f") if exclude_self else ("a", "b", "c", "d", "e")
    assert [row.customer_id for row in results] == [
        f"synthetic-tie-{suffix}" for suffix in suffixes
    ]
    assert len({row.distance for row in results}) == 1
    np.testing.assert_allclose([row.distance for row in results], 0, atol=1e-7)


def test_internal_vectors_and_ids_match_without_exposing_them(
    store: ReferenceStore, queries: dict[str, dict[str, object]]
) -> None:
    if not PRIVATE.is_file():
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail(
                "Private golden observations required in strict local validation"
            )
        pytest.skip(
            "Private observations stay local; public digests still test all queries"
        )
    assert stat.S_IMODE(PRIVATE.stat().st_mode) == 0o600
    assert (
        subprocess.run(
            ["git", "check-ignore", "-q", "--", str(PRIVATE)], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    internal = json.loads(PRIVATE.read_text(encoding="utf-8"))
    assert internal["public_snapshot_sha256"] == digest_json(GOLDEN)
    retriever = SimilarCustomerRetriever(store)
    for case in CASES:
        _, actual = query_snapshot(store, retriever, queries[case["name"]])
        expected = internal["queries"][case["name"]]
        if actual["ordered_neighbor_ids"] != expected["ordered_neighbor_ids"]:
            pytest.fail(
                "Internal neighbor order changed; inspect private observations locally"
            )
        if not np.array_equal(actual["vector"], expected["vector"]):
            pytest.fail("Internal vector changed; inspect private observations locally")


def test_public_fixture_and_prompts_do_not_expose_reference_ids(
    store: ReferenceStore, queries: dict[str, dict[str, object]]
) -> None:
    public_text = PUBLIC.read_text(encoding="utf-8")
    if any(row.customer_id in public_text for row in store.records):
        pytest.fail("A reference identifier entered the public fixture")
    retriever = SimilarCustomerRetriever(store)
    builder = ChurnPromptBuilder(store.feature_schema)
    for query in queries.values():
        results = retriever.retrieve(query)
        text = builder.build(store.prepare_query(query).features, results)
        if any(row.customer_id in text for row in store.records):
            pytest.fail("A reference identifier entered an LLM prompt")


def test_capture_refuses_to_overwrite_existing_snapshots() -> None:
    from scripts.capture_llm_rag_golden import capture

    with pytest.raises(FileExistsError, match="immutable"):
        capture(PUBLIC, PRIVATE)
