"""Neutral RAG contracts work without adapters, sklearn, or model bundle access."""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_retention.contracts.reference import (
    FeatureColumn,
    PreparedQuery,
    ReferenceArtifactError,
    ReferenceRecord,
    SimilarCustomer,
    StoredRecord,
    clean_features,
    json_bytes,
    sha256,
    validate_id,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests/fixtures/llm_rag_v1_golden.json").read_text())


@pytest.fixture
def schema() -> list[FeatureColumn]:
    return [FeatureColumn.model_validate(column) for column in BASELINE["preprocessing"]["schema"]]


@pytest.fixture
def categories() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(values)
        for name, values in BASELINE["preprocessing"]["categorical_vocabulary"].items()
    }


@pytest.fixture
def customer(categories: dict[str, tuple[str, ...]]) -> dict[str, object]:
    # Synthetic values; no dataset, saved model, or fitted transformer is used.
    return {
        "SeniorCitizen": "0",
        "tenure": "0",
        "MonthlyCharges": "20.5",
        "TotalCharges": None,
        **{name: values[0] for name, values in categories.items()},
    }


@pytest.mark.parametrize(
    "module",
    [
        "churn_retention.contracts.reference",
        "churn_retention.modules.scoring.llm_prompt",
    ],
)
def test_fresh_import_without_sklearn_adapters_or_model_files(tmp_path: Path, module: str) -> None:
    script = r"""
import importlib
import importlib.abc
import os
import socket
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
blocked = (
    "sklearn", "churn_retention.ml", "churn_retention.adapters",
    "churn_retention.workflows",
)

class ImportBarrier(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Neutral imports must not load legacy/adapters/sklearn")

def no_bundle_reads(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).replace("\\", "/")
        if path.endswith("/model.pkl") or path == "model.pkl" or "/artifacts/bundles/" in path:
            raise AssertionError("Neutral imports must not read model bundles")

def no_network(*args, **kwargs):
    raise AssertionError("No network in import isolation tests")

sys.meta_path.insert(0, ImportBarrier())
sys.addaudithook(no_bundle_reads)
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
loaded = importlib.import_module(sys.argv[2])
reference = importlib.import_module("churn_retention.contracts.reference")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules for prefix in blocked
)
if sys.argv[2].endswith("llm_prompt"):
    assert loaded.FeatureColumn is reference.FeatureColumn
    assert loaded.SimilarCustomer is reference.SimilarCustomer
print("isolated import passed")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT / "src"), module],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "isolated import passed"


@pytest.mark.parametrize("missing", [None, "", "  ", float("nan"), pd.NA])
def test_valid_inputs_clean_to_the_same_canonical_features(
    schema: list[FeatureColumn],
    categories: dict[str, tuple[str, ...]],
    customer: dict[str, object],
    missing: object,
) -> None:
    validate_schema(schema)
    customer["TotalCharges"] = missing
    original = dict(customer)
    cleaned = clean_features(dict(reversed(list(customer.items()))), schema, categories)
    assert list(cleaned) == [column.name for column in schema]
    assert cleaned == {
        **{name: values[0] for name, values in categories.items()},
        "SeniorCitizen": 0,
        "tenure": 0,
        "MonthlyCharges": 20.5,
        "TotalCharges": 0.0,
    }
    assert type(cleaned["tenure"]) is int
    assert type(cleaned["MonthlyCharges"]) is float
    assert customer == original


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"tenure": 1}, ValueError, "Missing TotalCharges"),
        ({"SeniorCitizen": True}, TypeError, "numerical scalar"),
        ({"SeniorCitizen": 2}, ValueError, "SeniorCitizen must be 0 or 1"),
        ({"tenure": 0.5}, ValueError, "tenure must be an integer"),
        ({"MonthlyCharges": -1}, ValueError, "finite and non-negative"),
        ({"MonthlyCharges": float("inf")}, ValueError, "finite and non-negative"),
        ({"MonthlyCharges": float("nan")}, ValueError, "finite and non-negative"),
        ({"MonthlyCharges": None}, TypeError, "numerical scalar"),
        ({"gender": None}, ValueError, "Invalid categorical value"),
        ({"gender": "__UNKNOWN__"}, ValueError, "Invalid categorical value"),
        ({"unexpected": 1}, ValueError, "unsupported fields"),
    ],
)
def test_invalid_inputs_retain_existing_validation_errors(
    schema: list[FeatureColumn],
    categories: dict[str, tuple[str, ...]],
    customer: dict[str, object],
    changes: Mapping[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        clean_features({**customer, **changes}, schema, categories)


def test_missing_feature_is_rejected(
    schema: list[FeatureColumn],
    categories: dict[str, tuple[str, ...]],
    customer: dict[str, object],
) -> None:
    customer.pop("gender")
    with pytest.raises(ValueError, match="missing required features"):
        clean_features(customer, schema, categories)


@pytest.mark.parametrize(
    "kind", ["count", "duplicate", "metadata", "numeric_dtype", "category_dtype"]
)
def test_invalid_schemas_are_rejected(schema: list[FeatureColumn], kind: str) -> None:
    if kind == "count":
        schema.pop()
    elif kind == "duplicate":
        schema[-1] = schema[-2]
    else:
        index = 0 if kind == "numeric_dtype" else -1
        update = (
            {"name": "customerID"}
            if kind == "metadata"
            else {"dtype": "object" if kind == "numeric_dtype" else "int64"}
        )
        schema[index] = schema[index].model_copy(update=update)
    with pytest.raises(ReferenceArtifactError):
        validate_schema(schema)


def test_shared_records_keep_id_free_projection_and_vector_repr(
    schema: list[FeatureColumn],
    categories: dict[str, tuple[str, ...]],
    customer: dict[str, object],
) -> None:
    features = clean_features(customer, schema, categories)
    stored = StoredRecord(customer_id="synthetic-reference", features=features, churn="No")
    reference = ReferenceRecord(features, stored.churn, stored.customer_id)
    similar = SimilarCustomer(features, stored.churn, stored.customer_id, 0.25)
    query = PreparedQuery(features, np.zeros((1, 45)), "synthetic-query")
    assert (
        reference.to_prompt_data()
        == similar.to_prompt_data()
        == {
            "features": features,
            "Churn": "No",
        }
    )
    assert "synthetic-reference" not in repr(reference)
    assert "synthetic-query" not in repr(query) and "array(" not in repr(query)


def test_canonical_serialization_and_hash_semantics() -> None:
    assert json_bytes({"b": "Việt", "a": 1}) == '{"a":1,"b":"Việt"}'.encode()
    assert sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        json_bytes({"value": float("nan")})


@pytest.mark.parametrize("invalid", ["", " ", " synthetic ", None, 1])
def test_invalid_ids_are_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="Customer ID"):
        validate_id(invalid)


def test_old_import_paths_reexport_the_same_contract_objects() -> None:
    from churn_retention.adapters.retrieval import (
        _artifact,
        nearest_neighbors,
        reference_store,
    )
    from churn_retention.contracts import reference
    from churn_retention.modules.scoring import llm_prompt

    for name in (
        "FeatureColumn",
        "StoredRecord",
        "ReferenceArtifactError",
        "clean_features",
        "validate_schema",
        "validate_id",
        "sha256",
        "json_bytes",
    ):
        assert getattr(_artifact, name) is getattr(reference, name)
    assert reference_store.ReferenceRecord is ReferenceRecord
    assert reference_store.PreparedQuery is PreparedQuery
    assert nearest_neighbors.SimilarCustomer is SimilarCustomer
    assert llm_prompt.FeatureColumn is FeatureColumn
    assert llm_prompt.SimilarCustomer is SimilarCustomer
