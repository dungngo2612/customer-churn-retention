"""Capture the existing v1 behavior offline; never overwrite a golden snapshot."""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from churn_retention.adapters.retrieval.legacy_reference_store import ReferenceStore
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/references/telco-train-v1.zip"
VALIDATION = ROOT / "data/processed/val.csv"
PUBLIC = ROOT / "tests/fixtures/llm_rag_v1_golden.json"
PRIVATE = ROOT / "artifacts/golden/llm-rag-v1-private.json"
# Fixed, zero-based source positions, selected using features only, never Churn.
QUERY_ROWS = (
    ("first", 0),
    ("zero_tenure", 104),
    ("no_phone", 31),
    ("no_internet", 7),
    ("senior", 9),
    ("one_year", 5),
    ("two_year", 1),
    ("maximum_total_charges", 393),
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_value(value: Any) -> Any:
    """Serialize estimator parameters/state without pickle or arbitrary repr()."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported preprocessing snapshot value")


def digest_json(value: Any) -> str:
    return digest_bytes(
        json.dumps(
            json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def digest_array(value: Any) -> str:
    """Hash all float64 values in C order with a fixed little-endian encoding."""
    return digest_bytes(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))


@contextmanager
def offline_without_fitting() -> Iterator[None]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Golden capture/tests must not fit or use the network")

    with ExitStack() as stack:
        for cls in (
            ColumnTransformer,
            SimpleImputer,
            StandardScaler,
            OneHotEncoder,
            Pipeline,
        ):
            for method in ("fit", "fit_transform"):
                stack.enter_context(patch.object(cls, method, forbidden))
        for method in ("predict", "predict_proba"):
            stack.enter_context(patch.object(Pipeline, method, forbidden))
        for method in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, method, forbidden))
        stack.enter_context(patch.object(socket, "create_connection", forbidden))
        yield


def validation_queries(
    store: ReferenceStore, cases: list[dict[str, Any]]
) -> dict[str, dict[str, object]]:
    # Read only feature columns: neither validation labels nor IDs enter this path.
    names = [column.name for column in store.feature_schema]
    frame = pd.read_csv(VALIDATION, usecols=names)
    return {case["name"]: {name: frame.iloc[case["row"]][name] for name in names} for case in cases}


def preprocessing_snapshot(store: ReferenceStore) -> dict[str, Any]:
    preprocessor = store._preprocessor
    steps = {}
    for group in ("num", "cat"):
        for name, estimator in preprocessor.named_transformers_[group].steps:
            state = {}
            for attribute in (
                "statistics_",
                "mean_",
                "var_",
                "scale_",
                "n_samples_seen_",
                "categories_",
                "drop_idx_",
                "n_features_in_",
            ):
                if hasattr(estimator, attribute):
                    state[attribute] = getattr(estimator, attribute)
            steps[f"{group}.{name}"] = {
                "class": type(estimator).__name__,
                "parameters": estimator.get_params(deep=False),
                "state": state,
            }
    return json_value(
        {
            "schema": [column.model_dump() for column in store.feature_schema],
            "input_feature_names": preprocessor.feature_names_in_,
            "output_feature_names": preprocessor.get_feature_names_out(),
            "categorical_vocabulary": store._categories,
            "parameters": {
                key: value
                for key, value in preprocessor.get_params(deep=False).items()
                if key != "transformers"
            },
            "transformer_order": [
                {"name": name, "columns": list(columns)}
                for name, _, columns in preprocessor.transformers_
            ],
            "sparse_output": preprocessor.sparse_output_,
            "steps": steps,
        }
    )


def prompt_snapshot(text: str, reference_count: int) -> dict[str, Any]:
    prefix, rest = text.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)
    encoded, suffix = rest.split("\nEND_CUSTOMER_DATA_JSON\n", 1)
    payload = json.loads(encoded)
    schema = json.loads(
        prefix.split("\nBEGIN_OUTPUT_JSON_SCHEMA\n", 1)[1].split("\nEND_OUTPUT_JSON_SCHEMA\n", 1)[0]
    )
    return {
        "sha256": digest_bytes(text.encode("utf-8")),
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "reference_count": reference_count,
        "reference_labels": [example["Churn"] for example in payload["reference_customers"]],
        "ordered_examples_sha256": digest_json(payload["reference_customers"]),
        "template_sha256": digest_bytes((prefix + "\n<DATA>\n" + suffix).encode()),
        "structure": {
            "data_keys": sorted(payload),
            "query_feature_names": sorted(payload["customer_to_predict"]),
            "example_keys": [sorted(example) for example in payload["reference_customers"]],
            "output_schema": schema,
        },
    }


def query_snapshot(
    store: ReferenceStore,
    retriever: SimilarCustomerRetriever,
    query: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = store.prepare_query(query)
    neighbors = retriever.retrieve(query)
    built = ChurnPromptBuilder(store.feature_schema).build_with_metadata(
        prepared.features, neighbors
    )
    ids = [neighbor.customer_id for neighbor in neighbors]
    if any(identifier in built.text for identifier in ids):
        raise AssertionError("Reference identifiers must never enter a prompt")
    public = {
        "cleaned_features_sha256": digest_json(dict(prepared.features)),
        "vector_sha256": digest_array(prepared.vector),
        "vector_shape": list(prepared.vector.shape),
        "vector_dtype": str(prepared.vector.dtype),
        # Only a digest of the entire ordered ID list is versioned publicly.
        "ordered_neighbor_ids_sha256": digest_json(ids),
        "distances": [neighbor.distance for neighbor in neighbors],
        "prompt": prompt_snapshot(built.text, built.reference_count),
    }
    private = {
        "vector": prepared.vector.tolist(),
        "ordered_neighbor_ids": ids,
    }
    return public, private


def missing_probes(query: Mapping[str, object]) -> dict[str, dict[str, object]]:
    probes = {
        name: {**query, "TotalCharges": value}
        for name, value in (
            ("total_none", None),
            ("total_empty", ""),
            ("total_whitespace", "  "),
            ("total_nan", float("nan")),
            ("total_pd_na", pd.NA),
        )
    }
    probes.update(
        total_missing_nonzero_tenure={**query, "tenure": 1, "TotalCharges": None},
        other_numeric_missing={**query, "MonthlyCharges": None},
        category_missing={**query, "gender": None},
        category_unknown={**query, "gender": "__UNKNOWN__"},
        senior_boolean={**query, "SeniorCitizen": True},
        fractional_tenure={**query, "tenure": 0.5},
        negative_numeric={**query, "MonthlyCharges": -1},
        nonfinite_numeric={**query, "MonthlyCharges": float("inf")},
        reversed_input_order=dict(reversed(list(query.items()))),
        missing_feature={key: value for key, value in query.items() if key != "gender"},
        unsupported_field={**query, "unsupported": "synthetic"},
    )
    return probes


def probe_snapshot(store: ReferenceStore, query: Mapping[str, object]) -> dict[str, Any]:
    try:
        prepared = store.prepare_query(query)
    except (ValueError, TypeError) as error:
        return {
            "accepted": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {
        "accepted": True,
        "cleaned_features_sha256": digest_json(dict(prepared.features)),
        "vector_sha256": digest_array(prepared.vector),
    }


def capture(public_path: Path, private_path: Path) -> None:
    if public_path.exists() or private_path.exists():
        raise FileExistsError("Golden snapshots are immutable; choose new output paths")
    if subprocess.run(
        ["git", "check-ignore", "-q", "--", str(private_path)], cwd=ROOT, check=False
    ).returncode:
        raise ValueError("Private golden output must be Git-ignored")
    with offline_without_fitting():
        store = ReferenceStore.load(ARTIFACT)
        retriever = SimilarCustomerRetriever(store, k=5)
        cases: list[dict[str, Any]] = [{"name": name, "row": row} for name, row in QUERY_ROWS]
        queries = validation_queries(store, cases)
        public: dict[str, Any] = {
            "baseline_format": 1,
            "source": {
                "reference_version": store.reference_version,
                "artifact_sha256": digest_bytes(ARTIFACT.read_bytes()),
                "validation_sha256": digest_bytes(VALIDATION.read_bytes()),
                "libraries": store.manifest.libraries,
            },
            "preprocessing": preprocessing_snapshot(store),
            "matrix": {
                "shape": list(store.matrix.shape),
                "dtype": str(store.matrix.dtype),
                "values_sha256": digest_array(store.matrix),
                "ordered_reference_ids_sha256": digest_json(
                    [record.customer_id for record in store.records]
                ),
            },
            "retrieval": {
                "k": 5,
                "metric": "euclidean",
                "algorithm": "brute",
                "tie_breaking": "distance ascending, then customer_id ascending",
                "self_exclusion": "exclude the query customer_id when provided",
            },
            "queries": [],
            "missing_values": {
                name: probe_snapshot(store, query)
                for name, query in missing_probes(queries["zero_tenure"]).items()
            },
        }
        private: dict[str, Any] = {"baseline_format": 1, "queries": {}}
        for case in cases:
            observed, internal = query_snapshot(store, retriever, queries[case["name"]])
            public["queries"].append({**case, **observed})
            private["queries"][case["name"]] = internal
        private["public_snapshot_sha256"] = digest_json(public)
    # No raw query rows, prompts, labels of validation, or IDs enter the public file.
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with public_path.open("x", encoding="utf-8") as handle:
        json.dump(public, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(private, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"queries": len(cases), "matrix_shape": public["matrix"]["shape"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args()
    capture(args.public.resolve(), args.private.resolve())


if __name__ == "__main__":
    main()
