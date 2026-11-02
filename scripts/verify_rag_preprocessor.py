"""Compare a freshly fitted RAG preprocessor with immutable v1 golden oracles.

The in-memory view below is only a verification harness for the existing
retriever. It is not a new Store loader, artifact format, or runtime integration.
No sklearn pickle is read; no classifier instance is deserialized or invoked.
"""

import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
from churn_retention.contracts.reference import (
    PreparedQuery,
    ReferenceRecord,
    StoredRecord,
    sha256,
    validate_id,
)
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder

if TYPE_CHECKING:
    from churn_retention.adapters.retrieval.reference_store import ReferenceStore

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/train.csv"
SPLITS = ROOT / "artifacts/splits/dataset_splits.csv"
VALIDATION = ROOT / "data/processed/val.csv"
ARTIFACT = ROOT / "artifacts/references/telco-train-v1.zip"
GOLDEN = ROOT / "tests/fixtures/llm_rag_v1_golden.json"
PRIVATE = ROOT / "artifacts/golden/llm-rag-v1-private.json"


def json_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def array_digest(value: NDArray[np.float64]) -> str:
    return sha256(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))


class CandidateReferenceView:
    """Test/tool-only Store shape; every query/vector comes from the new code."""

    def __init__(
        self,
        preprocessor: RAGPreprocessor,
        records: tuple[ReferenceRecord, ...],
        matrix: NDArray[np.float64],
    ) -> None:
        self.preprocessor = preprocessor
        self.records = records
        self.matrix = matrix

    def prepare_query(
        self, query: Mapping[str, object], *, customer_id: str | None = None
    ) -> PreparedQuery:
        ids = [validate_id(query[name]) for name in ("customerID", "customer_id") if name in query]
        if customer_id is not None:
            ids.append(validate_id(customer_id))
        if len(set(ids)) > 1:
            raise ValueError("Conflicting query customer IDs.")
        features = self.preprocessor.clean(query)
        vector = self.preprocessor.transform_one(query)
        vector.setflags(write=False)
        return PreparedQuery(MappingProxyType(features), vector, ids[0] if ids else None)


def candidate_view(
    preprocessor: RAGPreprocessor,
) -> tuple[CandidateReferenceView, NDArray[np.float64]]:
    with zipfile.ZipFile(ARTIFACT) as archive:
        # Records/labels are RAG reference data; matrix.npy is the immutable oracle.
        # bundle/model.pkl is deliberately never read or unpickled.
        stored = [
            StoredRecord.model_validate(row) for row in json.loads(archive.read("records.json"))
        ]
        expected = np.load(io.BytesIO(archive.read("matrix.npy")), allow_pickle=False)
    records = tuple(
        ReferenceRecord(MappingProxyType(row.features), row.churn, row.customer_id)
        for row in stored
    )
    actual = preprocessor.transform_many([row.features for row in records])
    return CandidateReferenceView(preprocessor, records, actual), expected


def make_retriever(view: CandidateReferenceView) -> SimilarCustomerRetriever:
    # Reuse the unchanged algorithm. This structural cast stays in test/tool code.
    return SimilarCustomerRetriever(cast("ReferenceStore", view), k=5)


def validation_queries(
    preprocessor: RAGPreprocessor, cases: list[dict[str, Any]]
) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(VALIDATION, usecols=list(preprocessor.input_feature_names))
    return {
        case["name"]: {
            name: frame.iloc[case["row"]][name] for name in preprocessor.input_feature_names
        }
        for case in cases
    }


def verify() -> dict[str, Any]:
    golden = json.loads(GOLDEN.read_text())
    if (
        sha256(ARTIFACT.read_bytes()) != golden["source"]["artifact_sha256"]
        or sha256(VALIDATION.read_bytes()) != golden["source"]["validation_sha256"]
    ):
        raise ValueError("The immutable v1/validation oracle sources changed.")
    # Actual: fit from train and splits. Expected: frozen JSON/NPY, never a model.
    preprocessor = RAGPreprocessor.fit_train(TRAIN, SPLITS)
    view, expected_matrix = candidate_view(preprocessor)
    expected_vectors = json.loads(PRIVATE.read_text())["queries"] if PRIVATE.is_file() else {}
    retriever = make_retriever(view)
    builder = ChurnPromptBuilder(preprocessor.feature_schema)
    queries = validation_queries(preprocessor, golden["queries"])
    observations = []
    for case in golden["queries"]:
        query = queries[case["name"]]
        vector = preprocessor.transform_one(query)
        neighbors = retriever.retrieve(query)
        built = builder.build_with_metadata(preprocessor.clean(query), neighbors)
        payload = json.loads(
            built.text.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split(
                "\nEND_CUSTOMER_DATA_JSON\n", 1
            )[0]
        )
        vector_matches = array_digest(vector) == case["vector_sha256"]
        private = expected_vectors.get(case["name"])
        max_error = (
            float(np.max(np.abs(vector - np.asarray(private["vector"]))))
            if private
            else 0.0
            if vector_matches
            else None
        )
        observations.append(
            {
                "name": case["name"],
                "vector_shape": list(vector.shape),
                "vector_dtype": str(vector.dtype),
                "vector_exact": vector_matches,
                "vector_max_absolute_error": max_error,
                "top5_ids_and_order_exact": json_digest([row.customer_id for row in neighbors])
                == case["ordered_neighbor_ids_sha256"],
                "distance_max_absolute_error": float(
                    np.max(
                        np.abs(np.asarray([row.distance for row in neighbors]) - case["distances"])
                    )
                ),
                "examples_exact": built.reference_count == case["prompt"]["reference_count"]
                and json_digest(payload["reference_customers"])
                == case["prompt"]["ordered_examples_sha256"],
                "prompt_hash": sha256(built.text.encode("utf-8")),
                "prompt_exact": sha256(built.text.encode("utf-8")) == case["prompt"]["sha256"],
                "prompt_has_no_reference_ids": not any(
                    row.customer_id in built.text for row in view.records
                ),
            }
        )
    report = {
        "fitted_statistics_and_parameters_exact": preprocessor.metadata["preprocessing"]
        == golden["preprocessing"],
        "fit_row_count": preprocessor.metadata["fit"]["row_count"],
        "matrix_shape": list(view.matrix.shape),
        "matrix_dtype": str(view.matrix.dtype),
        "matrix_exact": bool(np.array_equal(view.matrix, expected_matrix)),
        "matrix_max_absolute_error": float(np.max(np.abs(view.matrix - expected_matrix))),
        "matrix_hash": array_digest(view.matrix),
        "matrix_hash_exact": array_digest(view.matrix) == golden["matrix"]["values_sha256"],
        "reference_order_exact": json_digest([row.customer_id for row in view.records])
        == golden["matrix"]["ordered_reference_ids_sha256"],
        "queries": observations,
    }
    report["equivalent"] = all(
        report[key]
        for key in (
            "fitted_statistics_and_parameters_exact",
            "matrix_exact",
            "matrix_hash_exact",
            "reference_order_exact",
        )
    ) and all(
        all(
            row[key]
            for key in (
                "vector_exact",
                "top5_ids_and_order_exact",
                "examples_exact",
                "prompt_exact",
                "prompt_has_no_reference_ids",
            )
        )
        and row["distance_max_absolute_error"] <= 1e-12
        for row in observations
    )
    return report


def main() -> None:
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
