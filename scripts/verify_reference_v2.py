"""Non-identifying v2/v1 comparison; v1 supplies JSON/NPY oracles only."""

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.reference_v2 import VerificationReferenceV2
from churn_retention.contracts.reference import sha256
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder

if TYPE_CHECKING:
    from churn_retention.adapters.retrieval.reference_store import ReferenceStore

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts/references/telco-train-v2.zip"


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def verify(path: Path = V2) -> dict[str, Any]:
    baseline = json.loads((ROOT / "tests/fixtures/llm_rag_v1_golden.json").read_text())
    v1 = ROOT / "artifacts/references/telco-train-v1.zip"
    validation = ROOT / "data/processed/val.csv"
    if (
        sha256(v1.read_bytes()) != baseline["source"]["artifact_sha256"]
        or sha256(validation.read_bytes()) != baseline["source"]["validation_sha256"]
    ):
        raise ValueError("The immutable v1/validation oracle sources changed.")
    loaded = VerificationReferenceV2.load(path, trusted=True)
    with zipfile.ZipFile(v1) as archive:
        old_records = json.loads(archive.read("records.json"))
        old_matrix = np.load(io.BytesIO(archive.read("matrix.npy")), allow_pickle=False)
        old_manifest = json.loads(archive.read("manifest.json"))
    records = [
        {
            "customer_id": row.customer_id,
            "features": dict(row.features),
            "churn": row.churn,
        }
        for row in loaded.records
    ]
    retriever = SimilarCustomerRetriever(cast("ReferenceStore", loaded), k=5)
    builder = ChurnPromptBuilder(loaded.feature_schema)
    names = list(loaded.preprocessor.input_feature_names)
    frame = pd.read_csv(validation, usecols=names)
    queries = []
    for case in baseline["queries"]:
        query = {name: frame.iloc[case["row"]][name] for name in names}
        prepared = loaded.prepare_query(query)
        neighbors = retriever.retrieve(query)
        prompt = builder.build_with_metadata(prepared.features, neighbors)
        payload = json.loads(
            prompt.text.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split(
                "\nEND_CUSTOMER_DATA_JSON\n", 1
            )[0]
        )
        queries.append(
            {
                "name": case["name"],
                "vector_exact": sha256(
                    np.asarray(prepared.vector, dtype="<f8", order="C").tobytes()
                )
                == case["vector_sha256"],
                "top5_ids_and_order_exact": _digest([row.customer_id for row in neighbors])
                == case["ordered_neighbor_ids_sha256"],
                "distance_max_absolute_error": float(
                    np.max(
                        np.abs(np.asarray([row.distance for row in neighbors]) - case["distances"])
                    )
                ),
                "examples_exact": prompt.reference_count == 5
                and _digest(payload["reference_customers"])
                == case["prompt"]["ordered_examples_sha256"],
                "prompt_hash": sha256(prompt.text.encode()),
                "prompt_exact": sha256(prompt.text.encode()) == case["prompt"]["sha256"],
                "prompt_has_no_reference_ids": not any(
                    row.customer_id in prompt.text for row in loaded.records
                ),
            }
        )
    report = {
        "reference_version": loaded.reference_version,
        "reference_version_differs_from_v1": loaded.reference_version
        != old_manifest["reference_version"],
        "records_count": len(loaded.records),
        "records_features_labels_ids_and_order_exact": records == old_records,
        "train_hash_exact": loaded.manifest.source_train_sha256 == old_manifest["dataset_sha256"],
        "split_hash_exact": loaded.manifest.split_sha256 == old_manifest["split_sha256"],
        "preprocessing_exact": loaded.preprocessor.metadata["preprocessing"]
        == baseline["preprocessing"],
        "matrix_shape": list(loaded.matrix.shape),
        "matrix_dtype": str(loaded.matrix.dtype),
        "matrix_exact": bool(np.array_equal(loaded.matrix, old_matrix)),
        "matrix_max_absolute_error": float(np.max(np.abs(loaded.matrix - old_matrix))),
        "matrix_values_sha256": sha256(np.asarray(loaded.matrix, dtype="<f8", order="C").tobytes()),
        "matrix_hash_exact": loaded.manifest.matrix_values_sha256
        == baseline["matrix"]["values_sha256"],
        "queries": queries,
    }
    report["equivalent"] = all(
        report[key]
        for key in (
            "reference_version_differs_from_v1",
            "records_features_labels_ids_and_order_exact",
            "train_hash_exact",
            "split_hash_exact",
            "preprocessing_exact",
            "matrix_exact",
            "matrix_hash_exact",
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
        for row in queries
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=V2)
    report = verify(parser.parse_args().path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
