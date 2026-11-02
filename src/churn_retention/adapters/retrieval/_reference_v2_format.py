"""Strict v2 manifest and bounded archive/JSON rules; no legacy dependencies."""

import json
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from churn_retention.contracts.reference import (
    Digest,
    ReferenceArtifactError,
    json_bytes,
    sha256,
)

SCHEMA_VERSION: Literal["telco-19-v1"] = "telco-19-v1"
PREPROCESSING_VERSION: Literal["rag-sklearn-state-json-v1"] = "rag-sklearn-state-json-v1"
PAYLOAD_LIMITS = {
    "manifest.json": 64 * 1024,
    "schema.json": 16 * 1024,
    "preprocessing/state.json": 1024 * 1024,
    "preprocessing/fit_ids.json": 8 * 1024 * 1024,
    "records.jsonl": 32 * 1024 * 1024,
    "matrix.npy": 32 * 1024 * 1024,
    "splits.csv": 8 * 1024 * 1024,
}
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MATRIX_ELEMENTS = 2_000_000
PAYLOAD_NAMES = frozenset(PAYLOAD_LIMITS) - {"manifest.json"}


class ReferenceV2Error(ReferenceArtifactError):
    """Static error messages deliberately exclude customer data and local paths."""


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class FitProvenance(StrictContract):
    row_count: int = Field(gt=0, le=100_000)
    train_sha256: Digest
    splits_sha256: Digest
    membership_sha256: Digest
    ordered_ids_sha256: Digest
    row_order: Literal["original processed-train order"]
    target_used: Literal[False]


class ReferenceManifestV2(StrictContract):
    artifact_type: Literal["telco-customer-reference"] = "telco-customer-reference"
    format_version: Literal[2] = 2
    reference_version: str = Field(pattern=r"^rag-v2-[a-f0-9]{20}$")
    schema_version: Literal["telco-19-v1"] = SCHEMA_VERSION
    schema_sha256: Digest
    preprocessing_version: Literal["rag-sklearn-state-json-v1"] = PREPROCESSING_VERSION
    preprocessing_sha256: Digest
    source_train_sha256: Digest
    split_sha256: Digest
    row_count: int = Field(gt=0, le=100_000)
    input_feature_count: Literal[19] = 19
    transformed_feature_count: int = Field(gt=0, le=1024)
    output_feature_names: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=1024
    )
    output_feature_names_sha256: Digest
    matrix_dtype: Literal["float64"] = "float64"
    matrix_values_sha256: Digest
    reference_split: Literal["train"] = "train"
    membership_policy: Literal["exact_train_ids"] = "exact_train_ids"
    fit: FitProvenance
    libraries: dict[str, str]
    files: dict[str, Digest]


def runtime_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def reference_version(files: dict[str, str], libraries: dict[str, str]) -> str:
    return (
        "rag-v2-"
        + sha256(json_bytes({"format_version": 2, "files": files, "libraries": libraries}))[:20]
    )


def strict_json(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ReferenceV2Error("Duplicate JSON keys are forbidden.")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ReferenceV2Error("Non-standard JSON constants are forbidden.")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)
