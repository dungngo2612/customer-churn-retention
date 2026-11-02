"""Legacy v1 artifact validation and bundle-backed preprocessing; no fitting."""

import io
import json
import pickle
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import Field
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from churn_retention.contracts.reference import (
    NUMERIC_NAMES,
    QUERY_METADATA,
    ChurnLabel,
    Digest,
    FeatureColumn,
    ReferenceArtifactError,
    Scalar,
    StoredRecord,
    _StrictModel,
    clean_features,
    json_bytes,
    sha256,
    validate_id,
    validate_schema,
)

# Compatibility exports keep existing v1 import paths and class identities.
__all__ = [
    "BUNDLE_FILES",
    "NUMERIC_NAMES",
    "QUERY_METADATA",
    "ChurnLabel",
    "Digest",
    "FeatureColumn",
    "ReferenceArtifactError",
    "ReferenceManifest",
    "Scalar",
    "StoredRecord",
    "bundle_digest",
    "clean_features",
    "json_bytes",
    "load_preprocessor",
    "read_splits",
    "runtime_versions",
    "sha256",
    "transform",
    "validate_id",
    "validate_membership",
    "validate_schema",
]

BUNDLE_FILES = frozenset({"model.pkl", "metadata.json", "feature_schema.json"})


class ReferenceManifest(_StrictModel):
    artifact_type: Literal["telco-customer-reference"] = "telco-customer-reference"
    format_version: Literal[1] = 1
    reference_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    row_count: int = Field(gt=0)
    transformed_feature_count: int = Field(gt=0)
    matrix_dtype: Literal["float64"] = "float64"
    feature_schema: list[FeatureColumn]
    dataset_sha256: Digest
    dataset_name: str
    model_version: str = Field(min_length=1)
    bundle_sha256: Digest
    bundle_files: dict[str, Digest]
    split_sha256: Digest
    split_counts: dict[Literal["train", "val", "test"], int]
    reference_split: Literal["train"] = "train"
    membership_policy: Literal["exact_train_ids"] = "exact_train_ids"
    libraries: dict[str, str]
    files: dict[str, Digest]


def runtime_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def bundle_digest(files: Mapping[str, bytes]) -> str:
    return sha256(json_bytes({name: sha256(data) for name, data in files.items()}))


def load_preprocessor(
    bundle_files: Mapping[str, bytes], schema: list[FeatureColumn]
) -> tuple[Any, dict[str, tuple[str, ...]], str]:
    """Load a trusted local sklearn pickle, checking the exact fitted feature path."""
    validate_schema(schema)
    metadata = json.loads(bundle_files["metadata.json"])
    if metadata.get("model_type") != "logistic" or not isinstance(
        metadata.get("model_version"), str
    ):
        raise ReferenceArtifactError("A Logistic Regression bundle with model_version is required.")
    if metadata.get("label") != {"target_col": "Churn", "positive_label": "Yes"}:
        raise ReferenceArtifactError("Bundle label mapping must be Churn / Yes.")
    configured = metadata.get("config", {}).get("features", {})
    numeric = [c.name for c in schema if c.kind == "numeric"]
    categorical = [c.name for c in schema if c.kind == "categorical"]
    if (
        configured.get("numeric_cols") != numeric
        or configured.get("categorical_cols") != categorical
    ):
        raise ReferenceArtifactError("Bundle configuration and feature schema disagree.")
    recorded_version = metadata.get("libraries", {}).get("scikit-learn")
    if recorded_version is not None and recorded_version != sklearn.__version__:
        raise ReferenceArtifactError("Source sklearn bundle version does not match this runtime.")

    pipeline = pickle.loads(bundle_files["model.pkl"])
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != [
        "preprocessor",
        "model",
    ]:
        raise ReferenceArtifactError("Expected a preprocessor/model pipeline.")
    if not isinstance(pipeline.named_steps["model"], LogisticRegression):
        raise ReferenceArtifactError("Bundle classifier must be LogisticRegression.")
    preprocessor = pipeline.named_steps["preprocessor"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise ReferenceArtifactError("Expected a fitted ColumnTransformer.")
    check_is_fitted(preprocessor)
    if list(preprocessor.feature_names_in_) != [c.name for c in schema]:
        raise ReferenceArtifactError("Fitted preprocessor feature names do not match the schema.")
    configured_transformers = [(name, list(cols)) for name, _, cols in preprocessor.transformers_]
    if configured_transformers != [("num", numeric), ("cat", categorical)]:
        raise ReferenceArtifactError("Fitted transformer inputs do not match the feature schema.")
    num = preprocessor.named_transformers_["num"]
    cat = preprocessor.named_transformers_["cat"]
    if not isinstance(num, Pipeline) or not isinstance(cat, Pipeline):
        raise ReferenceArtifactError("Expected numerical and categorical pipelines.")
    if list(num.named_steps) != ["imputer", "scaler"] or list(cat.named_steps) != [
        "imputer",
        "onehot",
    ]:
        raise ReferenceArtifactError("Unsupported preprocessing steps.")
    if not isinstance(num.named_steps["imputer"], SimpleImputer) or not isinstance(
        num.named_steps["scaler"], StandardScaler
    ):
        raise ReferenceArtifactError("Expected numerical imputation and scaling.")
    encoder = cat.named_steps["onehot"]
    if not isinstance(encoder, OneHotEncoder):
        raise ReferenceArtifactError("Expected fitted categorical OneHotEncoder.")
    check_is_fitted(encoder)
    categories = {
        name: tuple(str(v) for v in values)
        for name, values in zip(categorical, encoder.categories_, strict=True)
    }
    return preprocessor, categories, metadata["model_version"]


def read_splits(
    data: bytes,
) -> tuple[dict[str, str], dict[Literal["train", "val", "test"], int]]:
    frame = pd.read_csv(io.BytesIO(data), dtype={"customerID": "string", "split": "string"})
    if set(frame.columns) != {"customerID", "split"} or len(frame.columns) != 2:
        raise ReferenceArtifactError("Split manifest requires customerID and split columns.")
    ids = [validate_id(value) for value in frame.customerID]
    if len(set(ids)) != len(ids) or not frame.split.isin(["train", "val", "test"]).all():
        raise ReferenceArtifactError("Split membership must be unique and valid.")
    membership = dict(zip(ids, frame.split.tolist(), strict=True))
    counts: dict[Literal["train", "val", "test"], int] = {
        "train": int(frame.split.eq("train").sum()),
        "val": int(frame.split.eq("val").sum()),
        "test": int(frame.split.eq("test").sum()),
    }
    if counts["train"] == 0:
        raise ReferenceArtifactError("Split manifest contains no train IDs.")
    return membership, counts


def validate_membership(ids: list[str], membership: Mapping[str, str]) -> None:
    if len(ids) != len(set(ids)):
        raise ReferenceArtifactError("Reference customer IDs must be unique.")
    train_ids = {key for key, value in membership.items() if value == "train"}
    if set(ids) != train_ids:
        raise ReferenceArtifactError(
            (
                "Reference IDs must match exactly the train split; val/test a"
                "nd unknown IDs are forbidden."
            )
        )


def transform(
    preprocessor: Any, rows: list[dict[str, Scalar]], schema: list[FeatureColumn]
) -> NDArray[np.float64]:
    frame = pd.DataFrame(rows, columns=[column.name for column in schema])
    matrix = preprocessor.transform(frame)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    result = np.asarray(matrix, dtype=np.float64, order="C")
    if result.ndim != 2 or len(result) != len(rows) or not np.isfinite(result).all():
        raise ReferenceArtifactError("Transformed features must form a finite 2D matrix.")
    if result.shape[1] != len(preprocessor.get_feature_names_out()):
        raise ReferenceArtifactError(
            "Transformed feature dimensions disagree with the preprocessor."
        )
    return result
