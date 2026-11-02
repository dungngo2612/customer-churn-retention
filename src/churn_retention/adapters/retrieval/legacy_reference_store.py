"""Validate a trusted reference archive once and retain its data in RAM."""

import io
import json
import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from churn_retention.adapters.retrieval._artifact import (
    BUNDLE_FILES,
    ReferenceManifest,
    bundle_digest,
    load_preprocessor,
    read_splits,
    runtime_versions,
    transform,
    validate_membership,
)
from churn_retention.contracts.reference import (
    ChurnLabel,
    FeatureColumn,
    PreparedQuery,
    ReferenceArtifactError,
    ReferenceRecord,
    Scalar,
    StoredRecord,
    clean_features,
    sha256,
    validate_id,
)

__all__ = [
    "ChurnLabel",
    "FeatureColumn",
    "PreparedQuery",
    "ReferenceArtifactError",
    "ReferenceRecord",
    "ReferenceStore",
    "Scalar",
    "StoredRecord",
]


class ReferenceStore:
    def __init__(
        self,
        manifest: ReferenceManifest,
        records: tuple[ReferenceRecord, ...],
        matrix: NDArray[np.float64],
        preprocessor: Any,
        categories: dict[str, tuple[str, ...]],
    ) -> None:
        self._manifest = manifest
        self._records = records
        self._matrix = matrix
        self._matrix.setflags(write=False)
        self._preprocessor = preprocessor
        self._categories = categories

    @property
    def reference_version(self) -> str:
        return self._manifest.reference_version

    @property
    def manifest(self) -> ReferenceManifest:
        return self._manifest.model_copy(deep=True)

    @property
    def records(self) -> tuple[ReferenceRecord, ...]:
        return self._records

    @property
    def matrix(self) -> NDArray[np.float64]:
        view = self._matrix.view()
        view.setflags(write=False)
        return view

    @property
    def feature_schema(self) -> tuple[FeatureColumn, ...]:
        return tuple(self._manifest.feature_schema)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_reference_version: str | None = None,
        expected_bundle_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
    ) -> "ReferenceStore":
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ReferenceArtifactError(
                        "Duplicate archive entries are forbidden."
                    )
                manifest = ReferenceManifest.model_validate_json(
                    archive.read("manifest.json")
                )
                versions = runtime_versions()
                if manifest.libraries.get("scikit-learn") != versions["scikit-learn"]:
                    raise ReferenceArtifactError(
                        "Artifact sklearn version is incompatible."
                    )
                if (
                    manifest.libraries.get("numpy", "").split(".")[0]
                    != versions["numpy"].split(".")[0]
                ):
                    raise ReferenceArtifactError(
                        "Artifact numpy major version is incompatible."
                    )
                for expected, actual in (
                    (expected_reference_version, manifest.reference_version),
                    (expected_bundle_sha256, manifest.bundle_sha256),
                    (expected_dataset_sha256, manifest.dataset_sha256),
                ):
                    if expected is not None and expected != actual:
                        raise ReferenceArtifactError(
                            "Artifact does not match the requested version or source hashes."
                        )
                bundle_names = set(manifest.bundle_files)
                if not BUNDLE_FILES <= bundle_names or bundle_names - BUNDLE_FILES - {
                    "effective_config.yaml"
                }:
                    raise ReferenceArtifactError(
                        "Artifact contains an invalid model bundle inventory."
                    )
                required = {"records.json", "matrix.npy", "splits.csv"} | {
                    f"bundle/{name}" for name in bundle_names
                }
                if set(manifest.files) != required or set(names) != required | {
                    "manifest.json"
                }:
                    raise ReferenceArtifactError(
                        "Artifact file inventory is incomplete or unsupported."
                    )
                payloads = {name: archive.read(name) for name in required}

            if any(
                sha256(payloads[name]) != digest
                for name, digest in manifest.files.items()
            ):
                raise ReferenceArtifactError("Artifact checksum verification failed.")
            bundle = {name: payloads[f"bundle/{name}"] for name in bundle_names}
            if (
                any(
                    sha256(bundle[name]) != digest
                    for name, digest in manifest.bundle_files.items()
                )
                or bundle_digest(bundle) != manifest.bundle_sha256
            ):
                raise ReferenceArtifactError("Model bundle checksums disagree.")
            if sha256(payloads["splits.csv"]) != manifest.split_sha256:
                raise ReferenceArtifactError("Split manifest checksum disagrees.")
            schema = [
                FeatureColumn.model_validate(c)
                for c in json.loads(bundle["feature_schema.json"])
            ]
            if schema != manifest.feature_schema:
                raise ReferenceArtifactError(
                    "Official feature schema and artifact manifest disagree."
                )
            # Payload integrity and runtime compatibility are checked before unpickling.
            preprocessor, categories, model_version = load_preprocessor(bundle, schema)
            if model_version != manifest.model_version:
                raise ReferenceArtifactError("Model versions disagree.")
            stored = [
                StoredRecord.model_validate(value)
                for value in json.loads(payloads["records.json"])
            ]
            if len(stored) != manifest.row_count:
                raise ReferenceArtifactError("Artifact row count disagrees.")
            ids = [validate_id(record.customer_id) for record in stored]
            membership, counts = read_splits(payloads["splits.csv"])
            validate_membership(ids, membership)
            if counts != manifest.split_counts or ids != sorted(ids):
                raise ReferenceArtifactError(
                    "Split counts or canonical row order disagree."
                )
            clean_rows = [
                clean_features(record.features, schema, categories) for record in stored
            ]
            feature_names = {column.name for column in schema}
            if any(
                set(record.features) != feature_names or record.features != clean
                for record, clean in zip(stored, clean_rows, strict=True)
            ):
                raise ReferenceArtifactError(
                    "Stored features must be canonical, cleaned values for exactly the official schema."
                )
            matrix = np.load(io.BytesIO(payloads["matrix.npy"]), allow_pickle=False)
            if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float64:
                raise ReferenceArtifactError(
                    "Expected a float64 feature matrix without object data."
                )
            if (
                matrix.shape != (manifest.row_count, manifest.transformed_feature_count)
                or not np.isfinite(matrix).all()
            ):
                raise ReferenceArtifactError("Matrix shape or values are invalid.")
            # One-time alignment check at load, never repeated during retrieval.
            expected_matrix = transform(preprocessor, clean_rows, schema)
            if not np.array_equal(matrix, expected_matrix):
                raise ReferenceArtifactError(
                    "Matrix rows do not match records and fitted preprocessing."
                )
            records = tuple(
                ReferenceRecord(
                    MappingProxyType(features), record.churn, record.customer_id
                )
                for record, features in zip(stored, clean_rows, strict=True)
            )
            return cls(manifest, records, matrix, preprocessor, categories)
        except ReferenceArtifactError:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            EOFError,
            OverflowError,
            pickle.UnpicklingError,
            ImportError,
            zipfile.BadZipFile,
        ):
            raise ReferenceArtifactError(
                "Invalid, unreadable, or incompatible reference artifact."
            ) from None

    def prepare_query(
        self, query: Mapping[str, object], *, customer_id: str | None = None
    ) -> PreparedQuery:
        if not isinstance(query, Mapping):
            raise TypeError("Query must be a mapping of customer features.")
        ids = [
            validate_id(query[name])
            for name in ("customerID", "customer_id")
            if name in query
        ]
        if customer_id is not None:
            ids.append(validate_id(customer_id))
        if len(set(ids)) > 1:
            raise ValueError("Conflicting query customer IDs.")
        # Churn is deliberately ignored; only official feature names are transformed.
        features = clean_features(
            query, self._manifest.feature_schema, self._categories
        )
        vector = transform(
            self._preprocessor, [features], self._manifest.feature_schema
        )
        vector.setflags(write=False)
        return PreparedQuery(
            MappingProxyType(features), vector, ids[0] if ids else None
        )
