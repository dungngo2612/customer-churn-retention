"""Bounded v2 JSON loader shared by the LLM runtime and offline verification."""

import io
import math
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from churn_retention.adapters.retrieval._preprocessing_state import load_preprocessing
from churn_retention.adapters.retrieval._reference_v2_format import (
    MAX_MATRIX_ELEMENTS,
    MAX_TOTAL_BYTES,
    PAYLOAD_LIMITS,
    PAYLOAD_NAMES,
    ReferenceManifestV2,
    ReferenceV2Error,
    reference_version,
    runtime_versions,
    strict_json,
)
from churn_retention.adapters.retrieval.rag_preprocessor import (
    FEATURE_SCHEMA,
    RAGPreprocessor,
    _train_ids,
)
from churn_retention.contracts.reference import (
    FeatureColumn,
    PreparedQuery,
    ReferenceRecord,
    Scalar,
    StoredRecord,
    json_bytes,
    sha256,
    validate_id,
)


def _read_payloads(data: bytes) -> tuple[ReferenceManifestV2, dict[str, bytes]]:
    if len(data) > MAX_TOTAL_BYTES + 1024 * 1024:
        raise ReferenceV2Error("Archive exceeds the compressed size limit.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(PAYLOAD_LIMITS) or set(names) != set(PAYLOAD_LIMITS):
            raise ReferenceV2Error(
                "Unexpected, missing, duplicate or unsafe ZIP payloads."
            )
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise ReferenceV2Error("Archive exceeds the total expanded size limit.")
        for info in infos:
            mode = info.external_attr >> 16
            if (
                info.file_size < 0
                or info.file_size > PAYLOAD_LIMITS[info.filename]
                or info.flag_bits & 1
                or stat.S_ISLNK(mode)
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or info.file_size > 200 * max(1, info.compress_size)
            ):
                raise ReferenceV2Error(
                    "ZIP payload exceeds limits or uses unsupported attributes."
                )
        manifest = ReferenceManifestV2.model_validate(
            strict_json(archive.read("manifest.json"))
        )
        if (
            set(manifest.files) != PAYLOAD_NAMES
            or manifest.libraries != runtime_versions()
        ):
            raise ReferenceV2Error(
                "Invalid payload inventory or incompatible libraries."
            )
        if (
            manifest.row_count * manifest.transformed_feature_count
            > MAX_MATRIX_ELEMENTS
        ):
            raise ReferenceV2Error("Matrix exceeds the element limit.")
        payloads = {name: archive.read(name) for name in PAYLOAD_NAMES}
    # Verify every payload BEFORE fitted-state restoration or NPY decoding.
    if any(sha256(payloads[name]) != digest for name, digest in manifest.files.items()):
        raise ReferenceV2Error("Artifact payload checksum verification failed.")
    if (
        manifest.reference_version
        != reference_version(manifest.files, manifest.libraries)
        or manifest.schema_sha256 != manifest.files["schema.json"]
        or manifest.preprocessing_sha256 != manifest.files["preprocessing/state.json"]
        or manifest.split_sha256 != manifest.files["splits.csv"]
        or manifest.output_feature_names_sha256
        != sha256(json_bytes(manifest.output_feature_names))
    ):
        raise ReferenceV2Error("Manifest hashes or reference version disagree.")
    return manifest, payloads


def _matrix(data: bytes, manifest: ReferenceManifestV2) -> NDArray[np.float64]:
    buffer = io.BytesIO(data)
    version = np.lib.format.read_magic(buffer)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
            buffer, max_header_size=10_000
        )
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
            buffer, max_header_size=10_000
        )
    else:
        raise ReferenceV2Error("Unsupported NPY header version.")
    if (
        shape != (manifest.row_count, manifest.transformed_feature_count)
        or dtype != np.dtype("float64")
        or fortran
        or len(data) - buffer.tell() != math.prod(shape) * 8
    ):
        raise ReferenceV2Error("Matrix header has invalid shape, dtype or size.")
    result = np.load(io.BytesIO(data), allow_pickle=False, max_header_size=10_000)
    if (
        not np.isfinite(result).all()
        or sha256(np.asarray(result, dtype="<f8", order="C").tobytes())
        != manifest.matrix_values_sha256
    ):
        raise ReferenceV2Error(
            "Matrix has non-finite values or a mismatched value hash."
        )
    return result


class VerificationReferenceV2:
    """Validated v2 implementation, also exported as the public ReferenceStore.

    The historical class name remains for compatibility with offline tools.
    """

    def __init__(
        self,
        manifest: ReferenceManifestV2,
        preprocessor: RAGPreprocessor,
        records: tuple[ReferenceRecord, ...],
        matrix: NDArray[np.float64],
    ) -> None:
        self._manifest = manifest
        self.preprocessor = preprocessor
        self.records = records
        self._matrix = matrix
        self._matrix.setflags(write=False)

    @property
    def matrix(self) -> NDArray[np.float64]:
        result = self._matrix.view()
        result.setflags(write=False)
        return result

    @property
    def manifest(self) -> ReferenceManifestV2:
        return self._manifest.model_copy(deep=True)

    @property
    def reference_version(self) -> str:
        return self._manifest.reference_version

    @property
    def feature_schema(self) -> tuple[FeatureColumn, ...]:
        return self.preprocessor.feature_schema

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_artifact_sha256: str | None = None,
        trusted: bool = False,
        expected_reference_version: str | None = None,
        expected_dataset_sha256: str | None = None,
    ) -> "VerificationReferenceV2":
        try:
            if not trusted and expected_artifact_sha256 is None:
                raise ReferenceV2Error(
                    "An explicitly trusted source or externally pinned artifact hash is required."
                )
            source = Path(path)
            if source.stat().st_size > MAX_TOTAL_BYTES + 1024 * 1024:
                raise ReferenceV2Error("Archive exceeds the compressed size limit.")
            with source.open("rb") as handle:
                archive_bytes = handle.read(MAX_TOTAL_BYTES + 1024 * 1024 + 1)
            if len(archive_bytes) > MAX_TOTAL_BYTES + 1024 * 1024:
                raise ReferenceV2Error("Archive exceeds the compressed size limit.")
            if (
                expected_artifact_sha256 is not None
                and sha256(archive_bytes) != expected_artifact_sha256
            ):
                raise ReferenceV2Error(
                    "Artifact does not match the externally pinned hash."
                )
            manifest, payloads = _read_payloads(archive_bytes)
            if (
                expected_reference_version is not None
                and manifest.reference_version != expected_reference_version
            ) or (
                expected_dataset_sha256 is not None
                and manifest.source_train_sha256 != expected_dataset_sha256
            ):
                raise ReferenceV2Error(
                    "Artifact does not match the requested reference or dataset hash."
                )
            schema = [
                FeatureColumn.model_validate(item)
                for item in strict_json(payloads["schema.json"])
            ]
            if schema != list(FEATURE_SCHEMA):
                raise ReferenceV2Error(
                    "Artifact schema must match the exact RAG 19-feature contract."
                )
            fit_ids = strict_json(payloads["preprocessing/fit_ids.json"])
            train_ids = _train_ids(payloads["splits.csv"])
            if not isinstance(fit_ids, list) or len(fit_ids) != manifest.row_count:
                raise ReferenceV2Error("Invalid fitted ID provenance.")
            fit_ids = [validate_id(value) for value in fit_ids]
            if (
                len(set(fit_ids)) != len(fit_ids)
                or set(fit_ids) != train_ids
                or sha256(json_bytes(fit_ids)) != manifest.fit.ordered_ids_sha256
                or sha256(json_bytes(sorted(fit_ids))) != manifest.fit.membership_sha256
                or manifest.fit.row_count != manifest.row_count
                or manifest.fit.train_sha256 != manifest.source_train_sha256
                or manifest.fit.splits_sha256 != manifest.split_sha256
            ):
                raise ReferenceV2Error("Train-only fit provenance does not agree.")
            preprocessor = load_preprocessing(payloads["preprocessing/state.json"])
            if (
                preprocessor.metadata["fit"] != manifest.fit.model_dump()
                or preprocessor.metadata["libraries"] != manifest.libraries
                or list(preprocessor.output_feature_names)
                != manifest.output_feature_names
                or len(manifest.output_feature_names)
                != manifest.transformed_feature_count
            ):
                raise ReferenceV2Error("Fitted preprocessing and manifest disagree.")
            lines = payloads["records.jsonl"].splitlines()
            if len(lines) != manifest.row_count:
                raise ReferenceV2Error("Reference record count disagrees.")
            stored = [StoredRecord.model_validate(strict_json(line)) for line in lines]
            ids = [validate_id(row.customer_id) for row in stored]
            if ids != sorted(train_ids):
                raise ReferenceV2Error(
                    "Reference records must be sorted and exactly train-only."
                )
            cleaned = [preprocessor.clean(row.features) for row in stored]
            if any(
                row.features != clean
                or set(row.features) != set(preprocessor.input_feature_names)
                for row, clean in zip(stored, cleaned, strict=True)
            ):
                raise ReferenceV2Error(
                    "Reference records must contain canonical features only."
                )
            matrix = _matrix(payloads["matrix.npy"], manifest)
            if not np.array_equal(matrix, preprocessor.transform_many(cleaned)):
                raise ReferenceV2Error(
                    "Matrix, records and restored preprocessing are not aligned."
                )
            records = tuple(
                ReferenceRecord(MappingProxyType(features), row.churn, row.customer_id)
                for row, features in zip(stored, cleaned, strict=True)
            )
            return cls(manifest, preprocessor, records, matrix)
        except ReferenceV2Error:
            raise
        except Exception:  # noqa: BLE001 -- Hide customer-bearing parser/transformer errors.
            raise ReferenceV2Error(
                "Invalid, unreadable or incompatible v2 reference artifact."
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
        features = self.clean_features(query)
        vector = self.preprocessor.transform_one(query)
        vector.setflags(write=False)
        return PreparedQuery(
            MappingProxyType(features), vector, ids[0] if ids else None
        )

    def clean_features(self, query: Mapping[str, object]) -> dict[str, Scalar]:
        return self.preprocessor.clean(query)
