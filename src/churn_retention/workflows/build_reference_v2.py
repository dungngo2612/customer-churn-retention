"""Fit and atomically publish an independent, deterministic reference v2 ZIP."""

import argparse
import io
import os
import stat
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from churn_retention.adapters.retrieval._preprocessing_state import dump_preprocessing
from churn_retention.adapters.retrieval._reference_v2_format import (
    MAX_MATRIX_ELEMENTS,
    MAX_TOTAL_BYTES,
    PAYLOAD_LIMITS,
    FitProvenance,
    ReferenceManifestV2,
    ReferenceV2Error,
    reference_version,
)
from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
from churn_retention.adapters.retrieval.reference_v2 import VerificationReferenceV2
from churn_retention.contracts.reference import (
    StoredRecord,
    json_bytes,
    sha256,
    validate_id,
)


def build_reference_v2(
    train_path: str | Path, splits_path: str | Path, output_path: str | Path
) -> ReferenceManifestV2:
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Reference output already exists; select a new path.")
    train_bytes, split_bytes = (
        Path(train_path).read_bytes(),
        Path(splits_path).read_bytes(),
    )
    preprocessor = RAGPreprocessor.fit_train(train_path, splits_path)
    metadata = preprocessor.metadata
    if metadata["fit"]["train_sha256"] != sha256(train_bytes) or metadata["fit"][
        "splits_sha256"
    ] != sha256(split_bytes):
        raise ReferenceV2Error("Source files changed during preprocessing fit.")
    frame = pd.read_csv(io.BytesIO(train_bytes), dtype={"customerID": "string"})
    names = list(preprocessor.input_feature_names)
    if set(frame.columns) != {"customerID", "Churn", *names} or len(frame.columns) != 21:
        raise ReferenceV2Error("Train CSV requires exactly the 19 features, ID and label.")
    if not frame.Churn.isin(["No", "Yes"]).all():
        raise ReferenceV2Error("Reference labels must be non-missing No/Yes values.")
    fit_ids = [validate_id(value) for value in frame.customerID]
    records = [
        StoredRecord(
            customer_id=validate_id(row["customerID"]),
            features=preprocessor.clean({name: row[name] for name in names}),
            churn=row["Churn"],
        )
        for row in frame.sort_values("customerID", kind="stable").to_dict("records")
    ]
    if len(records) * len(preprocessor.output_feature_names) > MAX_MATRIX_ELEMENTS:
        raise ReferenceV2Error("Reference matrix exceeds the element limit.")
    matrix = preprocessor.transform_many([row.features for row in records])
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    payloads = {
        "schema.json": json_bytes([column.model_dump() for column in preprocessor.feature_schema]),
        "preprocessing/state.json": dump_preprocessing(preprocessor),
        "preprocessing/fit_ids.json": json_bytes(fit_ids),
        "records.jsonl": b"".join(json_bytes(row.model_dump()) + b"\n" for row in records),
        "matrix.npy": buffer.getvalue(),
        "splits.csv": split_bytes,
    }
    files = {name: sha256(value) for name, value in payloads.items()}
    output_names = list(preprocessor.output_feature_names)
    manifest = ReferenceManifestV2(
        reference_version=reference_version(files, metadata["libraries"]),
        schema_sha256=files["schema.json"],
        preprocessing_sha256=files["preprocessing/state.json"],
        source_train_sha256=sha256(train_bytes),
        split_sha256=sha256(split_bytes),
        row_count=len(records),
        transformed_feature_count=matrix.shape[1],
        output_feature_names=output_names,
        output_feature_names_sha256=sha256(json_bytes(output_names)),
        matrix_values_sha256=sha256(np.asarray(matrix, dtype="<f8", order="C").tobytes()),
        fit=FitProvenance.model_validate(metadata["fit"]),
        libraries=metadata["libraries"],
        files=files,
    )
    payloads["manifest.json"] = json_bytes(manifest.model_dump())
    if sum(map(len, payloads.values())) > MAX_TOTAL_BYTES or any(
        len(value) > PAYLOAD_LIMITS[name] for name, value in payloads.items()
    ):
        raise ReferenceV2Error("Artifact payloads exceed the bounded format limits.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            with zipfile.ZipFile(
                handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                for name, value in sorted(payloads.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    archive.writestr(info, value)
            handle.flush()
            os.fsync(handle.fileno())
        VerificationReferenceV2.load(temporary, trusted=True)
        os.link(temporary, output)  # Atomic, exclusive publication; never replace.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits/dataset_splits.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_reference_v2(args.train, args.splits, args.out)
    print(
        {
            "reference_version": manifest.reference_version,
            "rows": manifest.row_count,
            "dimensions": manifest.transformed_feature_count,
        }
    )


if __name__ == "__main__":
    main()
