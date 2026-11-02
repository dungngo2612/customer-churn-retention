"""Build and atomically publish a train-only reference archive offline."""

import argparse
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

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
from churn_retention.adapters.retrieval.legacy_reference_store import ReferenceStore
from churn_retention.contracts.reference import (
    FeatureColumn,
    ReferenceArtifactError,
    StoredRecord,
    clean_features,
    json_bytes,
    sha256,
    validate_id,
)


def build_reference(
    train_path: str | Path,
    bundle_path: str | Path,
    splits_path: str | Path,
    output_path: str | Path,
    *,
    reference_version: str | None = None,
) -> ReferenceManifest:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Reference output already exists; use a new artifact path.")
    train_bytes = Path(train_path).read_bytes()
    split_bytes = Path(splits_path).read_bytes()
    bundle_dir = Path(bundle_path)
    bundle = {name: (bundle_dir / name).read_bytes() for name in sorted(BUNDLE_FILES)}
    if (bundle_dir / "effective_config.yaml").is_file():
        bundle["effective_config.yaml"] = (bundle_dir / "effective_config.yaml").read_bytes()
    schema = [
        FeatureColumn.model_validate(value) for value in json.loads(bundle["feature_schema.json"])
    ]
    preprocessor, categories, model_version = load_preprocessor(bundle, schema)
    frame = pd.read_csv(io.BytesIO(train_bytes), dtype={"customerID": "string"})
    expected_columns = {c.name for c in schema} | {"customerID", "Churn"}
    if set(frame.columns) != expected_columns or len(frame.columns) != 21:
        raise ReferenceArtifactError(
            "Train CSV must contain exactly 19 features, customerID, and Churn."
        )
    ids = [validate_id(value) for value in frame.customerID]
    membership, counts = read_splits(split_bytes)
    validate_membership(ids, membership)
    if not frame.Churn.isin(["No", "Yes"]).all():
        raise ReferenceArtifactError("Train labels must be non-missing No/Yes values.")
    frame = frame.sort_values("customerID", kind="stable")
    records = [
        StoredRecord(
            customer_id=validate_id(row["customerID"]),
            features=clean_features(
                {column.name: row[column.name] for column in schema}, schema, categories
            ),
            churn=row["Churn"],
        )
        for row in frame.to_dict("records")
    ]
    matrix = transform(preprocessor, [record.features for record in records], schema)
    matrix_buffer = io.BytesIO()
    np.save(matrix_buffer, matrix, allow_pickle=False)
    payloads = {
        "records.json": json_bytes([record.model_dump() for record in records]),
        "matrix.npy": matrix_buffer.getvalue(),
        "splits.csv": split_bytes,
        **{f"bundle/{name}": value for name, value in bundle.items()},
    }
    dataset_hash, bundle_hash, split_hash = (
        sha256(train_bytes),
        bundle_digest(bundle),
        sha256(split_bytes),
    )
    libraries = runtime_versions()
    if reference_version is None:
        reference_version = (
            "ref-"
            + sha256(
                json_bytes(
                    {
                        "format_version": 1,
                        "dataset": dataset_hash,
                        "bundle": bundle_hash,
                        "split": split_hash,
                        "libraries": libraries,
                    }
                )
            )[:20]
        )
    manifest = ReferenceManifest(
        reference_version=reference_version,
        row_count=len(records),
        transformed_feature_count=matrix.shape[1],
        feature_schema=schema,
        dataset_sha256=dataset_hash,
        dataset_name=Path(train_path).name,
        model_version=model_version,
        bundle_sha256=bundle_hash,
        bundle_files={name: sha256(data) for name, data in bundle.items()},
        split_sha256=split_hash,
        split_counts=counts,
        libraries=libraries,
        files={name: sha256(data) for name, data in payloads.items()},
    )
    payloads["manifest.json"] = json_bytes(manifest.model_dump())

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
                for name, data in sorted(payloads.items()):
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o600 << 16
                    archive.writestr(entry, data)
            handle.flush()
            os.fsync(handle.fileno())
        ReferenceStore.load(temporary, expected_reference_version=manifest.reference_version)
        # Same-filesystem hard link publishes a complete file atomically and
        # refuses an existing destination, including a concurrent builder's file.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only customer reference artifact")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--splits", type=Path, default=Path("artifacts/splits/dataset_splits.csv"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-version")
    args = parser.parse_args()
    manifest = build_reference(
        args.train,
        args.bundle,
        args.splits,
        args.out,
        reference_version=args.reference_version,
    )
    print(
        json.dumps(
            {
                "reference_version": manifest.reference_version,
                "rows": manifest.row_count,
                "dimensions": manifest.transformed_feature_count,
                "artifact_bytes": args.out.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
