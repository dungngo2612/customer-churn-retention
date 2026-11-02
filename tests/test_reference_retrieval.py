"""Small synthetic fixtures test mechanics; project-data checks live separately."""

import hashlib
import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from churn_retention.adapters.retrieval._artifact import (
    ReferenceArtifactError,
    ReferenceManifest,
    runtime_versions,
    sha256,
)
from churn_retention.adapters.retrieval.legacy_reference_store import ReferenceStore
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.ml.training.bundle import create_feature_schema, save_bundle
from churn_retention.workflows.build_reference import build_reference

ROOT = Path(__file__).resolve().parents[1]
FEATURES = yaml.safe_load((ROOT / "configs/train_config.yaml").read_text())["features"]
CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@dataclass
class Source:
    train: Path
    bundle: Path
    splits: Path
    output: Path
    frame: pd.DataFrame
    preprocessor: ColumnTransformer

    def build(self) -> ReferenceManifest:
        return build_reference(self.train, self.bundle, self.splits, self.output)


def make_source(directory: Path, n: int = 5) -> Source:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append(
            {
                "customerID": f"unit-{n - i:03d}",
                "Churn": "Yes" if i % 2 else "No",
                "SeniorCitizen": i % 2,
                "tenure": i,
                "MonthlyCharges": 20.0 + i,
                "TotalCharges": float(i * (20 + i)),
                **{
                    name: values[i % len(values)] for name, values in CATEGORIES.items()
                },
            }
        )
    frame = pd.DataFrame(rows)
    numeric, categorical = FEATURES["numeric_cols"], FEATURES["categorical_cols"]
    X = frame[numeric + categorical]
    # This new test-only transformer is fitted once to construct a fixture.
    # The classifier is never trained; production must only transform this fixture.
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    ).fit(X)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression())]
    )
    bundle = directory / "logistic-unit-fixture"
    save_bundle(
        pipeline,
        bundle,
        create_feature_schema(X, numeric, categorical),
        {
            "model_version": bundle.name,
            "model_type": "logistic",
            "label": {"target_col": "Churn", "positive_label": "Yes"},
            "config": {"features": FEATURES},
            "libraries": runtime_versions(),
        },
    )
    train, splits = directory / "train.csv", directory / "splits.csv"
    frame.to_csv(train, index=False)
    pd.DataFrame(
        {
            "customerID": frame.customerID.tolist() + ["held-out-val", "held-out-test"],
            "split": ["train"] * n + ["val", "test"],
        }
    ).to_csv(splits, index=False)
    return Source(
        train, bundle, splits, directory / "reference.zip", frame, preprocessor
    )


@pytest.fixture
def source(tmp_path: Path) -> Source:
    return make_source(tmp_path)


@pytest.fixture
def store(source: Source) -> ReferenceStore:
    source.build()
    return ReferenceStore.load(source.output)


def query(source: Source) -> dict[str, object]:
    return {str(name): value for name, value in source.frame.iloc[0].items()}


def rewrite_archive(path: Path, change, *, resign: bool = False) -> None:
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    change(payloads)
    if resign:
        manifest = json.loads(payloads["manifest.json"])
        manifest["files"] = {
            name: sha256(value)
            for name, value in payloads.items()
            if name != "manifest.json"
        }
        payloads["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)


def test_builder_produces_consistent_artifact(source: Source) -> None:
    manifest = source.build()
    store = ReferenceStore.load(
        source.output, expected_reference_version=manifest.reference_version
    )
    assert manifest.row_count == len(source.frame) == len(store.records)
    assert len(manifest.feature_schema) == 19
    assert (
        manifest.dataset_sha256 == hashlib.sha256(source.train.read_bytes()).hexdigest()
    )
    assert manifest.split_sha256 == sha256(source.splits.read_bytes())
    assert manifest.split_counts == {"train": 5, "val": 1, "test": 1}
    assert manifest.reference_split == "train"
    assert store.matrix.shape == (5, len(source.preprocessor.get_feature_names_out()))
    assert store.matrix.dtype == np.float64 and np.isfinite(store.matrix).all()
    assert {r.customer_id: r.churn for r in store.records} == dict(
        zip(source.frame.customerID, source.frame.Churn, strict=True)
    )
    with zipfile.ZipFile(source.output) as archive:
        assert set(archive.namelist()) == set(manifest.files) | {"manifest.json"}
        assert all(
            sha256(archive.read(name)) == digest
            for name, digest in manifest.files.items()
        )


@pytest.mark.parametrize("id_value", ["held-out-val", "held-out-test", "unknown"])
def test_builder_rejects_non_train_ids(source: Source, id_value: str) -> None:
    frame = source.frame.copy()
    frame.loc[0, "customerID"] = id_value
    frame.to_csv(source.train, index=False)
    with pytest.raises(ReferenceArtifactError, match="train split"):
        source.build()
    assert not source.output.exists()


def test_builder_rejects_incomplete_train_membership(source: Source) -> None:
    source.frame.iloc[:-1].to_csv(source.train, index=False)
    with pytest.raises(ReferenceArtifactError, match="exactly"):
        source.build()


@pytest.mark.parametrize("invalid", [None, "", " ", "Maybe", 1])
def test_builder_rejects_invalid_labels(source: Source, invalid: object) -> None:
    frame = source.frame.copy()
    frame["Churn"] = frame.Churn.astype(object)
    frame.loc[0, "Churn"] = cast(Any, invalid)
    frame.to_csv(source.train, index=False)
    with pytest.raises(ReferenceArtifactError, match="labels"):
        source.build()


@pytest.mark.parametrize("kind", ["duplicate", "empty", "missing"])
def test_builder_rejects_invalid_ids(source: Source, kind: str) -> None:
    frame = source.frame.copy()
    frame.loc[0, "customerID"] = (
        frame.loc[1, "customerID"]
        if kind == "duplicate"
        else (" " if kind == "empty" else None)
    )
    frame.to_csv(source.train, index=False)
    with pytest.raises(ValueError):
        source.build()


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_builder_requires_exact_feature_columns(source: Source, change: str) -> None:
    frame = source.frame.copy()
    if change == "missing":
        frame = frame.drop(columns="gender")
    elif change == "extra":
        frame["unwanted"] = "x"
    else:
        frame.columns = ["gender" if c == "Partner" else c for c in frame.columns]
    frame.to_csv(source.train, index=False)
    with pytest.raises(ReferenceArtifactError, match="19 features"):
        source.build()


def test_never_refits_or_mutates_preprocessing_or_classifier(
    source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Builder/store/retriever must not fit preprocessing or classifier")

    for cls in [
        ColumnTransformer,
        SimpleImputer,
        StandardScaler,
        OneHotEncoder,
        LogisticRegression,
    ]:
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    original_files = {p: sha256(p.read_bytes()) for p in source.bundle.iterdir()}
    source.build()
    store = ReferenceStore.load(source.output)
    state = pickle.dumps(store._preprocessor)
    retriever = SimilarCustomerRetriever(store)
    for _ in range(3):
        retriever.retrieve(query(source))
    assert pickle.dumps(store._preprocessor) == state
    assert {
        p: sha256(p.read_bytes()) for p in source.bundle.iterdir()
    } == original_files


def test_query_uses_exact_existing_transform_and_feature_order(
    source: Source, store: ReferenceStore
) -> None:
    row = query(source)
    prepared = store.prepare_query(dict(reversed(list(row.items()))))
    names = FEATURES["numeric_cols"] + FEATURES["categorical_cols"]
    expected = source.preprocessor.transform(pd.DataFrame([row])[names])
    if hasattr(expected, "toarray"):
        expected = expected.toarray()
    np.testing.assert_allclose(prepared.vector, expected)
    assert set(prepared.features) == set(names)
    assert "Churn" not in store._preprocessor.feature_names_in_
    assert "customerID" not in store._preprocessor.feature_names_in_


def test_top_k_matches_independent_euclidean_calculation(
    source: Source, store: ReferenceStore
) -> None:
    row = query(source)
    row.pop("customerID")
    prepared = store.prepare_query(row)
    expected = sorted(
        zip(
            np.linalg.norm(store.matrix - prepared.vector, axis=1),
            store.records,
            strict=True,
        ),
        key=lambda pair: (pair[0], pair[1].customer_id),
    )[:3]
    result = SimilarCustomerRetriever(store, k=3).retrieve(row)
    assert [r.customer_id for r in result] == [r.customer_id for _, r in expected]
    np.testing.assert_allclose(
        [r.distance for r in result], [d for d, _ in expected], atol=1e-7
    )
    assert [r.distance for r in result] == sorted(r.distance for r in result)


def test_query_labels_and_nonmember_ids_have_no_effect_on_distance(
    source: Source, store: ReferenceStore
) -> None:
    retriever = SimilarCustomerRetriever(store)
    first = {**query(source), "customerID": "new-one", "Churn": "Yes"}
    second = {**first, "customerID": "new-two", "Churn": {"ignored": "not a label"}}
    a, b = retriever.retrieve(first), retriever.retrieve(second)
    assert a == b
    np.testing.assert_array_equal(
        store.prepare_query(first).vector, store.prepare_query(second).vector
    )


def test_reference_labels_are_not_encoded_into_matrix(
    source: Source, tmp_path: Path
) -> None:
    source.build()
    original = ReferenceStore.load(source.output)
    frame = source.frame.copy()
    frame["Churn"] = frame.Churn.map({"Yes": "No", "No": "Yes"})
    frame.to_csv(source.train, index=False)
    other_path = tmp_path / "other.zip"
    build_reference(source.train, source.bundle, source.splits, other_path)
    other = ReferenceStore.load(other_path)
    np.testing.assert_array_equal(original.matrix, other.matrix)
    assert [r.churn for r in original.records] != [r.churn for r in other.records]


def test_same_customer_is_excluded_and_large_k_is_clamped(
    source: Source, store: ReferenceStore
) -> None:
    result = SimilarCustomerRetriever(store).retrieve(query(source), k=100)
    assert len(result) == len(store.records) - 1
    assert source.frame.iloc[0].customerID not in [r.customer_id for r in result]
    assert len({r.customer_id for r in result}) == len(result)


def test_explicit_customer_id_alias_and_conflicts(
    source: Source, store: ReferenceStore
) -> None:
    row = query(source)
    own_id = row.pop("customerID")
    assert isinstance(own_id, str)
    retriever = SimilarCustomerRetriever(store)
    assert all(
        r.customer_id != own_id for r in retriever.retrieve(row, customer_id=own_id)
    )
    assert retriever.retrieve({**row, "customer_id": own_id}) == retriever.retrieve(
        row, customer_id=own_id
    )
    with pytest.raises(ValueError, match="Conflicting"):
        retriever.retrieve({**row, "customerID": own_id}, customer_id="different")


def test_only_self_reference_returns_empty(tmp_path: Path) -> None:
    source = make_source(tmp_path, n=1)
    source.build()
    assert (
        SimilarCustomerRetriever(ReferenceStore.load(source.output)).retrieve(
            query(source)
        )
        == []
    )


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "5"])
def test_invalid_k_is_rejected(
    source: Source, store: ReferenceStore, k: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SimilarCustomerRetriever(store, k=cast(int, k))
    with pytest.raises(ValueError, match="positive integer"):
        SimilarCustomerRetriever(store).retrieve(query(source), k=cast(int, k))


@pytest.mark.parametrize("column", ["gender", "tenure", "TotalCharges"])
def test_missing_features_are_rejected(
    source: Source, store: ReferenceStore, column: str
) -> None:
    row = query(source)
    row.pop(column)
    with pytest.raises(ValueError, match="missing"):
        SimilarCustomerRetriever(store).retrieve(row)


@pytest.mark.parametrize("value", ["unknown-category", None, "", " Female ", 1])
def test_invalid_categorical_values_are_rejected(
    source: Source, store: ReferenceStore, value: object
) -> None:
    with pytest.raises(ValueError, match="categorical"):
        store.prepare_query({**query(source), "gender": value})


@pytest.mark.parametrize("value", [None, np.nan, "", " \t"])
def test_missing_total_charges_at_zero_tenure_is_zero(
    source: Source, store: ReferenceStore, value: object
) -> None:
    blank = store.prepare_query({**query(source), "tenure": 0, "TotalCharges": value})
    zero = store.prepare_query({**query(source), "tenure": 0, "TotalCharges": 0.0})
    assert blank.features["TotalCharges"] == 0.0
    np.testing.assert_array_equal(blank.vector, zero.vector)
    with pytest.raises(ValueError, match="tenure is zero"):
        store.prepare_query({**query(source), "tenure": 1, "TotalCharges": value})


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("MonthlyCharges", np.nan),
        ("MonthlyCharges", np.inf),
        ("TotalCharges", -np.inf),
        ("tenure", -1),
        ("tenure", 1.5),
        ("SeniorCitizen", 2),
        ("tenure", True),
        ("MonthlyCharges", "not-numeric"),
    ],
)
def test_invalid_numeric_input_is_rejected(
    source: Source, store: ReferenceStore, column: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        store.prepare_query({**query(source), column: value})


def test_builder_normalizes_only_valid_missing_total_charges(source: Source) -> None:
    frame = source.frame.copy()
    frame.loc[0, "TotalCharges"] = np.nan
    frame.to_csv(source.train, index=False)
    source.build()
    record = next(
        r
        for r in ReferenceStore.load(source.output).records
        if r.customer_id == frame.iloc[0].customerID
    )
    assert record.features["TotalCharges"] == 0.0


def test_no_disk_reads_or_full_train_transforms_per_request(
    source: Source, store: ReferenceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = SimilarCustomerRetriever(store)
    row = query(source)
    shapes = []
    original = store._preprocessor.transform

    def transform_query_only(frame):
        shapes.append(frame.shape)
        assert frame.shape == (1, 19)
        return original(frame)

    monkeypatch.setattr(store._preprocessor, "transform", transform_query_only)
    with (
        patch.object(
            pd, "read_csv", side_effect=AssertionError("No CSV reads at request time")
        ),
        patch.object(
            zipfile, "ZipFile", side_effect=AssertionError("No archive reloads")
        ),
        patch.object(Path, "read_bytes", side_effect=AssertionError("No disk reads")),
    ):
        for _ in range(3):
            retriever.retrieve(row)
    assert shapes == [(1, 19)] * 3


def test_prompt_projection_has_no_identifiers_or_distances(
    source: Source, store: ReferenceStore
) -> None:
    results = SimilarCustomerRetriever(store).retrieve(query(source))
    data = [r.to_prompt_data() for r in results]
    for item in data:
        assert set(item) == {"features", "Churn"}
        features = item["features"]
        assert isinstance(features, dict) and len(features) == 19
    serialized = json.dumps(data)
    assert (
        "customerID" not in serialized
        and "customer_id" not in serialized
        and "distance" not in serialized
    )
    assert all(r.customer_id not in serialized for r in results)


def test_store_data_cannot_be_accidentally_mutated(store: ReferenceStore) -> None:
    with pytest.raises(ValueError):
        store.matrix[0, 0] = 999
    with pytest.raises(TypeError):
        cast(dict, store.records[0].features)["tenure"] = 999
    manifest = store.manifest
    manifest.files.clear()
    assert store.manifest.files


def test_deterministic_bytes_and_results(source: Source, tmp_path: Path) -> None:
    first_manifest = source.build()
    second = tmp_path / "repeat.zip"
    other_manifest = build_reference(source.train, source.bundle, source.splits, second)
    assert first_manifest == other_manifest
    assert source.output.read_bytes() == second.read_bytes()
    first, repeated = ReferenceStore.load(source.output), ReferenceStore.load(second)
    retriever = SimilarCustomerRetriever(first)
    assert retriever.retrieve(query(source)) == retriever.retrieve(query(source))
    assert retriever.retrieve(query(source)) == SimilarCustomerRetriever(
        repeated
    ).retrieve(query(source))


def test_global_ties_have_deterministic_id_order(source: Source) -> None:
    frame = source.frame.copy()
    cols = FEATURES["numeric_cols"] + FEATURES["categorical_cols"]
    frame.loc[:, cols] = frame.loc[0, cols].to_numpy()
    frame.to_csv(source.train, index=False)
    source.build()
    result = SimilarCustomerRetriever(ReferenceStore.load(source.output), k=2).retrieve(
        {**query(source), "customerID": "non-member"}
    )
    assert [r.customer_id for r in result] == sorted(frame.customerID)[:2]
    np.testing.assert_allclose([r.distance for r in result], 0, atol=1e-7)


def test_atomic_failure_removes_staging_and_never_publishes_partial(
    source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        zipfile.ZipFile,
        "writestr",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated write failure")),
    )
    with pytest.raises(OSError, match="simulated"):
        source.build()
    assert not source.output.exists()
    assert not list(source.output.parent.glob(f".{source.output.name}.*.tmp"))


def test_existing_artifact_is_never_overwritten(source: Source) -> None:
    source.build()
    before = source.output.read_bytes()
    with pytest.raises(FileExistsError):
        source.build()
    assert source.output.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    [
        "format_version",
        "libraries",
        "feature_schema",
        "row_count",
        "transformed_feature_count",
    ],
)
def test_incompatible_manifest_is_rejected(source: Source, field: str) -> None:
    source.build()

    def change(payloads):
        manifest = json.loads(payloads["manifest.json"])
        manifest[field] = {
            "format_version": 999,
            "libraries": {"scikit-learn": "0.0"},
            "feature_schema": manifest["feature_schema"][:-1],
            "row_count": 999,
            "transformed_feature_count": 999,
        }[field]
        payloads["manifest.json"] = json.dumps(manifest).encode()

    rewrite_archive(source.output, change)
    with pytest.raises(ReferenceArtifactError):
        ReferenceStore.load(source.output)


def test_checksums_are_checked_before_unpickling(source: Source) -> None:
    source.build()
    rewrite_archive(
        source.output, lambda p: p.__setitem__("bundle/model.pkl", b"corrupt")
    )
    with (
        patch.object(
            pickle,
            "loads",
            side_effect=AssertionError("Do not unpickle corrupt payloads"),
        ),
        pytest.raises(ReferenceArtifactError, match="checksum"),
    ):
        ReferenceStore.load(source.output)


@pytest.mark.parametrize("kind", ["nan", "wrong_shape", "wrong_alignment", "object"])
def test_invalid_matrix_is_rejected_even_with_updated_checksum(
    source: Source, kind: str
) -> None:
    source.build()

    def change(payloads):
        matrix = np.load(io.BytesIO(payloads["matrix.npy"]), allow_pickle=False)
        if kind == "nan":
            matrix[0, 0] = np.nan
        elif kind == "wrong_shape":
            matrix = matrix[:, :-1]
        elif kind == "wrong_alignment":
            matrix = matrix[::-1]
        else:
            matrix = matrix.astype(object)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=kind == "object")
        payloads["matrix.npy"] = buffer.getvalue()

    rewrite_archive(source.output, change, resign=True)
    with pytest.raises(ReferenceArtifactError):
        ReferenceStore.load(source.output)


def test_non_train_record_is_rejected_even_with_updated_checksum(
    source: Source,
) -> None:
    source.build()

    def change(payloads):
        records = json.loads(payloads["records.json"])
        records[0]["customer_id"] = "held-out-val"
        payloads["records.json"] = json.dumps(records).encode()

    rewrite_archive(source.output, change, resign=True)
    with pytest.raises(ReferenceArtifactError, match="train split"):
        ReferenceStore.load(source.output)


@pytest.mark.parametrize(
    "field",
    ["expected_reference_version", "expected_bundle_sha256", "expected_dataset_sha256"],
)
def test_expected_source_and_version_are_enforced(source: Source, field: str) -> None:
    source.build()
    with pytest.raises(ReferenceArtifactError, match="requested"):
        ReferenceStore.load(source.output, **{field: "does-not-match"})


def test_bad_archive_and_unfitted_or_wrong_bundle_are_rejected(source: Source) -> None:
    source.output.write_bytes(b"not a zip archive")
    with pytest.raises(ReferenceArtifactError):
        ReferenceStore.load(source.output)
    source.output.unlink()
    metadata_path = source.bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_type"] = "catboost"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ReferenceArtifactError, match="Logistic"):
        source.build()


def test_unfitted_preprocessing_is_rejected_without_fitting(source: Source) -> None:
    model_path = source.bundle / "model.pkl"
    pipeline = pickle.loads(model_path.read_bytes())
    pipeline.set_params(preprocessor=clone(pipeline.named_steps["preprocessor"]))
    model_path.write_bytes(pickle.dumps(pipeline))
    with pytest.raises(ValueError):
        source.build()
    assert not source.output.exists()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("MonthlyCharges", np.nan),
        ("MonthlyCharges", np.inf),
        ("TotalCharges", -np.inf),
        ("gender", None),
    ],
)
def test_builder_rejects_unclean_values(
    source: Source, column: str, value: Any
) -> None:
    frame = source.frame.copy()
    frame.loc[0, column] = value
    frame.to_csv(source.train, index=False)
    with pytest.raises((ValueError, TypeError)):
        source.build()
    assert not source.output.exists()


def test_noncanonical_features_cannot_be_hidden_in_archive(source: Source) -> None:
    source.build()

    def change(payloads):
        records = json.loads(payloads["records.json"])
        records[0]["features"]["customerID"] = "hidden-id"
        payloads["records.json"] = json.dumps(records).encode()

    rewrite_archive(source.output, change, resign=True)
    with pytest.raises(ReferenceArtifactError, match="canonical"):
        ReferenceStore.load(source.output)


def test_atomic_publish_refuses_a_concurrent_destination(
    source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    link = os.link

    def concurrent_publish(temporary, destination):
        Path(destination).write_bytes(b"other-builder-result")
        link(temporary, destination)

    monkeypatch.setattr(os, "link", concurrent_publish)
    with pytest.raises(FileExistsError):
        source.build()
    assert source.output.read_bytes() == b"other-builder-result"
    assert not list(source.output.parent.glob(f".{source.output.name}.*.tmp"))


def test_query_cleaning_does_not_mutate_input(
    source: Source, store: ReferenceStore
) -> None:
    row = {**query(source), "TotalCharges": None, "tenure": "0"}
    prepared = store.prepare_query(row)
    assert row["TotalCharges"] is None and row["tenure"] == "0"
    assert prepared.features["TotalCharges"] == 0.0 and prepared.features["tenure"] == 0


def test_split_manifest_rejects_duplicate_membership(source: Source) -> None:
    splits = pd.read_csv(source.splits)
    pd.concat([splits, splits.iloc[[0]]]).to_csv(source.splits, index=False)
    with pytest.raises(ReferenceArtifactError, match="unique"):
        source.build()


def test_customer_ids_preserve_leading_zeros(source: Source) -> None:
    frame = source.frame.copy()
    frame["customerID"] = [f"{i:04d}" for i in range(len(frame))]
    frame.to_csv(source.train, index=False)
    pd.DataFrame({"customerID": frame.customerID, "split": "train"}).to_csv(
        source.splits, index=False
    )
    source.build()
    store = ReferenceStore.load(source.output)
    assert [record.customer_id for record in store.records] == sorted(frame.customerID)
