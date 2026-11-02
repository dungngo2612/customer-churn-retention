"""v2 security, deterministic generation, independence and frozen-v1 equivalence."""

import io
import json
import os
import pickle
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from churn_retention.adapters.retrieval import reference_v2
from churn_retention.adapters.retrieval._reference_v2_format import (
    PAYLOAD_NAMES,
    ReferenceV2Error,
    reference_version,
)
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
from churn_retention.adapters.retrieval.reference_v2 import VerificationReferenceV2
from churn_retention.contracts.reference import ReferenceRecord, json_bytes, sha256
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from churn_retention.workflows.build_reference_v2 import build_reference_v2
from scripts.verify_reference_v2 import verify

if TYPE_CHECKING:
    from churn_retention.adapters.retrieval.reference_store import ReferenceStore

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests/fixtures/llm_rag_v1_golden.json").read_text())
CASES = BASELINE["queries"]


def load_trusted(path: str | Path, **kwargs: Any) -> VerificationReferenceV2:
    return VerificationReferenceV2.load(path, trusted=True, **kwargs)


def digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def synthetic_source(directory: Path) -> tuple[Path, Path]:
    categories = BASELINE["preprocessing"]["categorical_vocabulary"]
    rows = [
        {
            "customerID": f"synthetic-{3 - i}",
            "Churn": "Yes" if i % 2 else "No",
            "SeniorCitizen": i % 2,
            "tenure": i,
            "MonthlyCharges": float(20 + i),
            "TotalCharges": float(i * (20 + i)),
            **{name: values[i % len(values)] for name, values in categories.items()},
        }
        for i in range(4)
    ]
    train, splits = directory / "train.csv", directory / "splits.csv"
    pd.DataFrame(rows).to_csv(train, index=False)
    pd.DataFrame(
        {
            "customerID": [row["customerID"] for row in rows] + ["held-val", "held-test"],
            "split": ["train"] * 4 + ["val", "test"],
        }
    ).to_csv(splits, index=False)
    return train, splits


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    directory = tmp_path_factory.mktemp("reference-v2")
    train, splits = synthetic_source(directory)
    output = directory / "reference-v2.zip"
    build_reference_v2(train, splits, output)
    return train, splits, output


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> VerificationReferenceV2:
    output = tmp_path_factory.mktemp("real-v2") / "reference-v2.zip"
    build_reference_v2(
        ROOT / "data/processed/train.csv",
        ROOT / "artifacts/splits/dataset_splits.csv",
        output,
    )
    return load_trusted(output)


def rewrite(source: Path, target: Path, change: Any, *, resign: bool = False) -> None:
    with zipfile.ZipFile(source) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    change(payloads)
    if resign:
        manifest = json.loads(payloads["manifest.json"])
        manifest["files"] = {
            name: sha256(value) for name, value in payloads.items() if name != "manifest.json"
        }
        for field, name in [
            ("schema_sha256", "schema.json"),
            ("preprocessing_sha256", "preprocessing/state.json"),
            ("split_sha256", "splits.csv"),
        ]:
            if name in manifest["files"]:
                manifest[field] = manifest["files"][name]
        manifest["reference_version"] = reference_version(manifest["files"], manifest["libraries"])
        payloads["manifest.json"] = json_bytes(manifest)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)


def test_generation_manifest_checksums_no_bundle_and_deterministic_bytes(
    small: tuple[Path, Path, Path],
) -> None:
    train, splits, output = small
    second = output.with_name("second.zip")
    first = load_trusted(output)
    built = build_reference_v2(train, splits, second)
    assert output.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert built == first.manifest and built.reference_version.startswith("rag-v2-")
    assert not {"model_type", "model_version", "bundle_sha256"} & built.model_dump().keys()
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == PAYLOAD_NAMES | {"manifest.json"}
        assert not any(
            name.endswith((".pkl", ".pickle", ".joblib")) or name.startswith("bundle/")
            for name in archive.namelist()
        )
        assert all(sha256(archive.read(name)) == value for name, value in built.files.items())
    assert np.isfinite(first.matrix).all() and len(first.records) == 4


def test_loader_never_fits_or_unpickles(
    small: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Load must not fit, unpickle or use the old runtime Store")

    for cls in (
        ColumnTransformer,
        SimpleImputer,
        StandardScaler,
        OneHotEncoder,
        Pipeline,
    ):
        for method in ("fit", "fit_transform"):
            monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(RAGPreprocessor, "fit_train", forbidden)
    loaded = load_trusted(small[2])
    query = loaded.records[0].features
    assert loaded.prepare_query(query).vector.shape == (1, 45)


@pytest.mark.parametrize("payload", sorted(PAYLOAD_NAMES))
def test_every_payload_checksum_precedes_state_decoding(
    small: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    target = tmp_path / "corrupt.zip"
    rewrite(
        small[2],
        target,
        lambda data: data.__setitem__(payload, data[payload] + b"corrupt"),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Checksum validation must precede state restoration")

    monkeypatch.setattr(reference_v2, "load_preprocessing", forbidden)
    with pytest.raises(ReferenceV2Error, match="checksum"):
        load_trusted(target)


@pytest.mark.parametrize("name", sorted(PAYLOAD_NAMES))
def test_missing_payloads_are_rejected(
    small: tuple[Path, Path, Path], tmp_path: Path, name: str
) -> None:
    target = tmp_path / "missing.zip"
    rewrite(small[2], target, lambda data: data.pop(name))
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize(
    "field",
    [
        "format_version",
        "schema_version",
        "preprocessing_version",
        "row_count",
        "input_feature_count",
        "transformed_feature_count",
        "reference_split",
        "matrix_dtype",
        "model_version",
    ],
)
def test_invalid_manifest_fields(
    small: tuple[Path, Path, Path], tmp_path: Path, field: str
) -> None:
    target = tmp_path / "manifest.zip"

    def change(data: dict[str, bytes]) -> None:
        value = json.loads(data["manifest.json"])
        value[field] = (
            0
            if field in {"row_count", "input_feature_count", "transformed_feature_count"}
            else "invalid"
        )
        data["manifest.json"] = json_bytes(value)

    rewrite(small[2], target, change)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize("kind", ["missing_feature", "wrong_order", "dtype", "extra_field"])
def test_invalid_schema_with_valid_checksums(
    small: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "schema.zip"

    def change(data: dict[str, bytes]) -> None:
        schema = json.loads(data["schema.json"])
        if kind == "missing_feature":
            schema.pop()
        elif kind == "wrong_order":
            schema.reverse()
        elif kind == "dtype":
            schema[0]["dtype"] = "float64"
        else:
            schema[0]["unexpected"] = 1
        data["schema.json"] = json_bytes(schema)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "nan", "infinity", "alignment", "object", "huge_header"]
)
def test_matrix_validation_before_allocation_or_use(
    small: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "matrix.zip"

    def change(data: dict[str, bytes]) -> None:
        matrix = np.load(io.BytesIO(data["matrix.npy"]), allow_pickle=False)
        buffer = io.BytesIO()
        if kind == "shape":
            matrix = matrix[:, :-1]
        elif kind == "dtype":
            matrix = matrix.astype(np.float32)
        elif kind == "nan":
            matrix[0, 0] = np.nan
        elif kind == "infinity":
            matrix[0, 0] = np.inf
        elif kind == "alignment":
            matrix = matrix[::-1]
        elif kind == "object":
            matrix = matrix.astype(object)
        if kind == "huge_header":
            np.lib.format.write_array_header_1_0(
                buffer, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 45)}
            )
        else:
            np.save(buffer, matrix, allow_pickle=kind == "object")
        data["matrix.npy"] = buffer.getvalue()
        manifest = json.loads(data["manifest.json"])
        manifest["matrix_values_sha256"] = sha256(
            np.asarray(matrix, dtype="<f8", order="C").tobytes()
        )
        data["manifest.json"] = json_bytes(manifest)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize(
    "kind", ["count", "order", "heldout_id", "label", "hidden_id", "feature_value"]
)
def test_reference_records_validate_membership_and_alignment(
    small: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "records.zip"

    def change(data: dict[str, bytes]) -> None:
        rows = [json.loads(line) for line in data["records.jsonl"].splitlines()]
        if kind == "count":
            rows.pop()
        elif kind == "order":
            rows.reverse()
        elif kind == "heldout_id":
            rows[0]["customer_id"] = "held-val"
        elif kind == "label":
            rows[0]["churn"] = "invalid"
        elif kind == "hidden_id":
            rows[0]["features"]["customerID"] = "synthetic-hidden"
        else:
            rows[0]["features"]["MonthlyCharges"] += 1
        data["records.jsonl"] = b"".join(json_bytes(row) + b"\n" for row in rows)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize(
    "kind",
    [
        "class",
        "unknown_field",
        "huge_array",
        "code_bytes",
        "wrong_statistics",
        "duplicate_json",
    ],
)
def test_state_codec_has_no_arbitrary_code_or_object_loading(
    small: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "state.zip"

    def change(data: dict[str, bytes]) -> None:
        state = json.loads(data["preprocessing/state.json"])
        if kind == "class":
            state["transformer"]["class"] = "os.system"
        elif kind == "unknown_field":
            state["transformer"]["state"]["items"]["execute"] = "untrusted"
        elif kind == "huge_array":
            state["transformer"] = {
                "kind": "array",
                "dtype": "<f8",
                "shape": [10**12],
                "items": {"kind": "list", "items": []},
            }
        elif kind == "wrong_statistics":
            state["metadata"]["preprocessing"]["steps"]["num.scaler"]["state"]["mean_"][0] += 1
        if kind == "code_bytes":
            data["preprocessing/state.json"] = b"cos\nsystem\n(S'never-execute'\ntR."
        elif kind == "duplicate_json":
            data["preprocessing/state.json"] = b'{"encoding_version":1,"encoding_version":2}'
        else:
            data["preprocessing/state.json"] = json_bytes(state)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


@pytest.mark.parametrize(
    "extra",
    [
        "../escape",
        "/absolute",
        "preprocessing/../../escape",
        "bundle/model.pkl",
        "unexpected",
        "back\\slash",
    ],
)
def test_extra_payloads_and_zip_path_traversal_are_rejected(
    small: tuple[Path, Path, Path], tmp_path: Path, extra: str
) -> None:
    target = tmp_path / "unsafe.zip"
    rewrite(small[2], target, lambda data: data.__setitem__(extra, b"untrusted"))
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)
    assert not (tmp_path / "escape").exists()


def test_zip_duplicate_symlink_limits_and_compression_bomb(
    small: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "duplicate.zip"
    duplicate.write_bytes(small[2].read_bytes())
    with zipfile.ZipFile(duplicate, "a") as archive, pytest.warns(UserWarning):
        archive.writestr("schema.json", b"[]")
    with pytest.raises(ReferenceV2Error):
        load_trusted(duplicate)
    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(small[2]) as source, zipfile.ZipFile(symlink, "w") as target:
        for original in source.infolist():
            info = zipfile.ZipInfo(original.filename)
            info.external_attr = (
                (stat.S_IFLNK | 0o600) << 16
                if original.filename == "schema.json"
                else (stat.S_IFREG | 0o600) << 16
            )
            target.writestr(info, source.read(original.filename))
    with pytest.raises(ReferenceV2Error):
        load_trusted(symlink)
    bomb = tmp_path / "bomb.zip"
    rewrite(small[2], bomb, lambda data: data.__setitem__("records.jsonl", b"0" * 200_000))
    with pytest.raises(ReferenceV2Error):
        load_trusted(bomb)
    monkeypatch.setitem(reference_v2.PAYLOAD_LIMITS, "records.jsonl", 1)
    with pytest.raises(ReferenceV2Error):
        load_trusted(small[2])


def test_corrupt_archive_pin_and_libraries(small: tuple[Path, Path, Path], tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a ZIP")
    with pytest.raises(ReferenceV2Error):
        load_trusted(corrupt)
    with pytest.raises(ReferenceV2Error, match="pinned"):
        load_trusted(small[2], expected_artifact_sha256="0" * 64)
    assert load_trusted(small[2], expected_artifact_sha256=sha256(small[2].read_bytes()))
    incompatible = tmp_path / "libraries.zip"

    def change(data: dict[str, bytes]) -> None:
        value = json.loads(data["manifest.json"])
        value["libraries"]["scikit-learn"] = "unsupported"
        data["manifest.json"] = json_bytes(value)

    rewrite(small[2], incompatible, change)
    with pytest.raises(ReferenceV2Error, match="libraries"):
        load_trusted(incompatible)


@pytest.mark.parametrize("kind", ["val", "test", "duplicate", "incomplete", "invalid_label"])
def test_builder_train_only_and_labels(tmp_path: Path, kind: str) -> None:
    train, splits = synthetic_source(tmp_path)
    frame = pd.read_csv(train)
    if kind in {"val", "test"}:
        frame.loc[0, "customerID"] = "held-" + kind
    elif kind == "duplicate":
        frame.loc[0, "customerID"] = frame.loc[1, "customerID"]
    elif kind == "incomplete":
        frame = frame.iloc[:-1]
    else:
        frame.loc[0, "Churn"] = "invalid"
    frame.to_csv(train, index=False)
    output = tmp_path / "ref.zip"
    with pytest.raises(ValueError):
        build_reference_v2(train, splits, output)
    assert not output.exists()


def test_no_overwrite_and_atomic_publish_race(
    small: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    train, splits, output = small
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_reference_v2(train, splits, output)
    assert output.read_bytes() == before
    other = output.with_name("racing.zip")
    link = os.link

    def race(temporary: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"other publisher")
        link(temporary, destination)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        build_reference_v2(train, splits, other)
    assert other.read_bytes() == b"other publisher"
    assert not list(other.parent.glob(f".{other.name}.*.tmp"))


def test_real_records_matrix_and_preprocessing_match_v1(
    real: VerificationReferenceV2,
    frozen_rag_oracle: Path,
) -> None:
    with zipfile.ZipFile(frozen_rag_oracle) as archive:
        old = json.loads(archive.read("records.json"))
        matrix = np.load(io.BytesIO(archive.read("matrix.npy")), allow_pickle=False)
    actual = [
        {
            "customer_id": row.customer_id,
            "features": dict(row.features),
            "churn": row.churn,
        }
        for row in real.records
    ]
    if actual != old:
        pytest.fail("Internal reference features, labels, IDs or order differ")
    assert len(real.records) == 4930 and real.matrix.shape == (4930, 45)
    assert real.matrix.dtype == np.float64 and np.array_equal(real.matrix, matrix)
    assert real.manifest.matrix_values_sha256 == BASELINE["matrix"]["values_sha256"]
    assert real.preprocessor.metadata["preprocessing"] == BASELINE["preprocessing"]
    assert real.reference_version != BASELINE["source"]["reference_version"]
    with pytest.raises(ValueError):
        real.matrix[0, 0] = 1


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_loaded_vectors_retrieval_and_prompt_match_golden_without_refit(
    real: VerificationReferenceV2, case: dict[str, Any]
) -> None:
    names = list(real.preprocessor.input_feature_names)
    frame = pd.read_csv(ROOT / "data/processed/val.csv", usecols=names)
    query = {name: frame.iloc[case["row"]][name] for name in names}
    prepared = real.prepare_query(query)
    assert prepared.vector.dtype == np.float64 and prepared.vector.shape == (1, 45)
    assert (
        sha256(np.asarray(prepared.vector, dtype="<f8", order="C").tobytes())
        == case["vector_sha256"]
    )
    neighbors = SimilarCustomerRetriever(cast("ReferenceStore", real)).retrieve(query)
    assert digest([row.customer_id for row in neighbors]) == case["ordered_neighbor_ids_sha256"]
    np.testing.assert_allclose(
        [row.distance for row in neighbors], case["distances"], atol=1e-12, rtol=1e-12
    )
    prompt = ChurnPromptBuilder(real.feature_schema).build_with_metadata(
        prepared.features, neighbors
    )
    assert prompt.reference_count == 5 and sha256(prompt.text.encode()) == case["prompt"]["sha256"]
    if any(row.customer_id in prompt.text for row in real.records):
        pytest.fail("An ID entered a prompt")


@pytest.mark.parametrize("exclude", [False, True])
def test_loaded_preprocessing_keeps_k_boundary_ties_and_self_exclusion(
    real: VerificationReferenceV2, exclude: bool
) -> None:
    features = real.records[0].features
    records = tuple(
        ReferenceRecord(features, "No", "synthetic-" + suffix)
        for suffix in ("f", "b", "e", "a", "d", "c")
    )
    view = VerificationReferenceV2(
        real.manifest,
        real.preprocessor,
        records,
        real.preprocessor.transform_many([features] * 6),
    )
    rows = SimilarCustomerRetriever(cast("ReferenceStore", view)).retrieve(
        features, customer_id="synthetic-c" if exclude else None
    )
    suffixes = ("a", "b", "d", "e", "f") if exclude else ("a", "b", "c", "d", "e")
    assert [row.customer_id for row in rows] == ["synthetic-" + suffix for suffix in suffixes]
    assert len({row.distance for row in rows}) == 1


def test_independent_build_load_restore_retrieve_and_prompt(tmp_path: Path) -> None:
    script = (
        "\n"
        "import hashlib, importlib.abc, json, os, socket, sys, types\n"
        "sys.dont_write_bytecode = True\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "blocked = (\n"
        ' "sklearn.linear_model._logistic", "churn_retention.ml",\n'
        ' "churn_retention.adapters.retrieval._artifact", "churn_rete'
        'ntion.adapters.retrieval.reference_store",\n'
        ' "churn_retention.adapters.predictor", "churn_retention.work'
        'flows.train_model",\n'
        ' "churn_retention.workflows.build_reference",\n'
        ")\n"
        "class Barrier(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        '  if any(fullname == p or fullname.startswith(p+".") for p i'
        "n blocked):\n"
        '   raise AssertionError("Legacy/classifier imports are unava'
        'ilable")\n'
        "class NoClassifier:\n"
        ' def __new__(cls,*args,**kwargs): raise AssertionError("Clas'
        'sifier unavailable")\n'
        'poison = types.ModuleType("sklearn.linear_model._logistic")\n'
        "poison.LogisticRegression = poison.LogisticRegressionCV = No"
        "Classifier\n"
        "sys.modules[poison.__name__] = poison\n"
        "sys.meta_path.insert(0, Barrier())\n"
        "def audit(event,args):\n"
        ' if event == "open" and isinstance(args[0],(str,bytes,os.Pat'
        "hLike)):\n"
        '  p = os.fsdecode(args[0]).replace("\\\\","/")\n'
        '  if p.endswith(".pkl") or "/artifacts/bundles/" in p or p.e'
        'ndswith("telco-train-v1.zip"):\n'
        '   raise AssertionError("Old bundles and v1 reference artifa'
        'ct are unavailable")\n'
        "sys.addaudithook(audit)\n"
        'def forbidden(*args,**kwargs): raise AssertionError("No fitt'
        'ing/network during v2 load")\n'
        "socket.socket.connect = socket.socket.connect_ex = socket.cr"
        "eate_connection = forbidden\n"
        "from churn_retention.workflows.build_reference_v2 import bui"
        "ld_reference_v2\n"
        "from churn_retention.adapters.retrieval.reference_v2 import "
        "VerificationReferenceV2\n"
        "from churn_retention.adapters.retrieval.nearest_neighbors im"
        "port SimilarCustomerRetriever\n"
        "from churn_retention.modules.scoring.llm_prompt import Churn"
        "PromptBuilder\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import StandardScaler,OneHotEncod"
        "er\n"
        "import numpy as np, pandas as pd\n"
        "out = sys.argv[6]\n"
        "build_reference_v2(sys.argv[2],sys.argv[3],out)\n"
        "for cls in (ColumnTransformer, Pipeline, SimpleImputer, Stan"
        "dardScaler, OneHotEncoder):\n"
        " cls.fit = cls.fit_transform = forbidden\n"
        "s = VerificationReferenceV2.load(out, trusted=True)\n"
        "b = json.load(open(sys.argv[5]))\n"
        "assert s.matrix.shape == (4930,45) and s.matrix.dtype == np."
        "float64\n"
        'assert hashlib.sha256(s.matrix.astype("<f8").tobytes()).hexd'
        'igest() == b["matrix"]["values_sha256"]\n'
        "frame = pd.read_csv(sys.argv[4],usecols=list(s.preprocessor."
        "input_feature_names))\n"
        "r = SimilarCustomerRetriever(s)\n"
        "builder = ChurnPromptBuilder(s.feature_schema)\n"
        'for case in b["queries"]:\n'
        ' query = {name:frame.iloc[case["row"]][name] for name in s.p'
        "reprocessor.input_feature_names}\n"
        " vector = s.prepare_query(query).vector\n"
        ' assert hashlib.sha256(vector.astype("<f8").tobytes()).hexdi'
        'gest() == case["vector_sha256"]\n'
        " neighbors = r.retrieve(query)\n"
        " ids = json.dumps([n.customer_id for n in neighbors],sort_ke"
        'ys=True,separators=(",",":"))\n'
        ' assert hashlib.sha256(ids.encode()).hexdigest() == case["or'
        'dered_neighbor_ids_sha256"]\n'
        " prompt = builder.build(s.prepare_query(query).features,neig"
        "hbors)\n"
        " assert hashlib.sha256(prompt.encode()).hexdigest() == case["
        '"prompt"]["sha256"]\n'
        "assert sys.modules[poison.__name__] is poison and not hasatt"
        'r(poison,"__file__")\n'
        'assert not any(name == p or name.startswith(p+".") for name '
        "in sys.modules for p in blocked[1:])\n"
        'print("independent v2 build/load/query/retrieval/prompt pass'
        'ed")\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(ROOT / "src"),
            str(ROOT / "data/processed/train.csv"),
            str(ROOT / "artifacts/splits/dataset_splits.csv"),
            str(ROOT / "data/processed/val.csv"),
            str(ROOT / "tests/fixtures/llm_rag_v1_golden.json"),
            str(tmp_path / "independent-v2.zip"),
        ],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "independent v2 build/load/query/retrieval/prompt passed"


def test_comparison_tool_uses_loaded_v2(
    real: VerificationReferenceV2, tmp_path: Path, frozen_rag_oracle: Path
) -> None:
    output = tmp_path / "compare-v2.zip"
    build_reference_v2(
        ROOT / "data/processed/train.csv",
        ROOT / "artifacts/splits/dataset_splits.csv",
        output,
    )
    report = verify(output)
    assert report["equivalent"] and report["matrix_max_absolute_error"] == 0.0
    assert len(report["queries"]) == 8


def test_loader_requires_explicit_source_trust_or_external_pin(
    small: tuple[Path, Path, Path],
) -> None:
    with pytest.raises(ReferenceV2Error, match="trusted"):
        VerificationReferenceV2.load(small[2])
    assert VerificationReferenceV2.load(
        small[2], expected_artifact_sha256=sha256(small[2].read_bytes())
    )


@pytest.mark.parametrize("kind", ["order", "heldout", "duplicate", "target", "membership_hash"])
def test_fit_provenance_must_agree_with_train_membership(
    small: tuple[Path, Path, Path], tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "provenance.zip"

    def change(data: dict[str, bytes]) -> None:
        ids = json.loads(data["preprocessing/fit_ids.json"])
        if kind == "order":
            ids.reverse()
        elif kind == "heldout":
            ids[0] = "held-val"
        elif kind == "duplicate":
            ids[0] = ids[1]
        if kind in {"order", "heldout", "duplicate"}:
            data["preprocessing/fit_ids.json"] = json_bytes(ids)
        else:
            manifest = json.loads(data["manifest.json"])
            manifest["fit"]["target_used" if kind == "target" else "membership_sha256"] = (
                True if kind == "target" else "0" * 64
            )
            data["manifest.json"] = json_bytes(manifest)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error):
        load_trusted(target)


def test_validation_vocabulary_must_equal_encoder_categories(
    small: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    target = tmp_path / "vocabulary.zip"

    def change(data: dict[str, bytes]) -> None:
        state = json.loads(data["preprocessing/state.json"])
        state["metadata"]["preprocessing"]["categorical_vocabulary"]["gender"].append("__UNSEEN__")
        data["preprocessing/state.json"] = json_bytes(state)

    rewrite(small[2], target, change, resign=True)
    with pytest.raises(ReferenceV2Error, match="vocabulary"):
        load_trusted(target)
