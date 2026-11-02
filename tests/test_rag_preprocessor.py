"""Actual values use fresh train-only RAG fitting; expected values stay v1."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn_retention.adapters.retrieval.rag_preprocessor import RAGPreprocessor
from churn_retention.contracts.reference import ReferenceRecord, sha256
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from scripts.verify_rag_preprocessor import (
    GOLDEN,
    PRIVATE,
    ROOT,
    SPLITS,
    TRAIN,
    CandidateReferenceView,
    array_digest,
    candidate_view,
    json_digest,
    make_retriever,
    validation_queries,
    verify,
)

BASELINE: dict[str, Any] = json.loads(GOLDEN.read_text())
CASES: list[dict[str, Any]] = BASELINE["queries"]


@pytest.fixture(scope="module")
def preprocessor() -> RAGPreprocessor:
    if not TRAIN.is_file() or not SPLITS.is_file():
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail("Real train and split files required for RAG equivalence")
        pytest.skip("Real train and splits are absent; no substitute project data")
    return RAGPreprocessor.fit_train(TRAIN, SPLITS)


@pytest.fixture(scope="module")
def queries(preprocessor: RAGPreprocessor) -> dict[str, dict[str, object]]:
    return validation_queries(preprocessor, CASES)


@pytest.fixture(scope="module")
def candidate(
    preprocessor: RAGPreprocessor, frozen_rag_oracle: Path
) -> tuple[CandidateReferenceView, Any]:
    return candidate_view(preprocessor)


def test_fit_and_transform_in_isolated_process_without_classifier_or_bundle(
    tmp_path: Path,
) -> None:
    script = (
        "\n"
        "import hashlib\n"
        "import importlib.abc\n"
        "import json\n"
        "import os\n"
        "import socket\n"
        "import sys\n"
        "import types\n"
        "\n"
        "sys.dont_write_bytecode = True\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "blocked = (\n"
        '    "sklearn.linear_model._logistic", "churn_retention.ml", '
        '"churn_retention.workflows",\n'
        '    "churn_retention.adapters.retrieval._artifact",\n'
        '    "churn_retention.adapters.retrieval.reference_store",\n'
        '    "churn_retention.adapters.retrieval.nearest_neighbors",\n'
        '    "churn_retention.adapters.predictor",\n'
        ")\n"
        "class ImportBarrier(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if any(fullname == name or fullname.startswith(name "
        '+ ".") for name in blocked):\n'
        '            raise AssertionError("New preprocessing must not'
        ' import classifier/legacy code")\n'
        "\n"
        "def no_models(event, args):\n"
        '    if event == "open" and isinstance(args[0], (str, bytes, '
        "os.PathLike)):\n"
        '        path = os.fsdecode(args[0]).replace("\\\\", "/")\n'
        '        if path.endswith("model.pkl") or "/artifacts/bundles'
        '/" in path or "/artifacts/references/" in path:\n'
        '            raise AssertionError("Model bundles and referenc'
        'e artifacts are unavailable")\n'
        "\n"
        "def no_network(*args, **kwargs):\n"
        '    raise AssertionError("No network in RAG equivalence test'
        's")\n'
        "\n"
        "# sklearn.compose also exports a target regressor importing "
        "LinearRegression.\n"
        "# Allow that unrelated public namespace, but replace only lo"
        "gistic exports with\n"
        "# poison symbols. Its actual implementation cannot be import"
        "ed or instantiated.\n"
        "class UnavailableLogistic:\n"
        "    def __new__(cls, *args, **kwargs):\n"
        '        raise AssertionError("Logistic Regression is unavail'
        'able")\n'
        "\n"
        'poison = types.ModuleType("sklearn.linear_model._logistic")\n'
        "poison.LogisticRegression = UnavailableLogistic\n"
        "poison.LogisticRegressionCV = UnavailableLogistic\n"
        "sys.modules[poison.__name__] = poison\n"
        "sys.meta_path.insert(0, ImportBarrier())\n"
        "sys.addaudithook(no_models)\n"
        "socket.socket.connect = no_network\n"
        "socket.socket.connect_ex = no_network\n"
        "socket.create_connection = no_network\n"
        "from churn_retention.adapters.retrieval.rag_preprocessor imp"
        "ort RAGPreprocessor\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "p = RAGPreprocessor.fit_train(sys.argv[2], sys.argv[3])\n"
        "b = json.load(open(sys.argv[5]))\n"
        "names = list(p.input_feature_names)\n"
        'train = pd.read_csv(sys.argv[2], usecols=["customerID", *nam'
        'es]).sort_values("customerID", kind="stable")\n'
        'actual = p.transform_many(train[names].to_dict("records"))\n'
        "assert actual.shape == (4930, 45) and actual.dtype == np.flo"
        "at64\n"
        'assert hashlib.sha256(actual.astype("<f8").tobytes()).hexdig'
        'est() == b["matrix"]["values_sha256"]\n'
        "validation = pd.read_csv(sys.argv[4], usecols=names)\n"
        'for case in b["queries"]:\n'
        '    vector = p.transform_one({name: validation.iloc[case["ro'
        'w"]][name] for name in names})\n'
        '    assert hashlib.sha256(vector.astype("<f8").tobytes()).he'
        'xdigest() == case["vector_sha256"]\n'
        'assert sys.modules["sklearn.linear_model._logistic"] is pois'
        "on\n"
        'assert not hasattr(poison, "__file__")\n'
        'assert not any(name == prefix or name.startswith(prefix + ".'
        '") for name in sys.modules for prefix in blocked[1:])\n'
        'print("classifier-free fit, matrix, and eight query vectors '
        'passed")\n'
    )
    if not TRAIN.is_file():
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail("Real train file is required")
        pytest.skip("Real train file is absent")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(ROOT / "src"),
            str(TRAIN),
            str(SPLITS),
            str(ROOT / "data/processed/val.csv"),
            str(GOLDEN),
        ],
        cwd=tmp_path,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "classifier-free fit, matrix, and eight query vectors passed"


def test_schema_names_vocabulary_parameters_and_fitted_statistics_exact(
    preprocessor: RAGPreprocessor,
) -> None:
    assert preprocessor.metadata["preprocessing"] == BASELINE["preprocessing"]
    assert preprocessor.input_feature_names == tuple(
        BASELINE["preprocessing"]["input_feature_names"]
    )
    assert preprocessor.output_feature_names == tuple(
        BASELINE["preprocessing"]["output_feature_names"]
    )
    assert preprocessor.metadata["fit"]["row_count"] == 4930
    assert preprocessor.metadata["fit"]["target_used"] is False
    assert preprocessor.metadata["fit"]["train_sha256"] == sha256(TRAIN.read_bytes())
    assert preprocessor.metadata["fit"]["splits_sha256"] == sha256(SPLITS.read_bytes())
    assert preprocessor.metadata["libraries"] == BASELINE["source"]["libraries"]


def test_new_reference_matrix_matches_all_original_values_and_hash(
    candidate: tuple[CandidateReferenceView, Any],
) -> None:
    view, expected = candidate
    assert view.matrix.shape == (4930, 45) and view.matrix.dtype == np.float64
    assert view.matrix.flags.c_contiguous
    assert np.array_equal(view.matrix, expected)
    assert float(np.max(np.abs(view.matrix - expected))) == 0.0
    assert array_digest(view.matrix) == BASELINE["matrix"]["values_sha256"]
    assert (
        json_digest([row.customer_id for row in view.records])
        == BASELINE["matrix"]["ordered_reference_ids_sha256"]
    )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_actual_new_query_vector_matches_frozen_v1(
    preprocessor: RAGPreprocessor,
    queries: dict[str, dict[str, object]],
    case: dict[str, Any],
) -> None:
    query = queries[case["name"]]
    actual = preprocessor.transform_one(query)
    assert actual.shape == (1, 45) and actual.dtype == np.float64
    assert actual.flags.c_contiguous
    assert array_digest(actual) == case["vector_sha256"]
    assert json_digest(preprocessor.clean(query)) == case["cleaned_features_sha256"]
    if PRIVATE.is_file():
        expected = json.loads(PRIVATE.read_text())["queries"][case["name"]]["vector"]
        if not np.array_equal(actual, expected):
            pytest.fail("New query vector differs from the private frozen values")


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_retrieval_uses_new_matrix_and_vectors_with_original_top_five(
    candidate: tuple[CandidateReferenceView, Any],
    queries: dict[str, dict[str, object]],
    case: dict[str, Any],
) -> None:
    retriever = make_retriever(candidate[0])
    actual = retriever.retrieve(queries[case["name"]])
    assert len(actual) == 5
    assert json_digest([row.customer_id for row in actual]) == case["ordered_neighbor_ids_sha256"]
    np.testing.assert_allclose(
        [row.distance for row in actual], case["distances"], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_eight_prompt_hashes_and_example_order_use_new_retrieval(
    preprocessor: RAGPreprocessor,
    candidate: tuple[CandidateReferenceView, Any],
    queries: dict[str, dict[str, object]],
    case: dict[str, Any],
) -> None:
    view = candidate[0]
    query = queries[case["name"]]
    neighbors = make_retriever(view).retrieve(query)
    built = ChurnPromptBuilder(preprocessor.feature_schema).build_with_metadata(
        preprocessor.clean(query), neighbors
    )
    assert built.reference_count == 5
    assert sha256(built.text.encode()) == case["prompt"]["sha256"]
    payload = json.loads(
        built.text.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split(
            "\nEND_CUSTOMER_DATA_JSON\n", 1
        )[0]
    )
    assert json_digest(payload["reference_customers"]) == case["prompt"]["ordered_examples_sha256"]
    assert [row["Churn"] for row in payload["reference_customers"]] == case["prompt"][
        "reference_labels"
    ]
    if any(row.customer_id in built.text for row in view.records):
        pytest.fail("A reference ID entered a prompt")


@pytest.mark.parametrize("exclude_self", [False, True])
def test_new_vectors_preserve_boundary_ties_and_self_exclusion(
    preprocessor: RAGPreprocessor,
    queries: dict[str, dict[str, object]],
    exclude_self: bool,
) -> None:
    features = preprocessor.clean(queries["first"])
    records = tuple(
        ReferenceRecord(features, "No", "synthetic-" + suffix)
        for suffix in ("f", "b", "e", "a", "d", "c")
    )
    matrix = preprocessor.transform_many([row.features for row in records])
    view = CandidateReferenceView(preprocessor, records, matrix)
    results = make_retriever(view).retrieve(
        features, customer_id="synthetic-c" if exclude_self else None
    )
    suffixes = ("a", "b", "d", "e", "f") if exclude_self else ("a", "b", "c", "d", "e")
    assert [row.customer_id for row in results] == ["synthetic-" + suffix for suffix in suffixes]
    assert len({row.distance for row in results}) == 1
    np.testing.assert_allclose([row.distance for row in results], 0.0, atol=1e-7)


def synthetic_source(directory: Path) -> tuple[Path, Path]:
    categories = BASELINE["preprocessing"]["categorical_vocabulary"]
    rows = [
        {
            "customerID": f"synthetic-train-{3 - index}",
            "Churn": "NEVER_FIT_THIS_LABEL",
            "SeniorCitizen": index % 2,
            "tenure": index,
            "MonthlyCharges": 20.0 + index,
            "TotalCharges": float(index * (20 + index)),
            **{name: values[index % len(values)] for name, values in categories.items()},
        }
        for index in range(4)
    ]
    train, splits = directory / "train.csv", directory / "splits.csv"
    pd.DataFrame(rows).to_csv(train, index=False)
    pd.DataFrame(
        {
            "customerID": [row["customerID"] for row in rows] + ["synthetic-val", "synthetic-test"],
            "split": ["train"] * 4 + ["val", "test"],
        }
    ).to_csv(splits, index=False)
    return train, splits


def test_fit_passes_only_nineteen_feature_columns_and_no_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]

    train, splits = synthetic_source(tmp_path)
    original = ColumnTransformer.fit
    calls = []

    def observe(self: Any, features: pd.DataFrame, y: Any = None, **kwargs: Any) -> Any:
        assert list(features.columns) == BASELINE["preprocessing"]["input_feature_names"]
        assert len(features) == 4 and y is None
        calls.append(True)
        return original(self, features, y=y, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit", observe)
    first = RAGPreprocessor.fit_train(train, splits)
    frame = pd.read_csv(train)
    frame["Churn"] = "CHANGED_TARGET_IS_IGNORED"
    frame.to_csv(train, index=False)
    second = RAGPreprocessor.fit_train(train, splits)
    assert first.metadata["preprocessing"] == second.metadata["preprocessing"]
    frame = frame.drop(columns="Churn")
    frame.to_csv(train, index=False)
    third = RAGPreprocessor.fit_train(train, splits)
    assert first.metadata["preprocessing"] == third.metadata["preprocessing"]
    assert len(calls) == 3


def test_fit_imputation_preserves_raw_processed_values_separately_from_query_cleaning(
    tmp_path: Path,
) -> None:
    train, splits = synthetic_source(tmp_path)
    frame = pd.read_csv(train)
    frame.loc[0, "TotalCharges"] = np.nan  # tenure=0 is a valid missing-charge row.
    frame.to_csv(train, index=False)
    fitted = RAGPreprocessor.fit_train(train, splits)
    steps = fitted.metadata["preprocessing"]["steps"]
    assert steps["num.imputer"]["state"]["statistics_"][-1] == 44.0
    assert steps["num.scaler"]["state"]["mean_"][-1] == 44.5
    assert steps["num.scaler"]["state"]["var_"][-1] == 288.25
    query = {name: frame.iloc[0][name] for name in fitted.input_feature_names}
    assert fitted.clean(query)["TotalCharges"] == 0.0
    assert fitted.transform_one(query)[0, 3] == (0.0 - 44.5) / np.sqrt(288.25)


@pytest.mark.parametrize(
    "kind",
    [
        "val",
        "test",
        "unknown",
        "duplicate",
        "incomplete",
        "full_dataset",
        "validation_only",
    ],
)
def test_non_train_populations_are_rejected_before_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]

    train, splits = synthetic_source(tmp_path)
    frame = pd.read_csv(train)
    if kind in {"val", "test", "unknown"}:
        frame.loc[0, "customerID"] = "synthetic-" + kind
    elif kind == "duplicate":
        frame.loc[0, "customerID"] = frame.loc[1, "customerID"]
    elif kind == "incomplete":
        frame = frame.iloc[:-1]
    elif kind == "full_dataset":
        frame = pd.concat([frame, frame.iloc[[0]].assign(customerID="synthetic-val")])
    else:
        frame = frame.iloc[[0]].assign(customerID="synthetic-val")
    frame.to_csv(train, index=False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid train membership reached fitting")

    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    with pytest.raises(ValueError, match="exactly the train split"):
        RAGPreprocessor.fit_train(train, splits)


@pytest.mark.parametrize("kind", ["duplicate_ids", "bad_split", "no_train", "bad_columns"])
def test_invalid_split_manifests_are_rejected(tmp_path: Path, kind: str) -> None:
    train, splits = synthetic_source(tmp_path)
    frame = pd.read_csv(splits)
    if kind == "duplicate_ids":
        frame.loc[0, "customerID"] = frame.loc[1, "customerID"]
    elif kind == "bad_split":
        frame.loc[0, "split"] = "unknown"
    elif kind == "no_train":
        frame["split"] = "val"
    else:
        frame["extra"] = 1
    frame.to_csv(splits, index=False)
    with pytest.raises(ValueError):
        RAGPreprocessor.fit_train(train, splits)


@pytest.mark.parametrize("missing", [None, "", "  ", float("nan"), pd.NA])
def test_missing_total_charges_rules_remain_unchanged(
    preprocessor: RAGPreprocessor,
    queries: dict[str, dict[str, object]],
    missing: object,
) -> None:
    query = {**queries["zero_tenure"], "TotalCharges": missing}
    assert (
        array_digest(preprocessor.transform_one(query))
        == next(case for case in CASES if case["name"] == "zero_tenure")["vector_sha256"]
    )
    with pytest.raises(ValueError, match="Missing TotalCharges"):
        preprocessor.transform_one({**query, "tenure": 1})


@pytest.mark.parametrize(
    "changes",
    [
        {"MonthlyCharges": None},
        {"MonthlyCharges": -1},
        {"MonthlyCharges": float("inf")},
        {"SeniorCitizen": True},
        {"tenure": 0.5},
        {"gender": None},
        {"gender": "__UNKNOWN__"},
        {"extra": 1},
    ],
)
def test_invalid_feature_values_are_rejected(
    preprocessor: RAGPreprocessor,
    queries: dict[str, dict[str, object]],
    changes: dict[str, object],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        preprocessor.transform_one({**queries["first"], **changes})


def test_batch_input_order_dtype_empty_and_no_transform_time_refitting(
    preprocessor: RAGPreprocessor,
    queries: dict[str, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = pickle.dumps(preprocessor._transformer)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Transform must never refit preprocessing")

    monkeypatch.setattr(type(preprocessor._transformer), "fit", forbidden)
    rows = [dict(reversed(list(query.items()))) for query in queries.values()]
    many = preprocessor.transform_many(rows)
    assert np.array_equal(many, np.vstack([preprocessor.transform_one(row) for row in rows]))
    assert many.shape == (8, 45) and many.dtype == np.float64
    assert preprocessor.transform_many([]).shape == (0, 45)
    assert pickle.dumps(preprocessor._transformer) == before
    with pytest.raises(ValueError, match="missing required features"):
        preprocessor.transform_one({"tenure": 0})
    with pytest.raises(TypeError):
        preprocessor.transform_many("invalid")  # type: ignore[arg-type]
    metadata = preprocessor.metadata
    metadata["fit"]["row_count"] = -1
    assert preprocessor.metadata["fit"]["row_count"] == 4930


def test_verification_tool_reports_actual_new_implementation_equivalence(
    frozen_rag_oracle: Path,
) -> None:
    report = verify()
    assert report["equivalent"] is True
    assert report["matrix_max_absolute_error"] == 0.0
    assert len(report["queries"]) == 8
