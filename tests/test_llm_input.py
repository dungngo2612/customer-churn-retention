import pickle
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from llm_v2_fixtures import CATEGORIES, make_source, query

from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.entrypoints.streamlit.llm_input import (
    FEATURE_GROUPS,
    FEATURE_LABELS,
    form_resources,
    load_form_resources,
    reference_artifact_sha256,
    reference_fingerprint,
    validate_customer,
)


@pytest.fixture(scope="module")
def input_fixture(tmp_path_factory: pytest.TempPathFactory):
    source = make_source(tmp_path_factory.mktemp("llm-input"), n=8)
    source.build()
    store = ReferenceStore.load(source.output, trusted=True)
    values = {
        key: value
        for key, value in query(source).items()
        if key not in {"customerID", "Churn"}
    }
    return source, form_resources(store), values


def test_form_has_exact_schema_and_requested_groups(input_fixture) -> None:
    _, resources, _ = input_fixture
    grouped = [name for _, names in FEATURE_GROUPS for name in names]
    assert len(grouped) == len(set(grouped)) == 19
    assert set(grouped) == set(resources.feature_names) == set(FEATURE_LABELS)
    assert [len(names) for _, names in FEATURE_GROUPS] == [5, 9, 5]
    assert set(resources.choices) == set(CATEGORIES)


def test_category_choices_are_verified_train_values_without_identifiers(
    input_fixture,
) -> None:
    source, resources, _ = input_fixture
    for name, choices in resources.choices.items():
        assert choices == tuple(sorted(set(source.frame[name])))
        assert set(choices) == set(CATEGORIES[name])
    assert (
        "customerID" not in resources.feature_names
        and "Churn" not in resources.feature_names
    )
    with pytest.raises(TypeError):
        resources.choices["gender"] = ("new",)


def test_loader_uses_existing_reference_store_once(
    input_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, resources, _ = input_fixture
    loader = Mock(return_value=resources.store)
    monkeypatch.setattr(ReferenceStore, "load", loader)
    loaded = load_form_resources(source.output)
    loader.assert_called_once_with(
        source.output, expected_artifact_sha256=reference_artifact_sha256()
    )
    assert loaded.store is resources.store


def test_valid_input_delegates_to_prepare_query_and_preserves_raw_values(
    input_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, resources, values = input_fixture
    prepare = Mock(wraps=resources.store.prepare_query)
    monkeypatch.setattr(resources.store, "prepare_query", prepare)
    cleaned = validate_customer(resources, values)
    prepare.assert_called_once_with(values)
    assert cleaned == values
    assert list(cleaned) == list(resources.feature_names)
    assert isinstance(cleaned["tenure"], int)
    assert len(cleaned) == 19 and all(
        not name.startswith(("num__", "cat__")) for name in cleaned
    )


@pytest.mark.parametrize(
    "change", ["missing", "extra", "customerID", "customer_id", "Churn"]
)
def test_rejects_any_input_other_than_exactly_19_features(
    input_fixture, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _, resources, values = input_fixture
    changed = dict(values)
    if change == "missing":
        changed.pop("gender")
    else:
        changed[change] = "PRIVATE_INPUT"
    prepare = Mock(
        side_effect=AssertionError("Must reject shape before preparing query")
    )
    monkeypatch.setattr(resources.store, "prepare_query", prepare)
    with pytest.raises(ValueError, match="19 đặc trưng"):
        validate_customer(resources, changed)
    prepare.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("tenure", None),
        ("tenure", True),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", True),
        ("MonthlyCharges", float("inf")),
        ("MonthlyCharges", float("nan")),
        ("MonthlyCharges", -1.0),
        ("MonthlyCharges", "PRIVATE_INVALID_VALUE"),
        ("TotalCharges", float("inf")),
        ("gender", "invalid-category"),
        ("gender", None),
        ("gender", " Female "),
    ],
)
def test_existing_validator_rejects_invalid_values_with_safe_message(
    input_fixture, field: str, value: object
) -> None:
    _, resources, values = input_fixture
    with pytest.raises(ValueError) as caught:
        validate_customer(resources, {**values, field: value})
    assert "PRIVATE_INVALID_VALUE" not in str(caught.value)
    assert "invalid-category" not in str(caught.value)


@pytest.mark.parametrize("missing", [None, "", " \t", float("nan"), pd.NA])
def test_missing_total_charges_only_zero_tenure_is_cleaned(
    input_fixture, missing: object
) -> None:
    _, resources, values = input_fixture
    cleaned = validate_customer(
        resources, {**values, "tenure": 0, "TotalCharges": missing}
    )
    assert cleaned["TotalCharges"] == 0.0
    with pytest.raises(ValueError, match="tenure.*0"):
        validate_customer(resources, {**values, "tenure": 1, "TotalCharges": missing})


def test_does_not_calculate_charges_or_mutate_preprocessing(
    input_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, resources, values = input_fixture
    original = pickle.dumps(resources.store.preprocessor._transformer)
    monkeypatch.setattr(
        pd, "read_csv", Mock(side_effect=AssertionError("No CSV read per submission"))
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            Path, "read_bytes", Mock(side_effect=AssertionError("No artifact reread"))
        )
        row = {**values, "tenure": 2, "MonthlyCharges": 20.0, "TotalCharges": 7.0}
        first = validate_customer(resources, row)
        second = validate_customer(resources, row)
    assert first == second and first["TotalCharges"] == 7.0
    assert row["TotalCharges"] == 7.0
    assert pickle.dumps(resources.store.preprocessor._transformer) == original


def test_unknown_validation_exception_is_not_displayed(
    input_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, resources, values = input_fixture
    monkeypatch.setattr(
        resources.store,
        "prepare_query",
        Mock(side_effect=ValueError("PRIVATE_CUSTOMER_OR_SECRET")),
    )
    with pytest.raises(ValueError) as caught:
        validate_customer(resources, values)
    assert "PRIVATE_" not in str(caught.value)


def test_form_snapshot_is_separate_from_original_input(input_fixture) -> None:
    _, resources, values = input_fixture
    cleaned = validate_customer(resources, values)
    cleaned["tenure"] = 999
    assert values["tenure"] != 999


def test_reference_fingerprint_is_stable_without_reading_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "reference.zip"
    artifact.write_bytes(b"test-only")
    monkeypatch.setattr(
        Path, "read_bytes", Mock(side_effect=AssertionError("Stat only"))
    )
    assert reference_fingerprint(artifact) == reference_fingerprint(artifact)


def test_reference_fingerprint_detects_edit_and_atomic_replacement(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "reference.zip"
    artifact.write_bytes(b"test-only")
    original = reference_fingerprint(artifact)
    artifact.write_bytes(b"test-only-edited")
    changed = reference_fingerprint(artifact)
    assert changed != original
    replacement = tmp_path / "new.zip"
    replacement.write_bytes(b"test-only-edited")
    replacement.replace(artifact)
    assert reference_fingerprint(artifact) != changed


@pytest.mark.parametrize("directory", [False, True])
def test_reference_fingerprint_rejects_missing_or_directory(
    tmp_path: Path, directory: bool
) -> None:
    path = tmp_path if directory else tmp_path / "missing.zip"
    with pytest.raises((OSError, ValueError)):
        reference_fingerprint(path)
