import pandas as pd
import pytest

from churn_retention.adapters.predictor.fake import FakePredictor


def test_same_seed_same_results():
    predictor1 = FakePredictor(seed=100)
    predictor1.load("dummy/path")
    predictor2 = FakePredictor(seed=100)
    predictor2.load("dummy/path")

    features = pd.DataFrame({"feature1": [1, 2, 3]})
    customer_ids = ["C001", "C002", "C003"]

    result1 = predictor1.predict_batch(features, customer_ids)
    result2 = predictor2.predict_batch(features, customer_ids)

    assert result1 == result2


def test_result_length_matches_input():
    predictor = FakePredictor(seed=42)
    predictor.load("dummy/path")

    features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    customer_ids = ["C001", "C002", "C003", "C004", "C005"]

    result = predictor.predict_batch(features, customer_ids)

    assert len(result) == len(customer_ids)
    assert len(result) == len(features)


def test_all_probabilities_in_valid_range():
    predictor = FakePredictor(seed=42)
    predictor.load("dummy/path")

    features = pd.DataFrame({"feature1": range(100)})
    customer_ids = [f"C{i:03d}" for i in range(100)]

    result = predictor.predict_batch(features, customer_ids)

    assert all(0.0 <= prob <= 1.0 for prob in result)


def test_length_mismatch_raises_error():
    predictor = FakePredictor(seed=42)
    predictor.load("dummy/path")

    features = pd.DataFrame({"feature1": [1, 2, 3]})
    customer_ids = ["C001", "C002"]

    with pytest.raises(ValueError) as exc_info:
        predictor.predict_batch(features, customer_ids)

    assert "Length mismatch" in str(exc_info.value)


def test_empty_batch_returns_empty():
    predictor = FakePredictor(seed=42)
    predictor.load("dummy/path")

    features = pd.DataFrame()
    customer_ids = []

    result = predictor.predict_batch(features, customer_ids)

    assert result == []


def test_same_customer_id_same_probability_across_batches():
    predictor = FakePredictor(seed=42)
    predictor.load("dummy/path")

    features1 = pd.DataFrame({"feature1": [1, 2, 3]})
    customer_ids1 = ["C001", "C002", "C003"]
    result1 = predictor.predict_batch(features1, customer_ids1)

    features2 = pd.DataFrame({"feature1": [10, 20]})
    customer_ids2 = ["C002", "C001"]
    result2 = predictor.predict_batch(features2, customer_ids2)

    assert result1[1] == result2[0]  # C002 same in both
    assert result1[0] == result2[1]  # C001 same in both


def test_model_version_is_fake_v1():
    predictor = FakePredictor(seed=42)
    assert predictor.model_version == "fake-v1"


def test_model_version_is_readonly_property():
    predictor = FakePredictor(seed=42)
    with pytest.raises(AttributeError):
        predictor.model_version = "modified"
