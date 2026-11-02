from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from llm_v2_fixtures import make_source, query
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from streamlit.testing.v1 import AppTest

from churn_retention.adapters.llm.gemini_provider import GeminiProvider
from churn_retention.adapters.llm.groq_provider import GroqProvider
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.reference import ReferenceArtifactError
from churn_retention.entrypoints.streamlit import llm_input, llm_prediction_service
from churn_retention.entrypoints.streamlit.llm_input import (
    ERROR_KEY,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    FINGERPRINT_KEY,
    RESOURCES_KEY,
    RESULT_INPUT_KEY,
    RESULT_KEY,
    VALIDATED_INPUT_KEY,
    ReferenceFingerprint,
    form_resources,
)
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor
from churn_retention.modules.scoring.llm_router import LLMRouter

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "apps/streamlit/pages/2_Predict.py"
WIDGET_PREFIX = "llm_input.feature."
PREDICTION_MESSAGE = "Khách hàng có khả năng rời bỏ"
REAL_PREDICT = llm_prediction_service.predict_customer


@pytest.fixture(scope="module")
def input_fixture(tmp_path_factory: pytest.TempPathFactory):
    source = make_source(tmp_path_factory.mktemp("llm-page"), n=8)
    source.build()
    store = ReferenceStore.load(source.output, trusted=True)
    values = {
        key: value for key, value in query(source).items() if key not in {"customerID", "Churn"}
    }
    return form_resources(store), values


@pytest.fixture(autouse=True)
def no_inference(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    calls = []
    for cls, method in [
        (LLMChurnPredictor, "predict"),
        (LLMRouter, "generate"),
        (LLMRouter, "generate_result"),
        (GroqProvider, "generate"),
        (GeminiProvider, "generate"),
    ]:
        blocked = AsyncMock(side_effect=AssertionError("UI tests mock prediction service"))
        monkeypatch.setattr(cls, method, blocked)
        calls.append(blocked)
    for cls, method in [
        (GroqProvider, "__init__"),
        (GeminiProvider, "__init__"),
        (LogisticRegression, "predict"),
        (LogisticRegression, "predict_proba"),
        (LogisticRegression, "fit"),
    ]:
        blocked = Mock(side_effect=AssertionError("No provider/client or LR inference"))
        monkeypatch.setattr(cls, method, blocked)
        calls.append(blocked)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    assert all(mock.call_count == 0 for mock in calls)


@pytest.fixture
def prediction(input_fixture, monkeypatch: pytest.MonkeyPatch):
    resources, _ = input_fixture
    predictor = Mock(
        return_value=LLMChurnPrediction(
            churn_prediction=1,
            churn_probability=0.6,
            reason="Hợp đồng là bằng chứng tham khảo; ước lượng còn chưa chắc chắn.",
            provider="Groq",
            reference_count=5,
            reference_version=resources.store.reference_version,
            probability_calibrated=False,
        )
    )
    monkeypatch.setattr(llm_prediction_service, "predict_customer", predictor)
    return predictor


@pytest.fixture
def page(input_fixture, prediction, monkeypatch: pytest.MonkeyPatch):
    resources, _ = input_fixture
    loader = Mock(return_value=resources)
    monkeypatch.setattr(llm_input, "load_form_resources", loader)
    monkeypatch.setattr(
        llm_input,
        "reference_fingerprint",
        Mock(return_value=ReferenceFingerprint("unit-artifact", 1, 2, 3, 4, 5)),
    )
    return AppTest.from_file(str(PAGE), default_timeout=15), loader


def fill(at: AppTest, values: dict[str, Any]) -> None:
    for selection in at.selectbox:
        assert selection.key is not None
        selection.select(values[selection.key.removeprefix(WIDGET_PREFIX)])
    for number in at.number_input:
        assert number.key is not None
        number.set_value(values[number.key.removeprefix(WIDGET_PREFIX)])


def test_page_initializes_with_19_fields_in_three_groups(page, prediction) -> None:
    at, loader = page
    at.run()
    assert not at.exception and not at.error
    assert at.title[0].value == "Dự đoán khách hàng rời bỏ bằng LLM"
    assert [item.value for item in at.subheader] == [label for label, _ in FEATURE_GROUPS]
    widgets = list(at.selectbox) + list(at.number_input)
    assert len(widgets) == 19
    assert {widget.key.removeprefix(WIDGET_PREFIX) for widget in widgets} == set(FEATURE_LABELS)
    assert at.button[0].label == "Dự đoán bằng LLM"
    assert not at.metric and not at.dataframe and not at.success
    assert VALIDATED_INPUT_KEY not in at.session_state
    loader.assert_called_once()
    prediction.assert_not_called()


def test_category_choices_use_store_values_with_vietnamese_labels(page, input_fixture) -> None:
    at, _ = page
    resources, _ = input_fixture
    at.run()
    for name, choices in resources.choices.items():
        widget = at.selectbox(WIDGET_PREFIX + name)
        assert widget.options == [llm_input.value_label(value) for value in choices]
        assert widget.value is None
    assert at.selectbox(WIDGET_PREFIX + "SeniorCitizen").options == ["Không", "Có"]


def test_valid_submit_saves_only_clean_19_feature_snapshot(page, input_fixture, prediction) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.success[0].value == PREDICTION_MESSAGE
    snapshot = at.session_state[VALIDATED_INPUT_KEY]
    assert snapshot == values and len(snapshot) == 19
    assert not {"customerID", "customer_id", "Churn"} & snapshot.keys()
    assert at.metric[0].value == "60%" and not at.dataframe
    prediction.assert_called_once_with(input_fixture[0].store, snapshot)


def test_empty_submit_shows_clear_validation_error(page, prediction) -> None:
    at, _ = page
    at.run().button[0].click().run()
    assert not at.exception and len(at.error) == 1 and not at.success
    assert VALIDATED_INPUT_KEY not in at.session_state
    prediction.assert_not_called()


@pytest.mark.parametrize("tenure,valid", [(0, True), (1, False)])
def test_total_charges_missing_uses_existing_rule(
    page, input_fixture, tenure: int, valid: bool
) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, {**values, "tenure": tenure, "TotalCharges": None})
    at.button[0].click().run()
    assert not at.exception
    if valid:
        assert at.success[0].value == PREDICTION_MESSAGE
        assert at.session_state[VALIDATED_INPUT_KEY]["TotalCharges"] == 0.0
    else:
        assert "tenure" in at.error[0].value and not at.success
        assert VALIDATED_INPUT_KEY not in at.session_state


def test_widget_changes_and_reruns_do_not_prepare_query_or_infer(
    page, input_fixture, prediction, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, loader = page
    resources, values = input_fixture
    validator = Mock(wraps=resources.store.prepare_query)
    monkeypatch.setattr(resources.store, "prepare_query", validator)
    at.run()
    fill(at, values)
    at.run()
    validator.assert_not_called()
    # Explicit reruns do not submit the browser-side form draft.
    fill(at, values)
    at.button[0].click().run()
    validator.assert_called_once()
    previous = dict(at.session_state[VALIDATED_INPUT_KEY])
    at.number_input(WIDGET_PREFIX + "tenure").set_value(5).run()
    at.run()
    validator.assert_called_once()
    assert at.session_state[VALIDATED_INPUT_KEY] == previous
    if at.success:
        assert any("lần gửi dữ liệu gần nhất" in item.value for item in at.caption)
    prediction.assert_called_once()
    loader.assert_called_once()


def test_invalid_resubmit_clears_old_success_and_snapshot(page, input_fixture) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert at.success
    at.number_input(WIDGET_PREFIX + "MonthlyCharges").set_value(None)
    at.button[0].click().run()
    assert at.error and not at.success and VALIDATED_INPUT_KEY not in at.session_state
    assert RESULT_KEY not in at.session_state


def test_form_keeps_scoring_context_and_does_not_store_keys_or_clients(page, input_fixture) -> None:
    at, _ = page
    _, values = input_fixture
    original_context = {"existing-scoring-page": "untouched"}
    at.session_state["active_scoring_context"] = original_context
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert at.session_state["active_scoring_context"] == original_context
    assert RESOURCES_KEY in at.session_state and ERROR_KEY not in at.session_state
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "router", "client", "task"):
        assert name not in at.session_state


def test_missing_reference_artifact_shows_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_input,
        "load_form_resources",
        Mock(side_effect=ReferenceArtifactError("PRIVATE_PATH_OR_DATA")),
    )
    at = AppTest.from_file(str(PAGE), default_timeout=15).run()
    assert not at.exception and len(at.error) == 1
    assert "dữ liệu tham chiếu" in at.error[0].value
    assert "PRIVATE" not in at.error[0].value
    assert not at.button and not at.success and RESOURCES_KEY not in at.session_state


def test_existing_scoring_and_shap_pages_still_initialize() -> None:
    # Warm the existing SHAP/matplotlib import outside the script timeout.
    # Importing does not compute SHAP or execute any prediction.
    from importlib import import_module

    import_module("churn_retention.ml.explainability")
    at = AppTest.from_file(str(ROOT / "apps/streamlit/scoring_page.py"), default_timeout=30).run()
    assert not at.exception and at.title[0].value == "Chấm điểm churn"
    assert len(at.text_input) == 2
    at.switch_page("pages/1_Explain.py").run()
    assert not at.exception
    assert at.title[0].value == "Giải thích mô hình và khách hàng"
    assert "chấm điểm dataset" in at.warning[0].value


def test_second_submit_predicts_once_more_but_reruns_do_not(
    page, input_fixture, prediction
) -> None:
    at, loader = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    at.run()
    assert prediction.call_count == 1
    fill(at, {**values, "tenure": 5})
    at.button[0].click().run()
    assert prediction.call_count == 2
    assert prediction.call_args.args[1]["tenure"] == 5
    at.run()
    assert prediction.call_count == 2 and not at.exception
    loader.assert_called_once()


@pytest.mark.parametrize(
    "label,provider,message,provider_label",
    [
        (1, "Groq", PREDICTION_MESSAGE, "Groq/Qwen"),
        (0, "Gemini", "Khách hàng có khả năng tiếp tục sử dụng", "Gemini"),
    ],
)
def test_result_displays_actual_fields_and_uncalibrated_warning(
    page, input_fixture, prediction, label, provider, message, provider_label
) -> None:
    at, _ = page
    _, values = input_fixture
    prediction.return_value = prediction.return_value.model_copy(
        update={"churn_prediction": label, "provider": provider}
    )
    at.run()
    fill(at, values)
    at.button[0].click().run()
    result = prediction.return_value
    assert not at.exception and not at.error
    assert at.success[0].value == message
    assert at.metric[0].label == "Xác suất do LLM ước lượng"
    assert at.metric[0].value == "60%"
    assert any("chưa được hiệu chuẩn thành xác suất thống kê" in w.value for w in at.warning)
    assert result.reason in [item.value for item in at.text]
    captions = [item.value for item in at.caption]
    assert f"Provider: {provider_label}" in captions
    assert "Số khách hàng tham chiếu: 5" in captions
    assert any(result.reference_version in item.value for item in at.text)
    assert not at.dataframe and not at.get("progress")
    assert not any("Groq đã lỗi" in item for item in captions)


def test_reason_is_literal_text_not_rendered_markdown(page, input_fixture, prediction) -> None:
    at, _ = page
    _, values = input_fixture
    literal = "[không chắc chắn](https://invalid.example) <script>test</script>"
    prediction.return_value = prediction.return_value.model_copy(update={"reason": literal})
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert literal in [item.value for item in at.text]
    assert literal not in [item.value for item in at.markdown]


@pytest.mark.parametrize("safe", [False, True])
def test_prediction_failure_shows_safe_error_without_old_result(
    page, input_fixture, prediction, safe: bool
) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert at.success
    prediction.side_effect = (
        llm_prediction_service.LLMPredictionServiceError("Dịch vụ LLM hết thời gian chờ.")
        if safe
        else RuntimeError("PRIVATE_SECRET_PROMPT_OR_RESPONSE")
    )
    at.button[0].click().run()
    assert not at.exception and len(at.error) == 1
    assert "PRIVATE" not in at.error[0].value
    assert not at.success and not at.metric
    assert RESULT_KEY not in at.session_state and RESULT_INPUT_KEY not in at.session_state
    assert prediction.call_count == 2
    at.run()
    assert prediction.call_count == 2


def test_missing_keys_in_actual_service_shows_friendly_error_no_clients(
    page, input_fixture, prediction
) -> None:
    at, _ = page
    _, values = input_fixture
    prediction.side_effect = REAL_PREDICT
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert not at.exception and len(at.error) == 1 and not at.success
    assert "cả GROQ_API_KEY và GEMINI_API_KEY" in at.error[0].value
    prediction.assert_called_once()


def test_server_observed_draft_hides_previous_result_without_inference(
    page, input_fixture, prediction
) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    at.session_state[WIDGET_PREFIX + "tenure"] = values["tenure"] + 1
    at.run()
    assert not at.exception and not at.success and not at.metric
    assert any("form đã thay đổi" in item.value for item in at.info)
    prediction.assert_called_once()


def test_fingerprint_change_reloads_store_and_discards_old_result(
    page, input_fixture, prediction, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, loader = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert at.success and loader.call_count == 1
    updated = ReferenceFingerprint("unit-artifact", 1, 9, 3, 8, 7)
    monkeypatch.setattr(llm_input, "reference_fingerprint", Mock(return_value=updated))
    at.run()
    assert not at.exception and not at.success and loader.call_count == 2
    assert RESULT_KEY not in at.session_state and VALIDATED_INPUT_KEY not in at.session_state
    assert at.session_state[FINGERPRINT_KEY] == (
        updated,
        llm_input.reference_artifact_sha256(),
    )
    prediction.assert_called_once()


def test_reference_failure_after_success_clears_cached_store_and_result(
    page, input_fixture, prediction, monkeypatch: pytest.MonkeyPatch
) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    monkeypatch.setattr(
        llm_input, "reference_fingerprint", Mock(side_effect=OSError("PRIVATE_PATH"))
    )
    at.run()
    assert not at.exception and at.error and "PRIVATE" not in at.error[0].value
    assert RESOURCES_KEY not in at.session_state and RESULT_KEY not in at.session_state
    assert FINGERPRINT_KEY not in at.session_state
    prediction.assert_called_once()


def test_reference_replaced_during_load_is_not_cached(page, monkeypatch) -> None:
    at, _ = page
    fingerprints = [
        ReferenceFingerprint("unit-artifact", 1, 2, 3, 4, 5),
        ReferenceFingerprint("unit-artifact", 1, 2, 3, 4, 6),
    ]
    monkeypatch.setattr(llm_input, "reference_fingerprint", Mock(side_effect=fingerprints))
    at.run()
    assert not at.exception and at.error
    assert RESOURCES_KEY not in at.session_state and not at.button


def test_result_and_snapshot_have_no_keys_clients_tasks_or_identifiers(page, input_fixture) -> None:
    at, _ = page
    _, values = input_fixture
    at.run()
    fill(at, values)
    at.button[0].click().run()
    result = at.session_state[RESULT_KEY]
    assert isinstance(result, LLMChurnPrediction)
    assert not {"customerID", "Churn"} & at.session_state[RESULT_INPUT_KEY].keys()
    assert "API_KEY" not in result.model_dump_json()
    assert (
        not {"client", "router", "task", "GROQ_API_KEY", "GEMINI_API_KEY"}
        & result.model_dump().keys()
    )


def test_unexpected_validation_error_is_hidden_without_prediction(
    page, input_fixture, prediction, monkeypatch
) -> None:
    at, _ = page
    resources, values = input_fixture
    monkeypatch.setattr(
        resources.store,
        "prepare_query",
        Mock(side_effect=RuntimeError("PRIVATE_TRANSFORMER_DETAILS")),
    )
    at.run()
    fill(at, values)
    at.button[0].click().run()
    assert not at.exception and at.error
    assert "PRIVATE" not in at.error[0].value
    assert VALIDATED_INPUT_KEY not in at.session_state
    prediction.assert_not_called()
