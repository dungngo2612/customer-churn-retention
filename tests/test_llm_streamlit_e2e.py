"""Offline UI-to-SDK integration; only SDK I/O and the artifact path are replaced."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest
import test_llm_prediction_service as service_tests
from google.genai import errors
from groq import APIConnectionError, APITimeoutError, RateLimitError
from llm_v2_fixtures import make_source, query
from streamlit.testing.v1 import AppTest
from test_llm_input_page import fill
from test_llm_prediction_service import PRIVATE, VALID
from test_llm_router import Clock

from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.entrypoints.streamlit import llm_input
from churn_retention.entrypoints.streamlit.llm_input import RESOURCES_KEY, RESULT_KEY
from churn_retention.modules.scoring import llm_router
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor

ROOT = Path(__file__).resolve().parents[1]
# Register the existing fixture without copying its fake SDK/lifecycle logic.
sdk = service_tests.sdk


@pytest.fixture(scope="module")
def reference_source(tmp_path_factory: pytest.TempPathFactory):
    source = make_source(tmp_path_factory.mktemp("llm-e2e"), n=8)
    source.build()
    return source


@pytest.fixture
def app(reference_source, sdk, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "CHURN_LLM_REFERENCE_SHA256",
        hashlib.sha256(reference_source.output.read_bytes()).hexdigest(),
    )
    real_load = llm_input.load_form_resources
    real_fingerprint = llm_input.reference_fingerprint
    loader = Mock(side_effect=lambda _: real_load(reference_source.output))
    monkeypatch.setattr(llm_input, "load_form_resources", loader)
    monkeypatch.setattr(
        llm_input,
        "reference_fingerprint",
        lambda _: real_fingerprint(reference_source.output),
    )
    calls = Mock()
    stores = []
    references = []
    real_predict = LLMChurnPredictor.predict
    real_retrieve = SimilarCustomerRetriever.retrieve

    async def counted_predict(self, customer):
        calls()
        return await real_predict(self, customer)

    def observed_retrieve(self, customer, **kwargs):
        result = real_retrieve(self, customer, **kwargs)
        stores.append(self._store)
        references.append(result)
        return result

    monkeypatch.setattr(LLMChurnPredictor, "predict", counted_predict)
    monkeypatch.setattr(SimilarCustomerRetriever, "retrieve", observed_retrieve)
    at = AppTest.from_file(str(ROOT / "apps/streamlit/scoring_page.py"), default_timeout=20).run()
    assert not at.exception and at.title[0].value == "Chấm điểm churn"
    at.session_state["active_scoring_context"] = {"protected": "existing context"}
    at.switch_page("pages/2_Predict.py").run()
    assert not at.exception and not at.error
    values = {
        name: value
        for name, value in query(reference_source).items()
        if name not in {"customerID", "Churn"}
    }
    return SimpleNamespace(
        ui=at,
        loader=loader,
        calls=calls,
        stores=stores,
        references=references,
        values=values,
    )


def submit(app) -> AppTest:
    fill(app.ui, app.values)
    return app.ui.button[0].click().run()


def prompt_data(client) -> dict:
    kwargs = client.request.call_args.kwargs
    prompt = kwargs["messages"][0]["content"] if client.role == "Groq" else kwargs["contents"]
    assert "customerID" not in prompt and "customer_id" not in prompt
    return json.loads(
        prompt.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split("\nEND_CUSTOMER_DATA_JSON\n", 1)[0]
    )


def test_navigation_and_invalid_input_never_create_clients(app, sdk) -> None:
    at = app.ui
    widgets = list(at.selectbox) + list(at.number_input)
    assert len(widgets) == 19
    assert len({widget.key for widget in widgets}) == 19
    assert len(at.subheader) == 3
    at.button[0].click().run()
    assert not at.exception and at.error and not at.success
    app.calls.assert_not_called()
    sdk.groq.assert_not_called()
    sdk.gemini.assert_not_called()


@pytest.mark.parametrize("prediction", [0, 1])
def test_qwen_result_flows_through_real_rag_and_json_validation(app, sdk, prediction) -> None:
    sdk.outcomes["Groq"] = json.dumps({**json.loads(VALID), "churn_prediction": prediction})
    at = submit(app)
    assert not at.exception and not at.error
    app.calls.assert_called_once_with()
    assert sdk.clients[0].request.await_count == 1
    assert sdk.clients[1].request.await_count == 0
    result = at.session_state[RESULT_KEY]
    assert result.churn_prediction == prediction and result.provider == "Groq"
    assert result.reference_count == 5 and not result.probability_calibrated
    assert at.metric[0].value == "60%"
    assert at.success[0].value == (
        "Khách hàng có khả năng rời bỏ"
        if prediction == 1
        else "Khách hàng có khả năng tiếp tục sử dụng"
    )
    assert result.reason in [item.value for item in at.text]
    assert "Provider: Groq/Qwen" in [item.value for item in at.caption]
    assert any("chưa được hiệu chuẩn" in item.value for item in at.warning)
    store = at.session_state[RESOURCES_KEY].store
    assert app.stores == [store] and len(app.references[0]) == 5
    prepared = store.prepare_query(app.values)
    # Independent full-distance calculation verifies the actual retrieved order.
    distances = np.linalg.norm(store.matrix - prepared.vector, axis=1)
    ids = np.asarray([record.customer_id for record in store.records])
    expected = np.lexsort((ids, distances))[:5]
    references = app.references[0]
    assert [item.customer_id for item in references] == [
        store.records[int(index)].customer_id for index in expected
    ]
    assert np.allclose([item.distance for item in references], distances[expected])
    data = prompt_data(sdk.clients[0])
    assert len(data["customer_to_predict"]) == 19
    assert "Churn" not in data["customer_to_predict"]
    assert data["reference_count"] == len(data["reference_customers"]) == 5
    assert data["reference_customers"] == [item.to_prompt_data() for item in references]
    assert result.reference_version == store.reference_version
    assert store.manifest.reference_split == "train"
    assert at.session_state["active_scoring_context"] == {"protected": "existing context"}
    assert not at.dataframe and not at.get("progress")


@pytest.mark.parametrize("kind", ["429", "timeout", "connection", "invalid_json"])
def test_fallback_crosses_entire_ui_pipeline_once(app, sdk, kind: str) -> None:
    request = httpx.Request("POST", "https://groq.invalid")
    sdk.outcomes["Groq"] = {
        "429": RateLimitError(PRIVATE, response=httpx.Response(429, request=request), body={}),
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(message=PRIVATE, request=request),
        "invalid_json": PRIVATE,
    }[kind]
    at = submit(app)
    assert not at.exception and not at.error
    app.calls.assert_called_once_with()
    assert all(client.request.await_count == 1 for client in sdk.clients)
    assert at.session_state[RESULT_KEY].provider == "Gemini"
    assert "Provider: Gemini" in [item.value for item in at.caption]
    assert prompt_data(sdk.clients[0]) == prompt_data(sdk.clients[1])
    assert at.session_state[RESULT_KEY].reference_count == 5


@pytest.mark.parametrize("kind", ["quota", "json"])
def test_both_provider_failures_reach_safe_ui_error(app, sdk, kind: str) -> None:
    if kind == "quota":
        request = httpx.Request("POST", "https://groq.invalid")
        sdk.outcomes["Groq"] = RateLimitError(
            PRIVATE, response=httpx.Response(429, request=request), body={}
        )
        sdk.outcomes["Gemini"] = errors.ClientError(429, {"message": PRIVATE})
    else:
        sdk.outcomes.update(Groq=PRIVATE, Gemini=PRIVATE)
    at = submit(app)
    assert not at.exception and len(at.error) == 1 and not at.success
    assert PRIVATE not in at.error[0].value
    assert RESULT_KEY not in at.session_state
    assert all(client.request.await_count == 1 for client in sdk.clients)
    assert "quota" in at.error[0].value if kind == "quota" else "JSON hợp lệ" in at.error[0].value
    at.run()
    app.calls.assert_called_once_with()


def test_reruns_drafts_and_second_submit_reuse_store_without_duplicate_requests(
    app, sdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = submit(app)
    store = at.session_state[RESOURCES_KEY].store
    monkeypatch.setattr(pd, "read_csv", Mock(side_effect=AssertionError("No CSV reread")))
    for _ in range(2):
        at.run()
    at.session_state["llm_input.feature.tenure"] = app.values["tenure"] + 1
    at.run()
    assert not at.success and not at.metric
    app.calls.assert_called_once_with()
    assert len(sdk.clients) == 2
    at.button[0].click().run()
    assert not at.exception and not at.error and at.success
    assert app.calls.call_count == 2 and len(sdk.clients) == 4
    assert [client.request.await_count for client in sdk.clients] == [1, 0, 1, 0]
    assert app.stores == [store, store]
    app.loader.assert_called_once()
    assert sdk.clients[0].loop is not sdk.clients[2].loop


@pytest.mark.parametrize(
    "missing",
    [("GROQ_API_KEY",), ("GEMINI_API_KEY",), ("GROQ_API_KEY", "GEMINI_API_KEY")],
)
def test_missing_keys_do_not_open_clients(app, sdk, monkeypatch, missing) -> None:
    for name in missing:
        monkeypatch.delenv(name, raising=False)
    at = submit(app)
    assert not at.exception and at.error and not at.success
    assert "cấu hình" in at.error[0].value
    assert not sdk.clients
    sdk.groq.assert_not_called()
    sdk.gemini.assert_not_called()


def test_hedge_at_60_seconds_returns_gemini_and_cleans_up_qwen(app, sdk, monkeypatch) -> None:
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def slow_primary():
        clock.advance(60)
        await asyncio.Event().wait()

    sdk.outcomes["Groq"] = slow_primary
    at = submit(app)
    assert not at.exception and not at.error
    assert at.session_state[RESULT_KEY].provider == "Gemini"
    assert sdk.clients[0].cancelled and not sdk.clients[0].active
    assert all(client.http.is_closed and client.loop.is_closed() for client in sdk.clients)
    assert not [timer for timer in clock.timers if not timer.cancelled and not timer.fired]
    app.calls.assert_called_once_with()


@pytest.mark.parametrize("tenure,valid", [(0, True), (1, False)])
def test_total_charges_cleaning_runs_before_any_provider_call(app, sdk, tenure, valid) -> None:
    app.values.update(tenure=tenure, TotalCharges=None)
    at = submit(app)
    assert not at.exception
    if valid:
        assert at.success and not at.error
        assert prompt_data(sdk.clients[0])["customer_to_predict"]["TotalCharges"] == 0
        app.calls.assert_called_once_with()
    else:
        assert at.error and not at.success
        app.calls.assert_not_called()
        assert not sdk.clients


@pytest.mark.skipif(
    not (ROOT / "artifacts/references/telco-train-v2.zip").is_file()
    or not (ROOT / "data/processed/val.csv").is_file(),
    reason="Optional project reference/validation assets are not available.",
)
def test_project_artifact_and_validation_row_offline(sdk, monkeypatch) -> None:
    """Use local project assets without printing IDs, input values, or prompts."""
    validation = pd.read_csv(ROOT / "data/processed/val.csv")
    loader = Mock(wraps=llm_input.load_form_resources)
    monkeypatch.setattr(llm_input, "load_form_resources", loader)
    at = AppTest.from_file(str(ROOT / "apps/streamlit/scoring_page.py"), default_timeout=30).run()
    at.switch_page("pages/2_Predict.py").run()
    if at.exception or at.error:
        pytest.fail("Project reference page initialization failed.", pytrace=False)
    store = at.session_state[RESOURCES_KEY].store
    names = [column.name for column in store.feature_schema]
    row = validation.iloc[0]
    values = {name: row[name] for name in names}
    validation_ids = set(validation["customerID"].astype(str))
    reference_ids = {record.customer_id for record in store.records}
    monkeypatch.setattr(pd, "read_csv", Mock(side_effect=AssertionError("No CSV reread")))
    fill(at, values)
    at.button[0].click().run()
    if at.exception or at.error or RESULT_KEY not in at.session_state:
        pytest.fail("Project-data offline prediction failed.", pytrace=False)
    result = at.session_state[RESULT_KEY]
    primary, fallback = sdk.clients
    prompt = primary.request.call_args.kwargs["messages"][0]["content"]
    data = json.loads(
        prompt.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split("\nEND_CUSTOMER_DATA_JSON\n", 1)[0]
    )
    checks = {
        "reference_rows": len(store.records) == 4930,
        "train_only": store.manifest.reference_split == "train",
        "validation_excluded": reference_ids.isdisjoint(validation_ids),
        "19_query_features": len(data["customer_to_predict"]) == 19,
        "no_query_label": "Churn" not in data["customer_to_predict"],
        "no_identifiers": "customerID" not in prompt and "customer_id" not in prompt,
        "five_references": len(data["reference_customers"]) == result.reference_count == 5,
        "provider": result.provider == "Groq",
        "version": result.reference_version == store.reference_version,
        "one_primary_request": primary.request.await_count == 1,
        "zero_fallback_requests": fallback.request.await_count == 0,
        "display": bool(at.success) and at.metric[0].value == "60%",
    }
    at.run()
    checks["rerun_no_new_clients"] = len(sdk.clients) == 2
    checks["rerun_no_new_requests"] = primary.request.await_count == 1
    checks["store_loaded_once"] = loader.call_count == 1
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        pytest.fail("Project-data checks failed: " + ", ".join(failures), pytrace=False)
