import asyncio
import json
import pickle
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pandas as pd
import pytest
from google.genai import errors
from groq import APIConnectionError, APITimeoutError, RateLimitError
from llm_v2_fixtures import make_source, query
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from test_llm_router import Clock

from churn_retention.adapters.llm import gemini_provider, groq_provider
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.entrypoints.streamlit import llm_prediction_service as service
from churn_retention.modules.scoring import llm_router

VALID = json.dumps(
    {
        "churn_prediction": 1,
        "churn_probability": 0.6,
        "reason": "Hợp đồng là bằng chứng tham khảo; ước lượng còn không chắc chắn.",
    },
    ensure_ascii=False,
)
PRIVATE = "PRIVATE_SECRET_CUSTOMER_OR_RESPONSE"


@pytest.fixture(scope="module")
def input_fixture(tmp_path_factory: pytest.TempPathFactory):
    source = make_source(tmp_path_factory.mktemp("llm-service"), n=8)
    source.build()
    store = ReferenceStore.load(source.output, trusted=True)
    snapshot = {
        key: value for key, value in query(source).items() if key not in {"customerID", "Churn"}
    }
    return store, snapshot


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch):
    """Real providers/router/predictor, fake SDKs, real local HTTP-client cleanup."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-no-network")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-no-network")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    outcomes: dict[str, object] = {"Groq": VALID, "Gemini": VALID}
    clients = []
    forbidden_network = AsyncMock(side_effect=AssertionError("No network in tests"))
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden_network)

    def make_client(role: str, http_client: httpx.AsyncClient):
        loop = asyncio.get_running_loop()
        state = SimpleNamespace(
            role=role, loop=loop, http=http_client, active=False, cancelled=False
        )

        async def request(**kwargs):
            assert asyncio.get_running_loop() is loop
            assert not http_client.is_closed
            state.active = True
            try:
                outcome = outcomes[role]
                if callable(outcome):
                    outcome = await outcome()
                if isinstance(outcome, BaseException):
                    raise outcome
                if role == "Groq":
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
                    )
                return SimpleNamespace(text=outcome)
            except asyncio.CancelledError:
                state.cancelled = True
                raise
            finally:
                state.active = False

        async def close():
            assert asyncio.get_running_loop() is loop and not state.active
            assert not [
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith("llm-router:") and not task.done()
            ]
            if role == "Groq":
                await http_client.aclose()

        state.request = AsyncMock(side_effect=request)
        state.close = AsyncMock(side_effect=close)

        def sync_close():
            assert asyncio.get_running_loop() is loop and not state.active
            assert http_client.is_closed

        state.sync_close = Mock(side_effect=sync_close)
        clients.append(state)
        return state

    def groq_sdk(**kwargs):
        assert kwargs["timeout"] == 90 and kwargs["max_retries"] == 0
        state = make_client("Groq", httpx.AsyncClient(timeout=kwargs["timeout"]))
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=state.request)),
            close=state.close,
        )

    def gemini_sdk(**kwargs):
        options = kwargs["http_options"]
        assert options.timeout == 30_000 and options.retry_options.attempts == 1
        state = make_client("Gemini", options.httpx_async_client)
        return SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=state.request),
                aclose=state.close,
            ),
            close=state.sync_close,
        )

    groq_constructor = Mock(side_effect=groq_sdk)
    gemini_constructor = Mock(side_effect=gemini_sdk)
    monkeypatch.setattr(groq_provider, "AsyncGroq", groq_constructor)
    monkeypatch.setattr(gemini_provider.genai, "Client", gemini_constructor)
    forbidden_lr = []
    for method in ("fit", "predict", "predict_proba"):
        blocked = Mock(side_effect=AssertionError("No LR call"))
        monkeypatch.setattr(LogisticRegression, method, blocked)
        forbidden_lr.append(blocked)
    yield SimpleNamespace(
        outcomes=outcomes,
        clients=clients,
        groq=groq_constructor,
        gemini=gemini_constructor,
    )
    forbidden_network.assert_not_called()
    assert all(mock.call_count == 0 for mock in forbidden_lr)
    for state in clients:
        assert state.http.is_closed and state.loop.is_closed()
        state.close.assert_awaited_once()
        if state.role == "Gemini":
            state.sync_close.assert_called_once()


def test_qwen_success_and_clients_closed(input_fixture, sdk) -> None:
    store, snapshot = input_fixture
    result = service.predict_customer(store, snapshot)
    assert result.provider == "Groq" and result.churn_prediction == 1
    assert result.churn_probability == 0.6 and not result.probability_calibrated
    assert result.reference_count == 5
    assert result.reference_version == store.reference_version
    primary, fallback = sdk.clients
    primary.request.assert_awaited_once()
    fallback.request.assert_not_called()
    assert primary.loop is fallback.loop and primary.loop.is_closed()


@pytest.mark.parametrize("failure", ["rate_limit", "timeout", "connection", "json"])
def test_qwen_failure_uses_gemini_once(input_fixture, sdk, failure: str) -> None:
    request = httpx.Request("POST", "https://groq.invalid")
    outcomes = {
        "rate_limit": RateLimitError(
            PRIVATE, response=httpx.Response(429, request=request), body={}
        ),
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(message=PRIVATE, request=request),
        "json": PRIVATE,
    }
    sdk.outcomes["Groq"] = outcomes[failure]
    result = service.predict_customer(*input_fixture)
    assert result.provider == "Gemini"
    assert all(state.request.await_count == 1 for state in sdk.clients)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("quota", "quota"),
        ("timeout", "thời gian chờ"),
        ("connection", "kết nối"),
        ("json", "JSON hợp lệ"),
        ("unknown", "Chưa thể hoàn tất"),
    ],
)
def test_failures_are_safe_and_clients_close(
    input_fixture, sdk, kind: str, expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    request = httpx.Request("POST", "https://groq.invalid")
    primary = {
        "quota": RateLimitError(PRIVATE, response=httpx.Response(429, request=request), body={}),
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(message=PRIVATE, request=request),
        "json": PRIVATE,
        "unknown": RuntimeError(PRIVATE),
    }
    fallback = {
        "quota": errors.ClientError(429, {"message": PRIVATE}),
        "timeout": httpx.ReadTimeout(PRIVATE),
        "connection": httpx.ConnectError(PRIVATE),
        "json": PRIVATE,
        "unknown": RuntimeError(PRIVATE),
    }
    sdk.outcomes.update(Groq=primary[kind], Gemini=fallback[kind])
    with pytest.raises(service.LLMPredictionServiceError) as caught:
        service.predict_customer(*input_fixture)
    assert expected in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value)) + caplog.text
    assert all(state.request.await_count == 1 for state in sdk.clients)


@pytest.mark.parametrize(
    "groq,gemini", [(None, None), ("fake", None), (None, "fake"), (" ", "fake")]
)
def test_missing_either_key_never_creates_clients(
    input_fixture, sdk, monkeypatch: pytest.MonkeyPatch, groq, gemini
) -> None:
    for name, value in (("GROQ_API_KEY", groq), ("GEMINI_API_KEY", gemini)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(service.LLMPredictionServiceError, match="cả GROQ_API_KEY"):
        service.predict_customer(*input_fixture)
    sdk.groq.assert_not_called()
    sdk.gemini.assert_not_called()


def test_two_submissions_use_new_loops_same_store_no_dataset_reads_or_fits(
    input_fixture, sdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, snapshot = input_fixture
    state = pickle.dumps(store.preprocessor._transformer)
    monkeypatch.setattr(ReferenceStore, "load", Mock(side_effect=AssertionError("Reuse Store")))
    monkeypatch.setattr(pd, "read_csv", Mock(side_effect=AssertionError("No dataset read")))
    transform = Mock(wraps=store.preprocessor._transformer.transform)

    def record_transform(self, *args, **kwargs):
        assert self is store.preprocessor._transformer
        return transform(*args, **kwargs)

    # Patch the class so undo does not leave a bound method in the instance state.
    monkeypatch.setattr(type(store.preprocessor._transformer), "transform", record_transform)
    predictor = Mock(wraps=service.LLMChurnPredictor)
    monkeypatch.setattr(service, "LLMChurnPredictor", predictor)
    first = service.predict_customer(store, snapshot)
    second = service.predict_customer(store, snapshot)
    assert first == second and len(sdk.clients) == 4
    assert sdk.clients[0].loop is not sdk.clients[2].loop
    assert predictor.call_count == 2
    assert all(call.args[0] is store for call in predictor.call_args_list)
    assert all(call.kwargs == {"k": 5} for call in predictor.call_args_list)
    assert all(call.args[0].shape[0] == 1 for call in transform.call_args_list)
    assert pickle.dumps(store.preprocessor._transformer) == state


def test_running_event_loop_is_rejected_without_creating_coroutine(input_fixture, sdk) -> None:
    async def case():
        with pytest.raises(service.LLMPredictionServiceError, match="async"):
            service.predict_customer(*input_fixture)

    asyncio.run(case())
    assert not sdk.clients


@pytest.mark.parametrize("extra", ["customerID", "Churn", "extra"])
def test_metadata_cannot_enter_snapshot(input_fixture, sdk, extra: str) -> None:
    store, snapshot = input_fixture
    with pytest.raises(service.LLMPredictionServiceError, match="19 đặc trưng"):
        service.predict_customer(store, {**snapshot, extra: PRIVATE})
    assert not sdk.clients


def test_predict_is_awaited_exactly_once(input_fixture, sdk, monkeypatch) -> None:
    result = LLMChurnPrediction(
        **json.loads(VALID),
        provider="Groq",
        reference_count=5,
        reference_version=input_fixture[0].reference_version,
    )
    predict = AsyncMock(return_value=result)
    constructor = Mock(return_value=SimpleNamespace(predict=predict))
    monkeypatch.setattr(service, "LLMChurnPredictor", constructor)
    assert service.predict_customer(*input_fixture) == result
    predict.assert_awaited_once_with(input_fixture[1])


def test_hedge_drains_loser_before_http_client_close(
    input_fixture, sdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def slow_primary():
        clock.advance(60)
        await asyncio.Event().wait()

    sdk.outcomes["Groq"] = slow_primary
    result = service.predict_customer(*input_fixture)
    assert result.provider == "Gemini"
    assert sdk.clients[0].cancelled and not sdk.clients[0].active
    assert not [timer for timer in clock.timers if not timer.cancelled and not timer.fired]


def test_overall_deadline_keeps_existing_timeout_and_cleans_up(
    input_fixture, sdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)

    async def slow_primary():
        clock.advance(120)
        await asyncio.Event().wait()

    sdk.outcomes["Groq"] = slow_primary
    with pytest.raises(service.LLMPredictionServiceError, match="vượt thời gian chờ"):
        service.predict_customer(*input_fixture)


def test_async_cancellation_closes_clients(input_fixture, sdk) -> None:
    async def case():
        started = asyncio.Event()

        async def slow_primary():
            started.set()
            await asyncio.Event().wait()

        sdk.outcomes["Groq"] = slow_primary
        task = asyncio.create_task(service._predict_once(*input_fixture))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(state.http.is_closed for state in sdk.clients)
        assert not [
            task for task in asyncio.all_tasks() if task.get_name().startswith("llm-router:")
        ]

    asyncio.run(case())
