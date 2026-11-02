import asyncio
import traceback
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai import errors, types

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from churn_retention.adapters.llm.gemini_provider import GeminiProvider

FAKE_API_KEY = "gemini-unit-test-key-never-used-for-network"


@pytest.fixture(autouse=True)
def mock_http_client() -> Iterator[MagicMock]:
    client = MagicMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    with patch(
        "churn_retention.adapters.llm.gemini_provider.httpx.AsyncClient",
        return_value=client,
    ) as constructor:
        yield constructor


@pytest.fixture(autouse=True)
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """All tests replace the SDK and HTTP clients; no real request is possible."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with patch("churn_retention.adapters.llm.gemini_provider.genai.Client") as sdk:
        sdk.return_value.aio.models.generate_content = AsyncMock()
        sdk.return_value.aio.aclose = AsyncMock()
        yield sdk


def completion(content: str | None) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=content)]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


def test_returns_content_and_awaits_sdk(
    mock_sdk: MagicMock, mock_http_client: MagicMock
) -> None:
    text = "  Khách hàng có nguy cơ rời bỏ.\n"
    mock_sdk.return_value.aio.models.generate_content.return_value = completion(text)
    provider: LLMProvider = GeminiProvider()

    assert asyncio.run(provider.generate("Phân tích khách hàng")) == text
    mock_sdk.return_value.aio.models.generate_content.assert_awaited_once_with(
        model="gemini-3.5-flash-lite",
        contents="Phân tích khách hàng",
        config=types.GenerateContentConfig(automatic_continuation=False),
    )
    mock_http_client.assert_called_once_with(timeout=30.0)
    mock_sdk.assert_called_once()
    options = mock_sdk.call_args.kwargs["http_options"]
    assert mock_sdk.call_args.kwargs["api_key"] == FAKE_API_KEY
    assert mock_sdk.call_args.kwargs["enterprise"] is False
    assert options.timeout == 30_000
    assert options.retry_options.attempts == 1
    assert options.httpx_async_client is mock_http_client.return_value


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_rejects_empty_content(mock_sdk: MagicMock, content: str | None) -> None:
    mock_sdk.return_value.aio.models.generate_content.return_value = completion(content)

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(GeminiProvider().generate("prompt"))


@pytest.mark.parametrize(
    "response",
    [
        types.GenerateContentResponse(),
        types.GenerateContentResponse(candidates=[]),
        types.GenerateContentResponse(candidates=[types.Candidate()]),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[]))]
        ),
        types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            )
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[types.Part(text="internal reasoning", thought=True)]
                    )
                )
            ]
        ),
    ],
)
def test_rejects_response_without_usable_text(
    mock_sdk: MagicMock, response: types.GenerateContentResponse
) -> None:
    mock_sdk.return_value.aio.models.generate_content.return_value = response

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(GeminiProvider().generate("prompt"))


@pytest.mark.parametrize(
    ("error", "expected_error", "status_code"),
    [
        (
            errors.ClientError(
                429,
                {"error": {"status": "RESOURCE_EXHAUSTED", "message": FAKE_API_KEY}},
            ),
            LLMRateLimitError,
            429,
        ),
        (httpx.ReadTimeout(FAKE_API_KEY), LLMTimeoutError, None),
        (httpx.ConnectTimeout(FAKE_API_KEY), LLMTimeoutError, None),
        (httpx.WriteTimeout(FAKE_API_KEY), LLMTimeoutError, None),
        (httpx.PoolTimeout(FAKE_API_KEY), LLMTimeoutError, None),
        (TimeoutError(FAKE_API_KEY), LLMTimeoutError, None),
        (httpx.ConnectError(FAKE_API_KEY), LLMConnectionError, None),
        (httpx.ReadError(FAKE_API_KEY), LLMConnectionError, None),
        (httpx.RemoteProtocolError(FAKE_API_KEY), LLMConnectionError, None),
        (ConnectionError(FAKE_API_KEY), LLMConnectionError, None),
        (errors.ClientError(408, {"message": FAKE_API_KEY}), LLMTimeoutError, 408),
        (errors.ServerError(504, {"message": FAKE_API_KEY}), LLMTimeoutError, 504),
        (errors.ClientError(400, {"message": FAKE_API_KEY}), LLMAPIError, 400),
        (errors.ClientError(401, {"message": FAKE_API_KEY}), LLMAPIError, 401),
        (errors.ClientError(403, {"message": FAKE_API_KEY}), LLMAPIError, 403),
        (errors.ServerError(500, {"message": FAKE_API_KEY}), LLMAPIError, 500),
        (errors.ServerError(503, {"message": FAKE_API_KEY}), LLMAPIError, 503),
        (errors.APIError(0, {"message": FAKE_API_KEY}), LLMAPIError, None),
    ],
)
def test_maps_failures_without_retrying_or_exposing_secrets(
    mock_sdk: MagicMock,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    expected_error: type[LLMProviderError],
    status_code: int | None,
) -> None:
    mock_sdk.return_value.aio.models.generate_content.side_effect = error

    with pytest.raises(expected_error) as caught:
        asyncio.run(GeminiProvider().generate("prompt"))

    assert type(caught.value) is expected_error
    if isinstance(caught.value, LLMAPIError):
        assert caught.value.status_code == status_code
    assert FAKE_API_KEY not in str(caught.value)
    assert FAKE_API_KEY not in "".join(traceback.format_exception(caught.value))
    assert FAKE_API_KEY not in caplog.text
    mock_sdk.return_value.aio.models.generate_content.assert_awaited_once()


@pytest.mark.parametrize("api_key", [None, "", " \n\t"])
def test_requires_api_key_before_creating_clients(
    monkeypatch: pytest.MonkeyPatch,
    mock_sdk: MagicMock,
    mock_http_client: MagicMock,
    api_key: str | None,
) -> None:
    # GEMINI_API_KEY is mandatory even if Google's alternative key is present.
    monkeypatch.setenv("GOOGLE_API_KEY", "different-unit-test-key")
    if api_key is None:
        monkeypatch.delenv("GEMINI_API_KEY")
    else:
        monkeypatch.setenv("GEMINI_API_KEY", api_key)

    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider()

    mock_sdk.assert_not_called()
    mock_http_client.assert_not_called()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "gemini-3.5-flash-lite"),
        ("", "gemini-3.5-flash-lite"),
        (" \t", "gemini-3.5-flash-lite"),
        ("custom-model", "custom-model"),
        ("  custom-model  ", "custom-model"),
    ],
)
def test_uses_configured_model_in_request(
    monkeypatch: pytest.MonkeyPatch,
    mock_sdk: MagicMock,
    configured: str | None,
    expected: str,
) -> None:
    if configured is not None:
        monkeypatch.setenv("GEMINI_MODEL", configured)
    mock_sdk.return_value.aio.models.generate_content.return_value = completion(
        "answer"
    )
    provider = GeminiProvider()

    asyncio.run(provider.generate("prompt"))

    assert provider.model == expected
    assert (
        mock_sdk.return_value.aio.models.generate_content.await_args.kwargs["model"]
        == expected
    )


@pytest.mark.parametrize("prompt", ["", " \n\t"])
def test_rejects_blank_prompt_before_request(mock_sdk: MagicMock, prompt: str) -> None:
    with pytest.raises(ValueError, match="prompt"):
        asyncio.run(GeminiProvider().generate(prompt))

    mock_sdk.return_value.aio.models.generate_content.assert_not_awaited()


def test_rejects_non_string_prompt_before_request(mock_sdk: MagicMock) -> None:
    with pytest.raises(TypeError, match="prompt"):
        asyncio.run(GeminiProvider().generate(None))  # type: ignore[arg-type]

    mock_sdk.return_value.aio.models.generate_content.assert_not_awaited()


def test_converts_custom_timeout_to_milliseconds(
    mock_sdk: MagicMock, mock_http_client: MagicMock
) -> None:
    GeminiProvider(timeout=5.0005)

    assert mock_sdk.call_args.kwargs["http_options"].timeout == 5001
    mock_http_client.assert_called_once_with(timeout=5.0005)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_timeout(
    mock_sdk: MagicMock, mock_http_client: MagicMock, timeout: float
) -> None:
    with pytest.raises(LLMConfigurationError, match="timeout"):
        GeminiProvider(timeout=timeout)

    mock_sdk.assert_not_called()
    mock_http_client.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
def test_context_manager_closes_all_clients(
    mock_sdk: MagicMock, mock_http_client: MagicMock, outcome: str
) -> None:
    mock_sdk.return_value.aio.models.generate_content.return_value = completion(
        "answer"
    )
    if outcome == "error":
        mock_sdk.return_value.aio.models.generate_content.side_effect = (
            httpx.ReadTimeout("mock timeout")
        )
    elif outcome == "cancelled":
        mock_sdk.return_value.aio.models.generate_content.side_effect = (
            asyncio.CancelledError()
        )

    async def run() -> None:
        async with GeminiProvider() as provider:
            await provider.generate("prompt")

    if outcome == "error":
        with pytest.raises(LLMTimeoutError):
            asyncio.run(run())
    elif outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
    else:
        asyncio.run(run())

    mock_sdk.return_value.aio.aclose.assert_awaited_once()
    mock_http_client.return_value.aclose.assert_awaited_once()
    mock_sdk.return_value.close.assert_called_once()


def test_explicit_close(mock_sdk: MagicMock, mock_http_client: MagicMock) -> None:
    asyncio.run(GeminiProvider().aclose())

    mock_sdk.return_value.aio.aclose.assert_awaited_once()
    mock_http_client.return_value.aclose.assert_awaited_once()
    mock_sdk.return_value.close.assert_called_once()


@pytest.mark.parametrize("failing_client", ["sdk", "http"])
def test_cleanup_continues_when_a_client_close_fails(
    mock_sdk: MagicMock, mock_http_client: MagicMock, failing_client: str
) -> None:
    if failing_client == "sdk":
        mock_sdk.return_value.aio.aclose.side_effect = RuntimeError(
            "mock close failure"
        )
    else:
        mock_http_client.return_value.aclose.side_effect = RuntimeError(
            "mock close failure"
        )

    with pytest.raises(RuntimeError, match="mock close failure"):
        asyncio.run(GeminiProvider().aclose())

    mock_sdk.return_value.aio.aclose.assert_awaited_once()
    mock_http_client.return_value.aclose.assert_awaited_once()
    mock_sdk.return_value.close.assert_called_once()
