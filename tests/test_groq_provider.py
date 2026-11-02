import asyncio
import traceback
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from groq.types.chat import ChatCompletion, ChatCompletionMessage
from groq.types.chat.chat_completion import Choice

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
from churn_retention.adapters.llm.groq_provider import GroqProvider

FAKE_API_KEY = "unit-test-key-never-used-for-network"


@pytest.fixture(autouse=True)
def mock_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Every test replaces AsyncGroq so no real client or API request is used."""
    monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    with patch("churn_retention.adapters.llm.groq_provider.AsyncGroq") as sdk:
        sdk.return_value.chat.completions.create = AsyncMock()
        sdk.return_value.close = AsyncMock()
        yield sdk


def completion(content: str | None) -> ChatCompletion:
    return ChatCompletion(
        id="mock-completion",
        created=0,
        model="qwen/qwen3.8-27b",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )


def test_returns_content_and_awaits_sdk(mock_sdk: MagicMock) -> None:
    text = "  Khách hàng có nguy cơ rời bỏ.\n"
    mock_sdk.return_value.chat.completions.create.return_value = completion(text)
    provider: LLMProvider = GroqProvider()

    assert asyncio.run(provider.generate("Phân tích khách hàng")) == text
    mock_sdk.return_value.chat.completions.create.assert_awaited_once_with(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "Phân tích khách hàng"}],
    )
    mock_sdk.assert_called_once_with(api_key=FAKE_API_KEY, timeout=30.0, max_retries=0)


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_rejects_empty_content(mock_sdk: MagicMock, content: str | None) -> None:
    mock_sdk.return_value.chat.completions.create.return_value = completion(content)

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(GroqProvider().generate("prompt"))


def test_rejects_missing_choices(mock_sdk: MagicMock) -> None:
    response = completion("unused")
    response.choices = []
    mock_sdk.return_value.chat.completions.create.return_value = response

    with pytest.raises(LLMEmptyResponseError, match="no response choices"):
        asyncio.run(GroqProvider().generate("prompt"))


@pytest.mark.parametrize(
    ("failure", "expected_error", "status_code"),
    [
        ("rate_limit", LLMRateLimitError, 429),
        ("timeout", LLMTimeoutError, None),
        ("connection", LLMConnectionError, None),
        ("status", LLMAPIError, 400),
        ("status", LLMAPIError, 401),
        ("status", LLMAPIError, 500),
        ("api", LLMAPIError, None),
    ],
)
def test_maps_sdk_failures_without_exposing_secrets(
    mock_sdk: MagicMock,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    expected_error: type[LLMProviderError],
    status_code: int | None,
) -> None:
    request = httpx.Request(
        "POST",
        "https://groq.invalid/chat/completions",
        headers={"Authorization": f"Bearer {FAKE_API_KEY}"},
    )
    error: APIError
    if failure == "rate_limit":
        error = RateLimitError(
            FAKE_API_KEY,
            response=httpx.Response(429, request=request),
            body={"error": FAKE_API_KEY},
        )
    elif failure == "timeout":
        error = APITimeoutError(request=request)
    elif failure == "connection":
        error = APIConnectionError(message=FAKE_API_KEY, request=request)
    elif failure == "status":
        assert status_code is not None
        error = APIStatusError(
            FAKE_API_KEY,
            response=httpx.Response(status_code, request=request),
            body={"error": FAKE_API_KEY},
        )
    else:
        error = APIError(FAKE_API_KEY, request, body={"error": FAKE_API_KEY})
    mock_sdk.return_value.chat.completions.create.side_effect = error

    with pytest.raises(expected_error) as caught:
        asyncio.run(GroqProvider().generate("prompt"))

    assert type(caught.value) is expected_error
    if isinstance(caught.value, LLMAPIError):
        assert caught.value.status_code == status_code
    rendered_error = "".join(traceback.format_exception(caught.value))
    assert FAKE_API_KEY not in str(caught.value)
    assert FAKE_API_KEY not in rendered_error
    assert FAKE_API_KEY not in caplog.text
    mock_sdk.return_value.chat.completions.create.assert_awaited_once()


@pytest.mark.parametrize("api_key", [None, "", " \n\t"])
def test_requires_api_key_before_creating_client(
    monkeypatch: pytest.MonkeyPatch, mock_sdk: MagicMock, api_key: str | None
) -> None:
    if api_key is None:
        monkeypatch.delenv("GROQ_API_KEY")
    else:
        monkeypatch.setenv("GROQ_API_KEY", api_key)

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        GroqProvider()

    mock_sdk.assert_not_called()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "qwen/qwen3.8-27b"),
        ("", "qwen/qwen3.8-27b"),
        (" \t", "qwen/qwen3.8-27b"),
        ("custom/model", "custom/model"),
        ("  custom/model  ", "custom/model"),
    ],
)
def test_uses_configured_model_in_request(
    monkeypatch: pytest.MonkeyPatch,
    mock_sdk: MagicMock,
    configured: str | None,
    expected: str,
) -> None:
    if configured is not None:
        monkeypatch.setenv("GROQ_MODEL", configured)
    mock_sdk.return_value.chat.completions.create.return_value = completion("answer")
    provider = GroqProvider()

    asyncio.run(provider.generate("prompt"))

    assert provider.model == expected
    assert mock_sdk.return_value.chat.completions.create.await_args.kwargs["model"] == expected


@pytest.mark.parametrize("prompt", ["", " \n\t"])
def test_rejects_blank_prompt_before_request(mock_sdk: MagicMock, prompt: str) -> None:
    with pytest.raises(ValueError, match="prompt"):
        asyncio.run(GroqProvider().generate(prompt))

    mock_sdk.return_value.chat.completions.create.assert_not_awaited()


def test_rejects_non_string_prompt_before_request(mock_sdk: MagicMock) -> None:
    with pytest.raises(TypeError, match="prompt"):
        asyncio.run(GroqProvider().generate(None))  # type: ignore[arg-type]

    mock_sdk.return_value.chat.completions.create.assert_not_awaited()


def test_supports_custom_timeout(mock_sdk: MagicMock) -> None:
    GroqProvider(timeout=5.0)

    assert mock_sdk.call_args.kwargs["timeout"] == 5.0


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_timeout(mock_sdk: MagicMock, timeout: float) -> None:
    with pytest.raises(LLMConfigurationError, match="timeout"):
        GroqProvider(timeout=timeout)

    mock_sdk.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_context_manager_closes_client_even_on_error(mock_sdk: MagicMock, fail: bool) -> None:
    mock_sdk.return_value.chat.completions.create.return_value = completion("answer")
    if fail:
        mock_sdk.return_value.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request("POST", "https://groq.invalid")
        )

    async def run() -> None:
        async with GroqProvider() as provider:
            await provider.generate("prompt")

    if fail:
        with pytest.raises(LLMTimeoutError):
            asyncio.run(run())
    else:
        asyncio.run(run())

    mock_sdk.return_value.close.assert_awaited_once()


def test_explicit_close(mock_sdk: MagicMock) -> None:
    asyncio.run(GroqProvider().aclose())

    mock_sdk.return_value.close.assert_awaited_once()
