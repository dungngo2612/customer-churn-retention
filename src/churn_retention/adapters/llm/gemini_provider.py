"""Asynchronous text generation using the Google GenAI SDK."""

import math
import os
from types import TracebackType
from typing import Self

import httpx
from google import genai
from google.genai import errors, types

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiProvider:
    """Read GEMINI_API_KEY and GEMINI_MODEL when the provider is created.

    Requests have a 30-second timeout by default. SDK retries and automatic
    continuation are disabled. Use an async context manager or call aclose()
    to release the SDK's resources and the provider-owned HTTP client.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY must be set and non-empty.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise LLMConfigurationError("timeout must be a positive, finite number.")

        self.model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
        # Pin the HTTP backend so optional aiohttp installations do not change
        # which transport exceptions callers receive through this adapter.
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self._client = genai.Client(
            api_key=api_key,
            enterprise=False,
            http_options=types.HttpOptions(
                timeout=math.ceil(timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_async_client=self._http_client,
            ),
        )

    async def generate(self, prompt: str) -> str:
        """Generate text, mapping SDK failures to the shared provider errors."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty.")

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(automatic_continuation=False),
            )
        except (httpx.TimeoutException, TimeoutError):
            raise LLMTimeoutError("Gemini request timed out.") from None
        except (httpx.TransportError, ConnectionError):
            raise LLMConnectionError("Could not connect to Gemini.") from None
        except errors.APIError as exc:
            # Never expose SDK messages, response bodies, or request headers.
            status_code = exc.code if isinstance(exc.code, int) else None
            if status_code == 429:
                raise LLMRateLimitError("Gemini rate limit or quota exceeded (HTTP 429).") from None
            if status_code in (408, 504):
                raise LLMTimeoutError("Gemini request timed out.") from None
            message = "Gemini API request failed."
            if status_code is not None:
                message = f"Gemini API request failed (HTTP {status_code})."
            raise LLMAPIError(message, status_code=status_code) from None

        content = response.text
        if content is None or not content.strip():
            raise LLMEmptyResponseError("Gemini returned empty response content.")
        return content

    async def aclose(self) -> None:
        """Close both SDK clients and the HTTP client owned by this provider."""
        try:
            await self._client.aio.aclose()
        finally:
            try:
                # The SDK does not close HTTP clients supplied by its caller.
                await self._http_client.aclose()
            finally:
                self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
