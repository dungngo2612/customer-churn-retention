"""Asynchronous text generation using the Groq Python SDK."""

import math
import os
from types import TracebackType
from typing import Self

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    RateLimitError,
)

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)

DEFAULT_MODEL = "qwen/qwen3.8-27b"


class GroqProvider:
    """Read GROQ_API_KEY and GROQ_MODEL when the provider is created.

    Requests have a 30-second timeout by default. Automatic SDK retries are
    disabled so callers can decide how to handle rate limits and transient errors.
    Use an async context manager or call aclose() to release HTTP connections.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError("GROQ_API_KEY must be set and non-empty.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise LLMConfigurationError("timeout must be a positive, finite number.")

        self.model = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
        self._client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=0)

    async def generate(self, prompt: str) -> str:
        """Generate text, mapping SDK failures to the shared provider errors."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty.")

        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        # Timeout is a subclass of APIConnectionError; handle it first.
        except APITimeoutError:
            raise LLMTimeoutError("Groq request timed out.") from None
        except APIConnectionError:
            raise LLMConnectionError("Could not connect to Groq.") from None
        except RateLimitError:
            raise LLMRateLimitError("Groq rate limit exceeded (HTTP 429).") from None
        except APIStatusError as exc:
            raise LLMAPIError(
                f"Groq API request failed (HTTP {exc.status_code}).",
                status_code=exc.status_code,
            ) from None
        except APIError:
            # Do not expose SDK messages, response bodies, or request headers.
            raise LLMAPIError("Groq API request failed.") from None

        if not completion.choices:
            raise LLMEmptyResponseError("Groq returned no response choices.")
        content = completion.choices[0].message.content
        if content is None or not content.strip():
            raise LLMEmptyResponseError("Groq returned empty response content.")
        return content

    async def aclose(self) -> None:
        """Close the underlying asynchronous SDK client."""
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
