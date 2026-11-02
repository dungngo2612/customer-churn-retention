"""Small, SDK-independent contract for text generation providers."""

from typing import Protocol


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str:
        """Return non-empty text or raise an LLMProviderError."""
        ...


class LLMProviderError(RuntimeError):
    """Base error exposed by an LLM provider."""


class LLMConfigurationError(LLMProviderError):
    """Required provider configuration is missing or invalid."""


class LLMEmptyResponseError(LLMProviderError):
    """The provider returned no usable text."""


class LLMRateLimitError(LLMProviderError):
    """The provider rejected a request because of a rate limit."""


class LLMTimeoutError(LLMProviderError):
    """The provider request timed out."""


class LLMConnectionError(LLMProviderError):
    """The provider could not be reached."""


class LLMAPIError(LLMProviderError):
    """An API failure, optionally carrying a safe HTTP status code."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseValidationError(LLMProviderError):
    """A response failed content validation; code never contains raw response data."""

    def __init__(
        self,
        message: str = "Provider response failed validation.",
        *,
        code: str = "validator_error",
    ) -> None:
        super().__init__(message)
        allowed = {
            "invalid_json",
            "invalid_object",
            "invalid_keys",
            "invalid_prediction",
            "invalid_probability",
            "invalid_reason",
            "response_too_large",
            "validator_error",
        }
        self.code = code if isinstance(code, str) and code in allowed else "validator_error"
