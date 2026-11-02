"""Primary-first LLM routing with a delayed, concurrent fallback."""

import asyncio
import inspect
import math
from asyncio import get_running_loop
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Self, TypeVar

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from churn_retention.contracts.llm import ProviderName

_T = TypeVar("_T")
_Role = Literal["primary", "fallback"]
ResponseValidator = Callable[[str], None]


class LLMRouterError(LLMProviderError):
    """Both providers failed; attributes contain sanitized, typed failures."""

    def __init__(
        self,
        primary_error: LLMProviderError,
        fallback_error: LLMProviderError,
        *,
        deadline_exceeded: bool = False,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        self.deadline_exceeded = deadline_exceeded
        reason = (
            "LLM request deadline exceeded" if deadline_exceeded else "Both LLM providers failed"
        )
        super().__init__(
            f"{reason}: primary={type(primary_error).__name__}; "
            f"fallback={type(fallback_error).__name__}."
        )

    def to_dict(self) -> dict[str, object]:
        def details(error: LLMProviderError) -> dict[str, object]:
            result: dict[str, object] = {"type": type(error).__name__}
            if isinstance(error, LLMResponseValidationError):
                result["code"] = error.code
            if isinstance(error, LLMAPIError):
                result["status_code"] = error.status_code
            return result

        return {
            "error": "llm_router_failed",
            "deadline_exceeded": self.deadline_exceeded,
            "primary": details(self.primary_error),
            "fallback": details(self.fallback_error),
        }


def _safe_error(error: Exception) -> LLMProviderError:
    """Copy only allowlisted categories and HTTP codes, never exception text."""
    message = "Provider request failed."
    if isinstance(error, LLMResponseValidationError):
        return LLMResponseValidationError("Provider response failed validation.", code=error.code)
    if isinstance(error, LLMAPIError):
        code = error.status_code
        code = code if type(code) is int and 100 <= code <= 599 else None
        return LLMAPIError(message, status_code=code)
    for category in (
        LLMConfigurationError,
        LLMConnectionError,
        LLMEmptyResponseError,
        LLMRateLimitError,
        LLMTimeoutError,
    ):
        if isinstance(error, category):
            return category(message)
    return LLMProviderError(message)


def _validate_time(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise LLMConfigurationError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}."
        )


def _validate_deadline(hedge_delay_seconds: float, total_timeout_seconds: float) -> None:
    _validate_time(hedge_delay_seconds, "hedge_delay_seconds", allow_zero=True)
    _validate_time(total_timeout_seconds, "total_timeout_seconds")
    if total_timeout_seconds <= hedge_delay_seconds:
        raise LLMConfigurationError("total_timeout_seconds must exceed hedge_delay_seconds.")


@dataclass(frozen=True)
class _Outcome:
    role: _Role
    content: str | None = None
    error: LLMProviderError | None = None


_Event = _Outcome | Literal["hedge", "deadline"]


@dataclass(frozen=True)
class RoutedResponse:
    content: str
    role: _Role
    provider: ProviderName | None


class LLMRouter:
    """Borrow providers; own all request tasks until completion or aclose().

    generate() never waits for a losing provider's cancellation cleanup. Keep
    this router inside the providers' lifetimes, preferably using async with.
    aclose() drains outstanding work before clients can safely be closed.
    Providers must support cooperative asyncio cancellation and nonblocking I/O.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        *,
        hedge_delay_seconds: float = 60.0,
        total_timeout_seconds: float = 120.0,
        primary_provider: ProviderName | None = None,
        fallback_provider: ProviderName | None = None,
    ) -> None:
        _validate_deadline(hedge_delay_seconds, total_timeout_seconds)
        if any(
            name is not None and name not in ("Groq", "Gemini")
            for name in (primary_provider, fallback_provider)
        ):
            raise LLMConfigurationError("Provider labels must be Groq, Gemini, or None.")
        self._primary = primary
        self._fallback = fallback
        self._primary_provider = primary_provider
        self._fallback_provider = fallback_provider
        self.hedge_delay_seconds = hedge_delay_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self._tasks: set[asyncio.Task[object]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    def _track(self, task: asyncio.Task[_T]) -> asyncio.Task[_T]:
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve failures even after the caller has returned.

    @staticmethod
    def _cancel(task: asyncio.Task[object]) -> None:
        # A second cancel can interrupt the provider's asynchronous cleanup.
        if not task.done() and not task.cancelling():
            task.cancel()

    async def _invoke(
        self,
        role: _Role,
        provider: LLMProvider,
        prompt: str,
        events: asyncio.Queue[_Event],
        response_validator: ResponseValidator | None,
    ) -> None:
        try:
            content = await provider.generate(prompt)
            if not isinstance(content, str) or not content.strip():
                raise LLMEmptyResponseError("Provider returned no usable text.")
            if response_validator is not None:
                try:
                    response_validator(content)
                except LLMResponseValidationError:
                    raise
                except Exception:  # noqa: BLE001 -- Never expose arbitrary validator errors.
                    raise LLMResponseValidationError(code="validator_error") from None
        except asyncio.CancelledError:
            events.put_nowait(_Outcome(role, error=LLMProviderError("Provider request cancelled.")))
            raise
        except Exception as error:  # noqa: BLE001 -- Any provider failure must permit fallback.
            events.put_nowait(_Outcome(role, error=_safe_error(error)))
        else:
            # Publish before finishing the task so queue order records which
            # valid response completed first, even within one event-loop turn.
            events.put_nowait(_Outcome(role, content=content))

    async def _race(
        self, prompt: str, response_validator: ResponseValidator | None
    ) -> RoutedResponse:
        events: asyncio.Queue[_Event] = asyncio.Queue()
        loop = get_running_loop()
        hedge = loop.call_later(self.hedge_delay_seconds, events.put_nowait, "hedge")
        deadline = loop.call_later(self.total_timeout_seconds, events.put_nowait, "deadline")
        children: list[asyncio.Task[None]] = []
        failures: dict[_Role, LLMProviderError] = {}
        fallback_started = False

        def start(role: _Role, provider: LLMProvider) -> None:
            children.append(
                self._track(
                    asyncio.create_task(
                        self._invoke(role, provider, prompt, events, response_validator),
                        name=f"llm-router:{role}",
                    )
                )
            )

        def start_fallback() -> None:
            nonlocal fallback_started
            if not fallback_started:
                fallback_started = True
                hedge.cancel()
                start("fallback", self._fallback)

        try:
            start("primary", self._primary)
            while True:
                event = await events.get()
                if event == "hedge":
                    start_fallback()
                elif event == "deadline":
                    raise LLMRouterError(
                        failures.get(
                            "primary",
                            LLMTimeoutError("Primary exceeded the request deadline."),
                        ),
                        failures.get(
                            "fallback",
                            LLMTimeoutError("Fallback exceeded the request deadline."),
                        ),
                        deadline_exceeded=True,
                    ) from None
                else:
                    assert isinstance(event, _Outcome)
                    if event.content is not None:
                        provider_name = (
                            self._primary_provider
                            if event.role == "primary"
                            else self._fallback_provider
                        )
                        return RoutedResponse(event.content, event.role, provider_name)
                    assert event.error is not None
                    failures[event.role] = event.error
                    if event.role == "primary":
                        start_fallback()
                    if len(failures) == 2:
                        raise LLMRouterError(failures["primary"], failures["fallback"]) from None
        finally:
            hedge.cancel()
            deadline.cancel()
            for child in children:
                self._cancel(child)

    async def generate(
        self, prompt: str, *, response_validator: ResponseValidator | None = None
    ) -> str:
        """Keep the default string API; validation is optional and per request."""
        return (await self.generate_result(prompt, response_validator=response_validator)).content

    async def generate_result(
        self, prompt: str, *, response_validator: ResponseValidator | None = None
    ) -> RoutedResponse:
        """Return request-local winner metadata; validators synchronously raise to reject."""
        if self._closed:
            raise LLMProviderError("LLM router is closed.")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty.")
        if response_validator is not None and (
            not callable(response_validator)
            or inspect.iscoroutinefunction(response_validator)
            or inspect.iscoroutinefunction(
                getattr(response_validator, "__call__", None)  # noqa: B004 -- Inspect async callable instances, not callability.
            )
        ):
            raise LLMConfigurationError("response_validator must be a synchronous callable.")
        request = self._track(
            asyncio.create_task(self._race(prompt, response_validator), name="llm-router:request")
        )
        return await request

    async def _drain(self) -> None:
        while self._tasks:
            tasks = set(self._tasks)
            for task in tasks:
                if task.done():
                    # gather() may return immediately for already-done tasks,
                    # before their removal callbacks get an event-loop turn.
                    self._task_done(task)
                else:
                    self._cancel(task)
            pending = {task for task in tasks if not task.done()}
            # Only explicit shutdown waits for tasks; generate() never does.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel and drain requests without closing borrowed providers.

        During shutdown, defer caller cancellation until work has stopped, so
        enclosing provider context managers cannot close clients still in use.
        """
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain(), name="llm-router:close")
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def __aenter__(self) -> Self:
        if self._closed:
            raise LLMProviderError("LLM router is closed.")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


@asynccontextmanager
async def groq_gemini_router(
    *,
    hedge_delay_seconds: float = 60.0,
    total_timeout_seconds: float = 120.0,
    primary_timeout_seconds: float = 90.0,
    fallback_timeout_seconds: float = 30.0,
) -> AsyncIterator[LLMRouter]:
    """Own Groq/Gemini clients with a primary timeout longer than the hedge."""
    _validate_deadline(hedge_delay_seconds, total_timeout_seconds)
    _validate_time(primary_timeout_seconds, "primary_timeout_seconds")
    _validate_time(fallback_timeout_seconds, "fallback_timeout_seconds")
    if primary_timeout_seconds <= hedge_delay_seconds:
        raise LLMConfigurationError("primary_timeout_seconds must exceed hedge_delay_seconds.")

    # Keep concrete SDK dependencies out of the injected router's import path.
    from churn_retention.adapters.llm.gemini_provider import GeminiProvider
    from churn_retention.adapters.llm.groq_provider import GroqProvider

    async with (
        GroqProvider(timeout=primary_timeout_seconds) as primary,
        GeminiProvider(timeout=fallback_timeout_seconds) as fallback,
        LLMRouter(
            primary,
            fallback,
            hedge_delay_seconds=hedge_delay_seconds,
            total_timeout_seconds=total_timeout_seconds,
            primary_provider="Groq",
            fallback_provider="Gemini",
        ) as router,
    ):
        yield router
