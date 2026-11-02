import asyncio
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

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
from churn_retention.modules.scoring import llm_router
from churn_retention.modules.scoring.llm_router import (
    LLMRouter,
    LLMRouterError,
    groq_gemini_router,
)

SECRET = "sensitive-key-and-customer-data"


@dataclass
class Timer:
    when: float
    callback: Callable[..., None]
    args: tuple[object, ...]
    cancelled: bool = False
    fired: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class Clock:
    """Only router deadlines use this clock; asyncio scheduling stays real."""

    def __init__(self) -> None:
        self.now = 0.0
        self.timers: list[Timer] = []

    def call_later(
        self, delay: float, callback: Callable[..., None], *args: object
    ) -> Timer:
        timer = Timer(self.now + delay, callback, args)
        self.timers.append(timer)
        return timer

    def advance(self, seconds: float) -> None:
        target = self.now + seconds
        for timer in sorted(self.timers, key=lambda t: t.when):
            if not timer.cancelled and not timer.fired and timer.when <= target:
                self.now = timer.when
                timer.fired = True
                timer.callback(*timer.args)
        self.now = target


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    clock = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: clock)
    yield clock
    assert not [t for t in clock.timers if not t.cancelled and not t.fired]


class Provider:
    """Controllable provider with observable cancellation and client ownership."""

    def __init__(self, clock: Clock, *, delayed_cleanup: bool = False) -> None:
        self.clock = clock
        self.response: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.delayed_cleanup = delayed_cleanup
        self.cleanup_error: Exception | None = None
        self.calls: list[tuple[str, float]] = []
        self.active = 0
        self.closed = False
        self.close_count = 0

    async def generate(self, prompt: str) -> str:
        assert not self.closed
        self.calls.append((prompt, self.clock.now))
        self.active += 1
        self.started.set()
        try:
            return await self.response
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.cleanup_started.set()
            if self.delayed_cleanup:
                await self.cleanup_release.wait()
            self.active -= 1
            self.finished.set()
            if self.cleanup_error is not None:
                raise self.cleanup_error

    def succeed(self, text: str = "answer") -> None:
        self.response.set_result(text)

    def fail(self, error: Exception) -> None:
        self.response.set_exception(error)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self.active == 0, "Client closed while provider still uses it"
        self.closed = True
        self.close_count += 1


async def settle() -> None:
    for _ in range(12):
        await asyncio.sleep(0)


def run_case(case: Coroutine[Any, Any, None]) -> None:
    async def checked() -> None:
        loop = asyncio.get_running_loop()
        unhandled: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        # A real-clock watchdog catches broken test waits, never simulates hedge.
        async with asyncio.timeout(5):
            await case
        await settle()
        assert not unhandled
        assert not [
            task
            for task in asyncio.all_tasks()
            if task.get_name().startswith("llm-router:") and not task.done()
        ]

    asyncio.run(checked())


def test_fast_primary_does_not_call_fallback(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed("  Qwen\n")
        async with LLMRouter(primary, fallback) as router:
            provider: LLMProvider = router
            assert await provider.generate("prompt") == "  Qwen\n"
        assert primary.calls == [("prompt", 0)]
        assert not fallback.calls
        assert not primary.closed and not fallback.closed  # Borrowed clients.

    run_case(case())


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(SECRET),
        LLMRateLimitError(SECRET),
        LLMTimeoutError(SECRET),
        LLMConnectionError(SECRET),
        LLMConfigurationError(SECRET),
        LLMAPIError(SECRET),
    ],
)
def test_primary_error_starts_fallback_immediately(
    clock: Clock, error: Exception
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.fail(error)
        fallback.succeed("Gemini")
        async with LLMRouter(primary, fallback) as router:
            assert await router.generate("prompt") == "Gemini"
        assert fallback.calls == [("prompt", 0)]
        assert clock.now == 0

    run_case(case())


def test_hedge_at_60_seconds_keeps_primary_running(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(30)
            await settle()
            assert not fallback.calls and primary.active == 1
            clock.advance(29)
            await settle()
            assert not fallback.calls
            clock.advance(1)
            await fallback.started.wait()
            assert fallback.calls == [("prompt", 60)]
            assert primary.active == 1 and not primary.cancelled.is_set()
            assert not request.done()
            primary.succeed("Qwen")
            assert await request == "Qwen"

    run_case(case())


@pytest.mark.parametrize("winner", ["primary", "fallback"])
@pytest.mark.parametrize("late_failure", [False, True])
def test_winner_returns_before_loser_cleanup(
    clock: Clock, winner: str, late_failure: bool
) -> None:
    async def case() -> None:
        primary = Provider(clock, delayed_cleanup=winner == "fallback")
        fallback = Provider(clock, delayed_cleanup=winner == "primary")
        winning = primary if winner == "primary" else fallback
        losing = fallback if winner == "primary" else primary
        if late_failure:
            losing.cleanup_error = RuntimeError(SECRET)
        router = LLMRouter(primary, fallback)
        try:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            winning.succeed(winner)
            await settle()
            assert request.done() and await request == winner
            await losing.cleanup_started.wait()
            assert losing.cancelled.is_set() and not losing.finished.is_set()
            assert losing.active == 1
        finally:
            losing.cleanup_release.set()
            await router.aclose()
        assert losing.finished.is_set()

    run_case(case())


@pytest.mark.parametrize("failing", ["primary", "fallback"])
def test_parallel_error_keeps_waiting_for_other_provider(
    clock: Clock, failing: str
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            loser = primary if failing == "primary" else fallback
            winner = fallback if failing == "primary" else primary
            loser.fail(LLMConnectionError(SECRET))
            await settle()
            assert not request.done()
            winner.succeed("survivor")
            assert await request == "survivor"

    run_case(case())


def test_both_failures_are_distinct_and_sanitized(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.fail(LLMRateLimitError(SECRET))
        fallback.fail(LLMAPIError(SECRET, status_code=503))
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate(SECRET)
        error = caught.value
        assert isinstance(error.primary_error, LLMRateLimitError)
        assert isinstance(error.fallback_error, LLMAPIError)
        assert error.fallback_error.status_code == 503
        assert not error.deadline_exceeded
        assert SECRET not in str(error) + repr(error.primary_error) + repr(
            error.fallback_error
        )
        assert SECRET not in "".join(traceback.format_exception(error))
        assert error.__cause__ is None
        assert SECRET not in caplog.text

    run_case(case())


def test_unknown_failure_messages_and_invalid_http_codes_are_not_exposed(
    clock: Clock,
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.fail(ValueError(SECRET))
        fallback.fail(LLMAPIError(SECRET, status_code=cast(int, SECRET)))
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate("prompt")
        assert type(caught.value.primary_error) is LLMProviderError
        assert isinstance(caught.value.fallback_error, LLMAPIError)
        assert caught.value.fallback_error.status_code is None
        assert SECRET not in repr(vars(caught.value))

    run_case(case())


def test_fallback_is_called_once_when_primary_fails_after_hedge(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            primary.fail(LLMRateLimitError(SECRET))
            clock.advance(1)
            await settle()
            assert len(fallback.calls) == 1
            fallback.succeed("Gemini")
            assert await request == "Gemini"
        assert len(fallback.calls) == 1

    run_case(case())


@pytest.mark.parametrize("hedged", [False, True])
def test_caller_cancellation_cancels_all_started_providers(
    clock: Clock, hedged: bool
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            if hedged:
                clock.advance(60)
                await fallback.started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        assert primary.cancelled.is_set() and primary.finished.is_set()
        assert fallback.cancelled.is_set() == hedged
        assert bool(fallback.calls) == hedged

    run_case(case())


def test_cancel_before_request_starts_leaks_no_task(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            task = asyncio.create_task(router.generate("prompt"))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not primary.calls and not fallback.calls

    run_case(case())


def test_shutdown_cancels_active_request_and_rejects_new_requests(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        router = LLMRouter(primary, fallback)
        request = asyncio.create_task(router.generate("prompt"))
        await primary.started.wait()
        await router.aclose()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(LLMProviderError, match="closed"):
            await router.generate("new")
        with pytest.raises(LLMProviderError, match="closed"):
            async with router:
                pytest.fail("A closed router must not be entered")
        await router.aclose()
        assert not primary.closed and not fallback.closed

    run_case(case())


def test_deadline_limits_two_hanging_providers(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            clock.advance(60)
            with pytest.raises(LLMRouterError) as caught:
                await request
            assert caught.value.deadline_exceeded
            assert isinstance(caught.value.primary_error, LLMTimeoutError)
            assert isinstance(caught.value.fallback_error, LLMTimeoutError)
        assert primary.cancelled.is_set() and fallback.cancelled.is_set()

    run_case(case())


def test_deadline_preserves_failure_of_provider_already_finished(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.fail(LLMRateLimitError(SECRET))
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await fallback.started.wait()
            clock.advance(120)
            with pytest.raises(LLMRouterError) as caught:
                await request
        assert isinstance(caught.value.primary_error, LLMRateLimitError)
        assert isinstance(caught.value.fallback_error, LLMTimeoutError)
        assert len(fallback.calls) == 1

    run_case(case())


@pytest.mark.parametrize("invalid", [None, "", " \t\n", 42])
def test_invalid_primary_response_triggers_fallback(
    clock: Clock, invalid: object
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(cast(str, invalid))
        fallback.succeed("valid")
        async with LLMRouter(primary, fallback) as router:
            assert await router.generate("prompt") == "valid"
        assert fallback.calls == [("prompt", 0)]

    run_case(case())


def test_invalid_fallback_response_does_not_defeat_pending_primary(
    clock: Clock,
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            fallback.succeed(" ")
            await settle()
            assert not request.done()
            primary.succeed("Qwen")
            assert await request == "Qwen"

    run_case(case())


def test_both_empty_responses_report_typed_failures(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(" ")
        fallback.succeed("")
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate("prompt")
        assert isinstance(caught.value.primary_error, LLMEmptyResponseError)
        assert isinstance(caught.value.fallback_error, LLMEmptyResponseError)

    run_case(case())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_deadline_and_caller_cancellation_do_not_wait_for_slow_cleanup(
    clock: Clock, cancel_caller: bool
) -> None:
    async def case() -> None:
        primary = Provider(clock, delayed_cleanup=True)
        fallback = Provider(clock)
        router = LLMRouter(primary, fallback)
        request = asyncio.create_task(router.generate("prompt"))
        try:
            await primary.started.wait()
            if cancel_caller:
                request.cancel()
            else:
                clock.advance(60)
                await fallback.started.wait()
                clock.advance(60)
            await settle()
            assert request.done()
            if cancel_caller:
                with pytest.raises(asyncio.CancelledError):
                    await request
            else:
                with pytest.raises(LLMRouterError) as caught:
                    await request
                assert caught.value.deadline_exceeded
            assert primary.cleanup_started.is_set() and not primary.finished.is_set()
        finally:
            primary.cleanup_release.set()
            await router.aclose()

    run_case(case())


@pytest.mark.parametrize("first", ["primary", "fallback"])
def test_first_valid_completion_wins_when_both_finish_in_one_turn(
    clock: Clock, first: str
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            if first == "primary":
                primary.succeed("primary")
                fallback.succeed("fallback")
            else:
                fallback.succeed("fallback")
                primary.succeed("primary")
            assert await request == first

    run_case(case())


def test_independent_concurrent_requests(clock: Clock) -> None:
    async def case() -> None:
        p1, p2, f1, f2 = [Provider(clock) for _ in range(4)]
        primary = AsyncMock(side_effect=lambda prompt: None)
        fallback = AsyncMock(side_effect=lambda prompt: None)

        async def primary_generate(prompt: str) -> str:
            return await {"one": p1, "two": p2}[prompt].generate(prompt)

        async def fallback_generate(prompt: str) -> str:
            return await {"one": f1, "two": f2}[prompt].generate(prompt)

        primary.generate = AsyncMock(side_effect=primary_generate)
        fallback.generate = AsyncMock(side_effect=fallback_generate)
        async with LLMRouter(primary, fallback) as router:
            one = asyncio.create_task(router.generate("one"))
            two = asyncio.create_task(router.generate("two"))
            await p1.started.wait()
            await p2.started.wait()
            p2.fail(LLMTimeoutError(SECRET))
            await f2.started.wait()
            f2.succeed("Gemini two")
            assert await two == "Gemini two"
            assert p1.active == 1 and not p1.cancelled.is_set()
            p1.succeed("Qwen one")
            assert await one == "Qwen one"
        assert not f1.calls

    run_case(case())


@pytest.mark.parametrize("invalid", ["", " \n\t", None])
def test_invalid_prompt_makes_no_provider_calls(clock: Clock, invalid: object) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises((TypeError, ValueError)):
                await router.generate(cast(str, invalid))
        assert not primary.calls and not fallback.calls

    run_case(case())


@pytest.mark.parametrize(
    ("hedge", "total"),
    [
        (-1, 120),
        (float("nan"), 120),
        (float("inf"), 120),
        (60, 0),
        (60, float("nan")),
        (60, float("inf")),
        (60, 60),
        (60, 30),
    ],
)
def test_invalid_router_timeouts_are_rejected(
    clock: Clock, hedge: float, total: float
) -> None:
    async def case() -> None:
        with pytest.raises(LLMConfigurationError):
            LLMRouter(
                Provider(clock),
                Provider(clock),
                hedge_delay_seconds=hedge,
                total_timeout_seconds=total,
            )

    run_case(case())


def test_custom_hedge_delay_and_zero_delay(clock: Clock) -> None:
    async def case() -> None:
        for delay in (0, 10):
            primary, fallback = Provider(clock), Provider(clock)
            async with LLMRouter(
                primary, fallback, hedge_delay_seconds=delay
            ) as router:
                request = asyncio.create_task(router.generate("prompt"))
                await primary.started.wait()
                clock.advance(delay)
                await fallback.started.wait()
                fallback.succeed()
                assert await request == "answer"
                assert fallback.calls[0][1] - primary.calls[0][1] == delay

    run_case(case())


def test_self_cancelled_primary_falls_back(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.response.cancel()
        fallback.succeed("Gemini")
        async with LLMRouter(primary, fallback) as router:
            assert await router.generate("prompt") == "Gemini"

    run_case(case())


def test_factory_groq_sdk_timeout_exceeds_hedge(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "router-unit-test-key")

    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        with (
            patch("churn_retention.adapters.llm.groq_provider.AsyncGroq") as sdk,
            patch(
                "churn_retention.adapters.llm.gemini_provider.GeminiProvider",
                return_value=fallback,
            ) as gemini,
        ):

            async def groq_call(**kwargs: Any) -> None:
                await primary.generate(kwargs["messages"][0]["content"])

            sdk.return_value.chat.completions.create = AsyncMock(side_effect=groq_call)
            sdk.return_value.close = AsyncMock()
            async with groq_gemini_router() as router:
                assert router.hedge_delay_seconds == 60
                assert router.total_timeout_seconds == 120
                sdk.assert_called_once_with(
                    api_key="router-unit-test-key", timeout=90.0, max_retries=0
                )
                gemini.assert_called_once_with(timeout=30.0)
                request = asyncio.create_task(router.generate("prompt"))
                await primary.started.wait()
                clock.advance(30)
                await settle()
                assert primary.active == 1 and not fallback.calls
                clock.advance(30)
                await fallback.started.wait()
                assert not primary.cancelled.is_set()
                fallback.succeed("Gemini")
                assert await request == "Gemini"
            assert primary.finished.is_set() and fallback.closed
            sdk.return_value.close.assert_awaited_once()

    run_case(case())


@pytest.mark.parametrize("primary_timeout", [30, 60, -1, float("inf"), float("nan")])
def test_factory_rejects_primary_timeout_that_disables_hedge(
    clock: Clock, primary_timeout: float
) -> None:
    async def case() -> None:
        with patch(
            "churn_retention.adapters.llm.groq_provider.GroqProvider"
        ) as primary:
            with pytest.raises(LLMConfigurationError):
                async with groq_gemini_router(primary_timeout_seconds=primary_timeout):
                    pytest.fail("Invalid configuration must not create clients")
            primary.assert_not_called()

    run_case(case())


def test_factory_supports_custom_timeouts(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        with (
            patch(
                "churn_retention.adapters.llm.groq_provider.GroqProvider",
                return_value=primary,
            ) as groq,
            patch(
                "churn_retention.adapters.llm.gemini_provider.GeminiProvider",
                return_value=fallback,
            ) as gemini,
        ):
            async with groq_gemini_router(
                hedge_delay_seconds=10,
                total_timeout_seconds=40,
                primary_timeout_seconds=20,
                fallback_timeout_seconds=7,
            ) as router:
                assert router.hedge_delay_seconds == 10
                assert router.total_timeout_seconds == 40
                groq.assert_called_once_with(timeout=20)
                gemini.assert_called_once_with(timeout=7)
        assert primary.closed and fallback.closed

    run_case(case())


def test_factory_never_closes_busy_clients_even_if_shutdown_is_cancelled_twice(
    clock: Clock,
) -> None:
    async def case() -> None:
        primary = Provider(clock, delayed_cleanup=True)
        fallback = Provider(clock)
        with (
            patch(
                "churn_retention.adapters.llm.groq_provider.GroqProvider",
                return_value=primary,
            ),
            patch(
                "churn_retention.adapters.llm.gemini_provider.GeminiProvider",
                return_value=fallback,
            ),
        ):
            manager = groq_gemini_router()
            router = await manager.__aenter__()
            request = asyncio.create_task(router.generate("prompt"))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            fallback.succeed("Gemini")
            assert await request == "Gemini"
            await primary.cleanup_started.wait()
            closing = asyncio.create_task(manager.__aexit__(None, None, None))
            try:
                await settle()
                assert not closing.done() and not primary.closed and not fallback.closed
                closing.cancel()
                await settle()
                closing.cancel()
                await settle()
                assert not closing.done()
                assert primary.active == 1 and not primary.closed
            finally:
                primary.cleanup_release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert primary.finished.is_set() and primary.closed and fallback.closed
            assert primary.close_count == fallback.close_count == 1

    run_case(case())


def test_factory_closes_primary_if_fallback_initialization_fails(clock: Clock) -> None:
    async def case() -> None:
        primary = Provider(clock)
        with (
            patch(
                "churn_retention.adapters.llm.groq_provider.GroqProvider",
                return_value=primary,
            ),
            patch(
                "churn_retention.adapters.llm.gemini_provider.GeminiProvider",
                side_effect=LLMConfigurationError("Missing fallback key"),
            ),
            pytest.raises(LLMConfigurationError),
        ):
            async with groq_gemini_router():
                pytest.fail("Missing fallback configuration must fail")
        assert primary.closed and primary.close_count == 1

    run_case(case())
