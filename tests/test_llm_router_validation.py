import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from test_llm_router import Clock, Provider, run_case, settle

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMConfigurationError,
    LLMResponseValidationError,
)
from churn_retention.modules.scoring import llm_router
from churn_retention.modules.scoring.llm_response import parse_churn_response
from churn_retention.modules.scoring.llm_router import (
    LLMRouter,
    LLMRouterError,
    groq_gemini_router,
)

VALID = json.dumps(
    {
        "churn_prediction": 1,
        "churn_probability": 0.75,
        "reason": "Bằng chứng còn hạn chế.",
    }
)


def validate(text: str) -> None:
    parse_churn_response(text)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    value = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: value)
    yield value
    assert not [t for t in value.timers if not t.cancelled and not t.fired]


def test_default_string_behavior_accepts_non_json_unchanged(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed("  not JSON\n")
        async with LLMRouter(primary, fallback) as router:
            assert await router.generate("prompt") == "  not JSON\n"
        assert not fallback.calls

    run_case(case())


def test_validated_primary_wins_with_request_local_metadata(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await router.generate_result("prompt", response_validator=validate)
        assert result.content == VALID and result.role == "primary" and result.provider == "Groq"
        assert not fallback.calls

    run_case(case())


def test_invalid_primary_immediately_falls_back_once(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed("not JSON")
        fallback.succeed(VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await router.generate_result("prompt", response_validator=validate)
        assert result.provider == "Gemini" and result.role == "fallback"
        assert result.content == VALID
        assert primary.calls == fallback.calls == [("prompt", 0)]

    run_case(case())


def test_both_invalid_errors_keep_validation_codes(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed("bad JSON")
        fallback.succeed(json.dumps({"churn_prediction": 1, "churn_probability": 2, "reason": "x"}))
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate("prompt", response_validator=validate)
        error = caught.value
        assert isinstance(error.primary_error, LLMResponseValidationError)
        assert isinstance(error.fallback_error, LLMResponseValidationError)
        assert error.primary_error.code == "invalid_json"
        assert error.fallback_error.code == "invalid_probability"
        assert error.to_dict()["primary"] == {
            "type": "LLMResponseValidationError",
            "code": "invalid_json",
        }

    run_case(case())


def test_format_and_api_errors_remain_distinguishable_and_safe(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    async def case() -> None:
        secret = "PRIVATE_VALIDATION_API_PAYLOAD"
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(secret)
        fallback.fail(LLMAPIError(secret, status_code=503))
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate("prompt", response_validator=validate)
        assert isinstance(caught.value.primary_error, LLMResponseValidationError)
        assert isinstance(caught.value.fallback_error, LLMAPIError)
        assert caught.value.fallback_error.status_code == 503
        assert secret not in json.dumps(caught.value.to_dict()) + str(caught.value) + caplog.text

    run_case(case())


@pytest.mark.parametrize("invalid", ["primary", "fallback"])
def test_invalid_hedged_response_never_defeats_pending_valid_provider(
    clock: Clock, invalid: str
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            request = asyncio.create_task(
                router.generate_result("prompt", response_validator=validate)
            )
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            loser = primary if invalid == "primary" else fallback
            winner = fallback if invalid == "primary" else primary
            loser.succeed("```json\n{}\n```")
            await settle()
            assert not request.done() and not winner.cancelled.is_set()
            winner.succeed(VALID)
            assert (await request).role == ("fallback" if invalid == "primary" else "primary")
        assert len(primary.calls) == len(fallback.calls) == 1

    run_case(case())


def test_valid_hedged_winner_returns_before_loser_cleanup(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock, delayed_cleanup=True), Provider(clock)
        router = LLMRouter(primary, fallback)
        try:
            request = asyncio.create_task(
                router.generate_result("prompt", response_validator=validate)
            )
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            fallback.succeed(VALID)
            await settle()
            assert request.done() and (await request).role == "fallback"
            assert primary.cancelled.is_set() and not primary.finished.is_set()
        finally:
            primary.cleanup_release.set()
            await router.aclose()

    run_case(case())


def test_validation_is_per_request_not_shared_router_state(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = AsyncMock(), AsyncMock()
        primary.generate = AsyncMock(return_value="not JSON")
        fallback.generate = AsyncMock(return_value=VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            legacy = asyncio.create_task(router.generate("legacy"))
            strict = asyncio.create_task(
                router.generate_result("strict", response_validator=validate)
            )
            assert await legacy == "not JSON"
            result = await strict
        assert result.provider == "Gemini" and result.content == VALID
        assert primary.generate.await_count == 2
        fallback.generate.assert_awaited_once_with("strict")

    run_case(case())


def test_provider_name_is_not_guessed_for_arbitrary_injected_provider(
    clock: Clock,
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        async with LLMRouter(primary, fallback) as router:
            result = await router.generate_result("prompt", response_validator=validate)
        assert result.provider is None and result.role == "primary"

    run_case(case())


def test_validator_bug_is_sanitized_and_does_not_win(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed("primary")
        fallback.succeed("fallback")

        def broken(text: str) -> None:
            raise RuntimeError("PRIVATE_VALIDATOR_SECRET")

        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await router.generate("prompt", response_validator=broken)
        assert isinstance(caught.value.primary_error, LLMResponseValidationError)
        assert caught.value.primary_error.code == "validator_error"
        assert "PRIVATE_VALIDATOR_SECRET" not in json.dumps(caught.value.to_dict())

    run_case(case())


def test_async_validator_is_rejected_before_calling_providers(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)

        async def unsupported(text: str) -> None:
            pass

        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMConfigurationError):
                await router.generate("prompt", response_validator=unsupported)  # type: ignore[arg-type]
        assert not primary.calls and not fallback.calls

    run_case(case())


def test_factory_declares_real_provider_labels_and_keeps_timeouts(clock: Clock) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
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
            async with groq_gemini_router() as router:
                result = await router.generate_result("prompt", response_validator=validate)
                assert router.hedge_delay_seconds == 60 and router.total_timeout_seconds == 120
            groq.assert_called_once_with(timeout=90.0)
            gemini.assert_called_once_with(timeout=30.0)
            assert result.provider == "Groq"

    run_case(case())
