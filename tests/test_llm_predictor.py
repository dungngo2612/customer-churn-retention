import asyncio
import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from llm_v2_fixtures import Source, make_source, query
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]
from test_llm_router import Clock, Provider, run_case, settle

from churn_retention.adapters.llm.base import (
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseValidationError,
)
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.modules.scoring import llm_router
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from churn_retention.modules.scoring.llm_router import LLMRouter, LLMRouterError

REASON = "Hợp đồng và thời gian sử dụng là bằng chứng; ước lượng còn không chắc chắn."
VALID_DATA = {"churn_prediction": 1, "churn_probability": 0.13, "reason": REASON}
VALID = json.dumps(VALID_DATA, ensure_ascii=False)


@pytest.fixture
def source(tmp_path: Path) -> Source:
    return make_source(tmp_path, n=8)


@pytest.fixture
def store(source: Source) -> ReferenceStore:
    source.build()
    return ReferenceStore.load(source.output, trusted=True)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    value = Clock()
    monkeypatch.setattr(llm_router, "get_running_loop", lambda: value)
    yield value
    assert not [t for t in value.timers if not t.cancelled and not t.fired]


def prompt_data(prompt: str) -> dict[str, Any]:
    return json.loads(
        prompt.split("\nBEGIN_CUSTOMER_DATA_JSON\n", 1)[1].split(
            "\nEND_CUSTOMER_DATA_JSON\n", 1
        )[0]
    )


def test_valid_prediction_runs_full_pipeline_without_reinterpreting_result(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        expected_references = SimilarCustomerRetriever(store).retrieve(query(source))
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await LLMChurnPredictor(store, router).predict(query(source))
        assert result.churn_prediction == 1 and result.churn_probability == 0.13
        assert result.reason == REASON and result.probability_calibrated is False
        assert (
            result.reference_count == 5
            and result.reference_version == store.reference_version
        )
        assert result.provider == "Groq" and not fallback.calls
        data = prompt_data(primary.calls[0][0])
        assert len(data["customer_to_predict"]) == 19
        assert "Churn" not in data["customer_to_predict"]
        assert data["reference_customers"] == [
            r.to_prompt_data() for r in expected_references
        ]
        assert "customerID" not in primary.calls[0][0]
        assert (
            not {"customer_id", "customerID", "references"} & result.model_dump().keys()
        )

    run_case(case())


@pytest.mark.parametrize("kind", ["missing", "category", "total", "extra"])
def test_invalid_inputs_never_call_providers(
    source: Source, store: ReferenceStore, clock: Clock, kind: str
) -> None:
    async def case() -> None:
        row = query(source)
        if kind == "missing":
            row.pop("gender")
        elif kind == "category":
            row["gender"] = "invalid-gender"
        elif kind == "total":
            row.update(tenure=1, TotalCharges=None)
        else:
            row["income"] = 1000
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            predictor = LLMChurnPredictor(store, router)
            with pytest.raises((ValueError, TypeError)):
                await predictor.predict(row)
        assert not primary.calls and not fallback.calls

    run_case(case())


@pytest.mark.parametrize("missing", [None, "", " ", float("nan")])
def test_total_charges_cleaning_reuses_existing_zero_tenure_rule(
    source: Source, store: ReferenceStore, clock: Clock, missing: object
) -> None:
    async def case() -> None:
        row = {**query(source), "tenure": 0, "TotalCharges": missing}
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        async with LLMRouter(primary, fallback) as router:
            await LLMChurnPredictor(store, router).predict(row)
        assert (
            prompt_data(primary.calls[0][0])["customer_to_predict"]["TotalCharges"]
            == 0.0
        )
        assert not fallback.calls

    run_case(case())


@pytest.mark.parametrize(
    "invalid",
    [
        "not JSON",
        "```json\n{}\n```",
        json.dumps({**VALID_DATA, "churn_probability": 2}),
        json.dumps({**VALID_DATA, "churn_probability": float("nan")}),
        json.dumps({**VALID_DATA, "churn_probability": float("inf")}),
        json.dumps({**VALID_DATA, "churn_prediction": True}),
        json.dumps({**VALID_DATA, "churn_prediction": 2}),
        json.dumps({**VALID_DATA, "reason": "  "}),
    ],
)
def test_invalid_qwen_response_triggers_exactly_one_gemini_call(
    source: Source, store: ReferenceStore, clock: Clock, invalid: str
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(invalid)
        fallback.succeed(VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await LLMChurnPredictor(store, router).predict(query(source))
        assert result.provider == "Gemini" and result.churn_probability == 0.13
        assert len(primary.calls) == len(fallback.calls) == 1
        assert primary.calls[0][1] == fallback.calls[0][1] == 0
        assert primary.calls[0][0] == fallback.calls[0][0]

    run_case(case())


def test_qwen_api_error_falls_back_without_waiting_for_hedge(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.fail(LLMRateLimitError("PRIVATE_API_KEY"))
        fallback.succeed(VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await LLMChurnPredictor(store, router).predict(query(source))
        assert result.provider == "Gemini" and fallback.calls[0][1] == 0

    run_case(case())


@pytest.mark.parametrize("kind", ["json", "api"])
def test_both_failures_have_separate_structured_errors(
    source: Source,
    store: ReferenceStore,
    clock: Clock,
    kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        if kind == "json":
            primary.succeed("PRIVATE_INVALID_JSON")
            fallback.succeed("PRIVATE_INVALID_JSON")
        else:
            primary.fail(LLMAPIError("PRIVATE_API_KEY", status_code=500))
            fallback.fail(LLMAPIError("PRIVATE_API_KEY", status_code=503))
        async with LLMRouter(primary, fallback) as router:
            with pytest.raises(LLMRouterError) as caught:
                await LLMChurnPredictor(store, router).predict(query(source))
        structured = caught.value.to_dict()
        assert structured["error"] == "llm_router_failed"
        if kind == "json":
            assert isinstance(caught.value.primary_error, LLMResponseValidationError)
            assert isinstance(caught.value.fallback_error, LLMResponseValidationError)
        else:
            assert isinstance(caught.value.primary_error, LLMAPIError)
            assert isinstance(caught.value.fallback_error, LLMAPIError)
            assert caught.value.primary_error.status_code == 500
            assert caught.value.fallback_error.status_code == 503
        assert (
            "PRIVATE_" not in json.dumps(structured) + str(caught.value) + caplog.text
        )
        assert len(primary.calls) == len(fallback.calls) == 1

    run_case(case())


def test_hedged_valid_winner_does_not_wait_for_qwen_cleanup(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock, delayed_cleanup=True), Provider(clock)
        router = LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        )
        try:
            predictor = LLMChurnPredictor(store, router)
            request = asyncio.create_task(predictor.predict(query(source)))
            await primary.started.wait()
            clock.advance(59)
            await settle()
            assert not fallback.calls
            clock.advance(1)
            await fallback.started.wait()
            assert not primary.cancelled.is_set()
            fallback.succeed(VALID)
            await settle()
            assert request.done() and (await request).provider == "Gemini"
            assert primary.cancelled.is_set() and not primary.finished.is_set()
        finally:
            primary.cleanup_release.set()
            await router.aclose()

    run_case(case())


def test_caller_cancellation_after_hedge_cleans_up_both_providers(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            predictor = LLMChurnPredictor(store, router)
            request = asyncio.create_task(predictor.predict(query(source)))
            await primary.started.wait()
            clock.advance(60)
            await fallback.started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        assert primary.finished.is_set() and fallback.finished.is_set()
        assert primary.cancelled.is_set() and fallback.cancelled.is_set()

    run_case(case())


@pytest.mark.parametrize("maximum", [0, 2, 5])
def test_reference_count_is_actual_prompt_count_not_requested_k(
    source: Source, store: ReferenceStore, clock: Clock, maximum: int
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        async with LLMRouter(primary, fallback) as router:
            predictor = LLMChurnPredictor(store, router, k=7, max_examples=maximum)
            result = await predictor.predict(query(source))
        assert result.reference_count == maximum
        assert prompt_data(primary.calls[0][0])["reference_count"] == maximum

    run_case(case())


def test_size_truncation_reports_actual_number_of_examples(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        row = query(source)
        prepared = store.prepare_query(row)
        references = SimilarCustomerRetriever(store).retrieve(row)
        builder = ChurnPromptBuilder(store.feature_schema)
        budget = len(builder.build(prepared.features, references[:1]))
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        async with LLMRouter(primary, fallback) as router:
            result = await LLMChurnPredictor(
                store, router, max_prompt_chars=budget
            ).predict(row)
        assert result.reference_count == 1
        assert len(primary.calls[0][0]) <= budget
        built = builder.build_with_metadata(prepared.features, references)
        assert (
            built.text == builder.build(prepared.features, references)
            and built.reference_count == 5
        )

    run_case(case())


def test_no_dataset_reads_fits_or_preprocessing_mutations_per_request(
    source: Source, store: ReferenceStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(VALID)
        row = query(source)
        state = pickle.dumps(store.preprocessor._transformer)
        async with LLMRouter(primary, fallback) as router:
            predictor = LLMChurnPredictor(store, router)

            def forbidden(*args, **kwargs):
                pytest.fail("Request must not read data or refit ML components")

            monkeypatch.setattr(pd, "read_csv", forbidden)
            for cls in (StandardScaler, OneHotEncoder, LogisticRegression):
                monkeypatch.setattr(cls, "fit", forbidden)
            with patch.object(
                Path, "read_bytes", side_effect=AssertionError("No artifact reread")
            ):
                first = await predictor.predict(row)
                second = await predictor.predict({**row, "Churn": "ignored"})
        assert first == second and first.provider is None
        assert pickle.dumps(store.preprocessor._transformer) == state
        assert len(primary.calls) == 2 and not fallback.calls

    run_case(case())


def test_reference_id_in_reason_is_rejected_without_editing_llm_result(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        row = query(source)
        reference = SimilarCustomerRetriever(store).retrieve(row)[0]
        primary, fallback = Provider(clock), Provider(clock)
        primary.succeed(
            json.dumps(
                {**VALID_DATA, "reason": "Tham chiếu " + reference.customer_id.upper()}
            )
        )
        fallback.succeed(VALID)
        async with LLMRouter(
            primary, fallback, primary_provider="Groq", fallback_provider="Gemini"
        ) as router:
            result = await LLMChurnPredictor(store, router).predict(row)
        serialized = result.model_dump_json()
        assert result.provider == "Gemini" and result.reason == REASON
        assert all(
            record.customer_id.casefold() not in serialized.casefold()
            for record in store.records
        )

    run_case(case())


def test_prompt_budget_failure_does_not_call_llm(
    source: Source, store: ReferenceStore, clock: Clock
) -> None:
    async def case() -> None:
        primary, fallback = Provider(clock), Provider(clock)
        async with LLMRouter(primary, fallback) as router:
            predictor = LLMChurnPredictor(store, router, max_prompt_chars=1)
            with pytest.raises(ValueError, match="max_prompt_chars"):
                await predictor.predict(query(source))
        assert not primary.calls and not fallback.calls

    run_case(case())
