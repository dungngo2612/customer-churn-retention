"""Async few-shot churn prediction using cached retrieval and validated routing."""

from collections.abc import Mapping

from churn_retention.adapters.llm.base import LLMResponseValidationError
from churn_retention.adapters.retrieval.nearest_neighbors import (
    SimilarCustomerRetriever,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.modules.scoring.llm_prompt import ChurnPromptBuilder
from churn_retention.modules.scoring.llm_response import parse_churn_response
from churn_retention.modules.scoring.llm_router import LLMRouter


class LLMChurnPredictor:
    """Borrow a Store/Router; build the index and prompt builder once, not per request.

    Keep the router inside its provider lifetimes. This use case neither opens
    nor closes clients and does not implement the synchronous ML Predictor port.
    """

    def __init__(
        self,
        store: ReferenceStore,
        router: LLMRouter,
        *,
        k: int = 5,
        max_examples: int = 5,
        max_prompt_chars: int = 16_000,
    ) -> None:
        self._store = store
        self._router = router
        self._builder = ChurnPromptBuilder(
            store.feature_schema,
            max_examples=max_examples,
            max_prompt_chars=max_prompt_chars,
        )
        self._retriever = SimilarCustomerRetriever(store, k=k)

    async def predict(self, customer: Mapping[str, object]) -> LLMChurnPrediction:
        prepared = self._store.prepare_query(customer)
        references = self._retriever.retrieve(
            prepared.features, customer_id=prepared.customer_id
        )
        built = self._builder.build_with_metadata(prepared.features, references)
        private_ids = tuple(
            reference.customer_id.casefold() for reference in references
        )
        if prepared.customer_id is not None:
            private_ids += (prepared.customer_id.casefold(),)

        def validate(text: str) -> None:
            parsed = parse_churn_response(text)
            if any(
                identifier in parsed.reason.casefold() for identifier in private_ids
            ):
                # Reject rather than edit the LLM's explanation or expose IDs.
                raise LLMResponseValidationError(
                    "Explanation contains a private identifier.", code="invalid_reason"
                )

        routed = await self._router.generate_result(
            built.text, response_validator=validate
        )
        # Reparse the same immutable winning text; do not keep shared validator
        # caches or mutable last-winner state across concurrent requests.
        parsed = parse_churn_response(routed.content)
        return LLMChurnPrediction(
            **parsed.model_dump(),
            provider=routed.provider,
            reference_count=built.reference_count,
            reference_version=self._store.reference_version,
            probability_calibrated=False,
        )
