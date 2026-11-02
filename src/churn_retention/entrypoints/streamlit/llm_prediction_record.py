"""Session-only accepted record capture; no inference or persistence side effects."""

from collections.abc import Mapping
from typing import Protocol

from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_prediction_record import PredictionRecord

RECORD_KEY = "llm_retention.accepted_record"
RECORD_CONTEXT_KEY = "llm_retention.accepted_reference_context"


class RecordState(Protocol):
    """The two operations shared by dict and Streamlit's session proxy."""

    def __setitem__(self, key: str, value: object) -> None: ...

    def pop(self, key: str, default: None, /) -> object: ...


def clear_record(state: RecordState) -> None:
    state.pop(RECORD_KEY, None)
    state.pop(RECORD_CONTEXT_KEY, None)


def capture_submission(
    state: RecordState,
    snapshot: Mapping[str, object],
    prediction: LLMChurnPrediction,
    reference_sha256: str,
    reference_context: object,
) -> None:
    record = PredictionRecord.capture(
        snapshot, prediction, reference_sha256=reference_sha256
    )
    state[RECORD_KEY] = record
    state[RECORD_CONTEXT_KEY] = reference_context
