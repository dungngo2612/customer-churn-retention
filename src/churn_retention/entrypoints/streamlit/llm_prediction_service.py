"""One synchronous UI submission owns one async prediction/client lifecycle."""

import asyncio
import os
from collections.abc import Mapping

from churn_retention.adapters.llm.base import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.contracts.application_security import reject_credentials
from churn_retention.contracts.llm import LLMChurnPrediction
from churn_retention.contracts.llm_failure import FAILURE_MESSAGES
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor
from churn_retention.modules.scoring.llm_router import (
    LLMRouterError,
    groq_gemini_router,
)


class LLMPredictionServiceError(RuntimeError):
    """A static, user-safe UI message; never carries an SDK response/prompt."""

    def __init__(self, message: str, *, code: str = "unavailable") -> None:
        self.code = code if code in FAILURE_MESSAGES else "unavailable"
        super().__init__(message)


def _error_code(error: Exception) -> str:
    if isinstance(error, LLMConfigurationError):
        return "configuration"
    if isinstance(error, LLMRouterError) and error.deadline_exceeded:
        return "deadline"
    failures = (
        (error.primary_error, error.fallback_error)
        if isinstance(error, LLMRouterError)
        else (error,)
    )
    for failure_type, code in (
        (LLMRateLimitError, "rate_limit"),
        (LLMTimeoutError, "timeout"),
        (LLMConnectionError, "connection"),
        (LLMResponseValidationError, "invalid_response"),
    ):
        if any(isinstance(failure, failure_type) for failure in failures):
            return code
    return "validation" if isinstance(error, (ValueError, TypeError)) else "unavailable"


def _error_message(error: Exception) -> str:
    if isinstance(error, LLMConfigurationError):
        return "Cấu hình dịch vụ LLM chưa hợp lệ. Vui lòng liên hệ người quản trị."
    failures = (
        (error.primary_error, error.fallback_error)
        if isinstance(error, LLMRouterError)
        else (error,)
    )
    if isinstance(error, LLMRouterError) and error.deadline_exceeded:
        return "Dự đoán đã vượt thời gian chờ. Vui lòng thử lại sau."
    if any(isinstance(failure, LLMRateLimitError) for failure in failures):
        return "Chưa thể dự đoán: dịch vụ LLM đang giới hạn lượt gọi hoặc hết quota. Vui lòng thử lại sau."
    if any(isinstance(failure, LLMTimeoutError) for failure in failures):
        return "Chưa thể dự đoán: dịch vụ LLM hết thời gian chờ. Vui lòng thử lại sau."
    if any(isinstance(failure, LLMConnectionError) for failure in failures):
        return (
            "Không thể kết nối dịch vụ LLM. Vui lòng kiểm tra kết nối hoặc thử lại sau."
        )
    if any(isinstance(failure, LLMResponseValidationError) for failure in failures):
        return "Dịch vụ LLM chưa trả về kết quả JSON hợp lệ. Vui lòng thử lại sau."
    if isinstance(error, (ValueError, TypeError)):
        return "Không thể dự đoán từ dữ liệu đã nhập. Vui lòng kiểm tra lại dữ liệu."
    return "Chưa thể hoàn tất dự đoán bằng LLM. Vui lòng thử lại sau hoặc liên hệ người quản trị."


async def _predict_once(
    store: ReferenceStore, snapshot: Mapping[str, object]
) -> LLMChurnPrediction:
    # Factory defaults are the backend contract: Groq 90s, Gemini 30s,
    # hedge 60s, overall deadline 120s. Router drains tasks BEFORE clients close.
    async with groq_gemini_router() as router:
        predictor = LLMChurnPredictor(store, router, k=5)
        return await predictor.predict(snapshot)


def predict_customer(
    store: ReferenceStore, snapshot: Mapping[str, object]
) -> LLMChurnPrediction:
    """Reuse the loaded Store; open/close async resources entirely inside run().

    Streamlit's script thread is synchronous. Refuse use from a running event
    loop instead of nesting loops or moving loop-bound clients between threads.
    The existing two-provider factory requires BOTH keys, even for a Groq win.
    No retries, clients, tasks, or credential caches are added at this boundary.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise LLMPredictionServiceError(
            "Không thể dự đoán trong luồng đang chạy tác vụ async. Vui lòng liên hệ người quản trị.",
            code="configuration",
        )
    if not all(
        os.environ.get(name, "").strip() for name in ("GROQ_API_KEY", "GEMINI_API_KEY")
    ):
        raise LLMPredictionServiceError(
            "Chưa đủ cấu hình LLM. Người quản trị cần cấu hình cả GROQ_API_KEY và GEMINI_API_KEY trên máy chủ.",
            code="configuration",
        )
    if (
        not isinstance(snapshot, Mapping)
        or len(snapshot) != 19
        or set(snapshot) != {column.name for column in store.feature_schema}
    ):
        raise LLMPredictionServiceError(
            "Dữ liệu dự đoán phải chứa đúng 19 đặc trưng, không có ID hoặc nhãn Churn.",
            code="validation",
        )
    try:
        reject_credentials(dict(snapshot))
        result = asyncio.run(_predict_once(store, dict(snapshot)))
        reject_credentials(result.model_dump())
        return result
    except Exception as error:  # noqa: BLE001 -- Sanitize all errors at the UI boundary.
        # Do not expose raw SDK/transformer/cleanup errors or exception chains.
        raise LLMPredictionServiceError(
            _error_message(error), code=_error_code(error)
        ) from None
