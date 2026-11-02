"""Allowlisted prediction failures: no provider response, credentials or prompt."""

from types import MappingProxyType

FAILURE_MESSAGES = MappingProxyType(
    {
        "configuration": "Cấu hình LLM chưa hợp lệ; chưa thể dự đoán.",
        "rate_limit": "Dịch vụ LLM đang giới hạn lượt gọi hoặc hết quota.",
        "timeout": "Dịch vụ LLM hết thời gian chờ.",
        "deadline": "Dự đoán vượt tổng thời gian chờ cho phép.",
        "connection": "Không thể kết nối dịch vụ LLM.",
        "invalid_response": "Dịch vụ LLM không trả về JSON/output hợp lệ.",
        "validation": "Dữ liệu đầu vào không hợp lệ.",
        "provenance": "Provider hoặc provenance của kết quả không khớp ngữ cảnh đã xác thực.",
        "snapshot": "Snapshot dự đoán không hợp lệ hoặc đã thay đổi.",
        "interrupted": "Lượt gọi bị ngắt; kết quả chưa xác định, không tự gọi lại API.",
        "not_run": "Chưa gọi LLM cho dòng này.",
        "prediction_failed": "Dự đoán không hoàn tất; không tự gọi lại API.",
        "unavailable": "Dịch vụ dự đoán chưa khả dụng.",
    }
)


def failure_message(code: str) -> str:
    return FAILURE_MESSAGES.get(code, FAILURE_MESSAGES["prediction_failed"])


class DashboardPredictionError(ValueError):
    """Safe failure metadata across the adapter/worker/UI boundaries."""

    def __init__(self, code: str, status: str = "FAILED") -> None:
        self.code = code if code in FAILURE_MESSAGES else "prediction_failed"
        self.status = status
        super().__init__(failure_message(self.code))
