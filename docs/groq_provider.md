# Groq Provider

Provider độc lập, chưa tích hợp scoring, Streamlit hay router. Sử dụng `AsyncGroq`
từ [Groq Python SDK](https://github.com/groq/groq-python#async-usage).

## Cấu hình

- Cài dependency từ `pyproject.toml`: `python3 -m pip install -e .`.
- Bắt buộc đặt `GROQ_API_KEY` trong môi trường chạy. Provider không tự đọc `.env`.
- `GROQ_MODEL` mặc định là `qwen/qwen3.8-27b` khi chưa đặt hoặc để trống.
- Cấu hình được đọc tại thời điểm khởi tạo provider.
- Timeout mặc định 30 giây; có thể truyền `GroqProvider(timeout=60.0)`.
- Tắt retry tự động của SDK (`max_retries=0`); caller quyết định việc thử lại.

Không đưa API key vào source code. Provider không ghi log và không đưa thông báo,
response body hay request header của lỗi SDK vào exception hiển thị.

## Interface và cách gọi

Contract `LLMProvider` tại `adapters/llm/base.py` chỉ yêu cầu
`async generate(prompt: str) -> str`. Các provider khác có thể đáp ứng cùng contract
mà không cần phụ thuộc Groq SDK.

Ví dụ dưới đây thực hiện request khi người dùng chủ động chạy trong môi trường đã
cấu hình API key. Đây không phải bước kiểm thử đã thực hiện trong giai đoạn này.

```python
import asyncio

from churn_retention.adapters.llm.base import LLMProvider
from churn_retention.adapters.llm.groq_provider import GroqProvider


async def main() -> None:
    async with GroqProvider() as groq:
        provider: LLMProvider = groq
        answer = await provider.generate("Giải thích customer churn bằng tiếng Việt.")
        print(answer)


asyncio.run(main())
```

Kết quả giữ nguyên nội dung chuỗi của model. Không có choices, content `None`,
chuỗi rỗng hoặc chỉ có khoảng trắng đều gây `LLMEmptyResponseError`.
Prompt sai kiểu gây `TypeError`; prompt rỗng gây `ValueError`, trước khi gọi API.
Sử dụng `async with` hoặc `await provider.aclose()` để đóng kết nối.

## Lỗi

Các lỗi provider kế thừa `LLMProviderError`, không yêu cầu caller import Groq SDK:

| Tình huống | Exception |
| --- | --- |
| Thiếu/rỗng `GROQ_API_KEY`, timeout cấu hình không hợp lệ | `LLMConfigurationError` |
| Phản hồi không có nội dung | `LLMEmptyResponseError` |
| HTTP 429 | `LLMRateLimitError` |
| Timeout | `LLMTimeoutError` |
| Lỗi kết nối | `LLMConnectionError` |
| HTTP lỗi khác hoặc lỗi API khác | `LLMAPIError` (`status_code` nếu có) |

## Kiểm thử

```bash
PYTHONPATH=src python3 -m pytest tests/test_groq_provider.py -q
PYTHONPATH=src python3 -m pytest tests --ignore=tests/test_groq_provider.py -q
```

Test provider mock toàn bộ `AsyncGroq`, dùng response/exception thật của SDK và
`asyncio.run` của standard library; không cần thêm `pytest-asyncio`.
Không có request API thật trong các test này. Mock test không xác nhận model ID
đang được Groq cung cấp hoặc API key có quyền truy cập model đó.
