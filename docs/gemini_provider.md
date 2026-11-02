# Gemini Provider

Provider sử dụng [Google GenAI SDK](https://github.com/googleapis/python-genai),
dùng chung `LLMProvider` và các exception trong `adapters/llm/base.py` với Groq.
Không tích hợp router, scoring hoặc Streamlit.

## Cấu hình và cách gọi

- Cài dependency từ `pyproject.toml`: `python3 -m pip install -e .`.
- Bắt buộc đặt `GEMINI_API_KEY` trong môi trường chạy; không tự đọc `.env`.
- `GEMINI_MODEL` mặc định `gemini-3.5-flash-lite` nếu chưa đặt hoặc để trống.
- Cấu hình được đọc khi khởi tạo; `GOOGLE_API_KEY` không thay thế `GEMINI_API_KEY`.
- Timeout mặc định 30 giây; đổi bằng `GeminiProvider(timeout=60.0)`.
- SDK nhận timeout theo milliseconds; interface provider nhận seconds như Groq.
- Mỗi lần generate chỉ cho phép một attempt và tắt automatic continuation.

Ví dụ sau thực hiện request khi người dùng chủ động chạy với API key đã cấu hình.
Đây không phải request API thật đã được thực hiện trong quá trình triển khai.

```python
import asyncio

from churn_retention.adapters.llm.base import LLMProvider
from churn_retention.adapters.llm.gemini_provider import GeminiProvider


async def main() -> None:
    async with GeminiProvider() as gemini:
        provider: LLMProvider = gemini
        print(await provider.generate("Giải thích customer churn bằng tiếng Việt."))


asyncio.run(main())
```

`async generate(prompt: str) -> str` giữ nguyên nội dung text của response đầu tiên.
Content `None`, rỗng, chỉ có khoảng trắng, không có candidate, bị chặn hoặc chỉ có
thought parts mà không có text trả lời đều gây `LLMEmptyResponseError`.
Prompt sai kiểu gây `TypeError`; prompt rỗng gây `ValueError`, trước khi gọi API.

## Lỗi và vòng đời client

| Tình huống | Exception dùng chung |
| --- | --- |
| Thiếu/rỗng API key, timeout cấu hình không hợp lệ | `LLMConfigurationError` |
| Response không có nội dung text | `LLMEmptyResponseError` |
| HTTP 429 / quota | `LLMRateLimitError` |
| Timeout transport hoặc HTTP 408/504 | `LLMTimeoutError` |
| Lỗi kết nối/transport | `LLMConnectionError` |
| API lỗi khác | `LLMAPIError`, có `status_code` nếu có |

Provider không ghi log, không hiển thị raw SDK errors, response body hoặc headers
trong exception. API key chỉ được đọc từ môi trường và truyền vào SDK.

Provider dùng HTTPX để loại lỗi transport không phụ thuộc việc máy có cài thêm
aiohttp. `httpx` được khai báo là dependency trực tiếp vì provider sử dụng nó;
đây cũng là dependency bắt buộc của Google GenAI SDK.
Sử dụng `async with` hoặc `await provider.aclose()` để đóng client async, HTTP client
do provider tạo và phần sync của SDK. Không đổi provider/client sang event loop khác.

## Kiểm thử

```bash
PYTHONPATH=src python3 -m pytest tests/test_gemini_provider.py -q
PYTHONPATH=src python3 -m pytest tests --ignore=tests/test_gemini_provider.py -q
```

Unit test mock cả SDK và HTTP client, dùng response/exception thật của SDK cùng
`asyncio.run`; không cần `pytest-asyncio`. Không gọi API thật và không xác nhận model
ID hiện được Google cung cấp hoặc API key có quyền truy cập model đó.
