# LLM Router

`modules/scoring/llm_router.py` dùng chung `LLMProvider` hiện có. Router không tích
hợp scoring, Logistic Regression, batch scoring hay Streamlit; không thay đổi các
provider hoặc dependency.

## Định tuyến

| Trường hợp | Hành vi |
| --- | --- |
| A: Qwen thành công trước hedge | Trả ngay Qwen; không gọi generate của Gemini. |
| B: Qwen lỗi trước hedge | Khởi chạy Gemini ngay, không chờ timer. |
| C: Qwen còn chạy tại hedge | Khởi chạy Gemini và giữ Qwen chạy; text hợp lệ đến trước thắng. |
| D: Một provider lỗi khi chạy song song | Chờ provider còn lại, trong giới hạn deadline. |
| E: Cả hai lỗi | Raise `LLMRouterError` với lỗi riêng của primary và fallback. |

Response phải là string, có nội dung ngoài khoảng trắng. Chuỗi hợp lệ được giữ
nguyên. Ngoại lệ bất thường, response rỗng và provider tự cancellation cũng được
xem là thất bại của provider; cancellation của caller được truyền về caller.
Mỗi request gọi primary một lần và fallback tối đa một lần. Không retry tại Router.
Queue lưu thứ tự hoàn tất, kể cả khi hai provider hoàn tất cùng một lượt event loop.

## Cấu hình timeout và cách gọi

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `hedge_delay_seconds` | 60 | Mốc khởi chạy Gemini nếu Qwen còn chờ. |
| `total_timeout_seconds` | 120 | Deadline chung, tính từ khi bắt đầu request. |
| `primary_timeout_seconds` | 90 | Timeout SDK Groq trong context manager tạo router. |
| `fallback_timeout_seconds` | 30 | Timeout SDK Gemini trong context manager tạo router. |

Context manager `groq_gemini_router` tạo và sở hữu hai provider; yêu cầu API key
được cấu hình trong `GROQ_API_KEY`, `GEMINI_API_KEY`. Model do mỗi provider đọc từ
`GROQ_MODEL`, `GEMINI_MODEL` như trước. Không có request API khi chỉ tạo client.
Ví dụ dưới đây chỉ thực hiện request khi người dùng chủ động chạy:

```python
import asyncio

from churn_retention.modules.scoring.llm_router import groq_gemini_router


async def main() -> None:
    async with groq_gemini_router(
        hedge_delay_seconds=60,
        primary_timeout_seconds=90,
        fallback_timeout_seconds=30,
        total_timeout_seconds=120,
    ) as router:
        answer = await router.generate("Giải thích customer churn bằng tiếng Việt.")
        print(answer)


asyncio.run(main())
```

Các tham số là cấu hình của caller, không sửa mặc định 30 giây trong provider.
Context manager từ chối `primary_timeout_seconds <= hedge_delay_seconds` trước khi
tạo client, nên cấu hình Qwen 30 giây không vô hiệu hóa hedge 60 giây tại đây.
Deadline chung phải lớn hơn hedge; các giá trị phải hữu hạn, timeout dương và hedge
không âm. Timeout SDK là timeout I/O; Router còn có deadline chung độc lập.

Để dependency injection thủ công, dùng
`LLMRouter(primary, fallback, hedge_delay_seconds=60, total_timeout_seconds=120)`.
`LLMProvider` chỉ khai báo generate, không cho phép đọc/đổi timeout hoặc đóng client.
Caller phải truyền primary đã cấu hình timeout lớn hơn hedge, chẳng hạn
`GroqProvider(timeout=90)`, và giữ provider sống đến khi Router đóng xong. Router
không tự thay đổi hoặc suy đoán cấu hình nội bộ của provider được inject.

## Task và vòng đời client

- Router mượn các provider được inject, sở hữu các task của từng request.
- Khi có kết quả thắng, `generate()` gửi cancellation cho task thua rồi trả ngay;
  không await cleanup của task thua trong finally.
- Deadline và cancellation từ caller cũng gửi cancellation, không chờ provider
  đang dọn dẹp trước khi trả lỗi/cancellation từ generate.
- Các task còn chạy có strong reference và callback thu thập exception. Timer
  hedge/deadline được hủy khi request kết thúc. Các request song song độc lập nhau.
- `await router.aclose()` hoặc thoát `async with router` chặn request mới, hủy các
  request đang chạy và chờ chúng dọn dẹp. Không gửi cancellation lần hai khi task
  đã đang xử lý cancellation.
- Trong lúc đóng, cancellation của caller được trì hoãn đến khi drain xong rồi
  được raise lại. Context manager tạo router đóng Router trước, rồi Gemini, Groq;
  không đóng client khi còn task đang dùng.

Chỉ shutdown (`aclose`) chờ drain bằng gather. Kết quả `generate` có thể đến trước
khi drain hoàn tất; thoát toàn bộ context manager vẫn phải đợi drain để đóng client
an toàn. Provider được inject phải sử dụng I/O không block event loop và hỗ trợ
cooperative cancellation. Python không thể cưỡng chế kết thúc coroutine nuốt
cancellation vô hạn; trong trường hợp đó generate vẫn có deadline nhưng shutdown
phải chờ để tránh đóng client đang dùng. Hai SDK hiện tại sử dụng async I/O.

## Lỗi và bảo vệ dữ liệu

`LLMRouterError` kế thừa `LLMProviderError`, có `primary_error`, `fallback_error`
và `deadline_exceeded`. Hai lỗi thành phần là bản sao an toàn thuộc các exception
chung: giữ loại lỗi và HTTP status hợp lệ (100–599), loại bỏ message, raw exception,
traceback SDK, headers, body và notes. Không đưa prompt hoặc API key vào log/lỗi.
Lỗi chưa biết được đưa về `LLMProviderError` với message cố định.
Deadline chung giữ lỗi của provider đã thất bại và gán `LLMTimeoutError` cho provider
chưa hoàn tất. Caller có thể phân biệt rate limit, API, timeout theo từng vai trò.

## Kiểm thử

```bash
PYTHONPATH=src python3 -m pytest tests/test_llm_router.py -q
PYTHONPATH=src python3 -m pytest tests/test_groq_provider.py tests/test_gemini_provider.py -q
PYTHONPATH=src python3 -m pytest tests --ignore=tests/test_llm_router.py \
    --ignore=tests/test_groq_provider.py --ignore=tests/test_gemini_provider.py -q
python3 -m ruff check src/churn_retention/modules/scoring/llm_router.py tests/test_llm_router.py
python3 -m ruff format --check src/churn_retention/modules/scoring/llm_router.py tests/test_llm_router.py
PYTHONPATH=src python3 -m mypy src/churn_retention/modules/scoring/llm_router.py tests/test_llm_router.py
```

Test dùng clock điều khiển timer Router, fake provider và SDK mock. Mốc 30/60/120
giây được mô phỏng, không đợi thời gian thật. Test kiểm tra cleanup chậm, lỗi lúc
cleanup, caller cancellation lặp lại, không còn task/timer và không có exception
bị bỏ quên. Không thực hiện request API thật để kiểm thử Router.
