# LLM Churn Predictor — bước 5.4

Use case async kết nối ReferenceStore → SimilarCustomerRetriever →
ChurnPromptBuilder → LLMRouter → parser JSON nghiêm ngặt. Không tích hợp Streamlit,
không dùng classifier Logistic Regression để dự đoán hoặc fallback, không thay đổi
contract Prediction của luồng ML hiện tại. Không thêm dependency.

## Interface và vòng đời

```python
from churn_retention.adapters.retrieval.reference_store import ReferenceStore
from churn_retention.modules.scoring.llm_predictor import LLMChurnPredictor
from churn_retention.modules.scoring.llm_router import groq_gemini_router

# Startup: đọc/kiểm tra artifact và nạp fitted preprocessing đúng một lần.
store = ReferenceStore.load("artifacts/references/telco-train-v1.zip")


async def serve(customers):
    async with groq_gemini_router() as router:
        predictor = LLMChurnPredictor(store, router, k=5, max_examples=5, max_prompt_chars=16_000)
        for customer in customers:
            result = await predictor.predict(customer)
            yield result.model_dump()
```

Đây là ví dụ cách kết nối provider thật, không được thực thi trong bước 5.4.
`predict(customer: Mapping[str, object]) -> LLMChurnPrediction` là async.
Mỗi request gọi `router.generate_result(...)` đúng một lần, không có vòng retry
hoặc fallback thứ hai tại Predictor.

Giữ Store, Predictor và Router để phục vụ nhiều request. Retriever dựng chỉ mục
một lần khi tạo Predictor. Mỗi request chỉ validate/clean và transform query,
tìm K láng giềng trên ma trận đã giữ trong RAM; không đọc CSV/archive, không fit
encoder/scaler, không chạy predict/predict_proba của Logistic Regression.
Interface Retriever hiện tự chuẩn bị query; Predictor cũng chuẩn bị query cho
Prompt Builder, nên có hai lần transform một hàng, không transform lại train.

Input dùng đúng 19 feature chính thức. Category, kiểu dữ liệu và TotalCharges
được kiểm tra bằng ReferenceStore.prepare_query hiện có. TotalCharges thiếu chỉ
được thay bằng 0 khi tenure=0. Metadata query như customerID/Churn không được
đưa vào prompt; ID nếu có vẫn được dùng nội bộ để loại self-match.
Các lỗi input/giới hạn prompt giữ nguyên TypeError/ValueError hiện có và xảy ra
trước khi gọi provider.

Predictor mượn Store/Router, không mở hoặc đóng client. Caller phải đóng Router
trước provider. `groq_gemini_router()` đã quản lý đúng thứ tự này.

## Parser, validator và fallback

`parse_churn_response(text: str) -> LLMChurnResponse` nằm tại
`modules/scoring/llm_response.py`, sử dụng schema riêng `contracts/llm.py`:

- Chỉ nhận một JSON object với đúng ba key, từ chối cả key trùng lặp.
- Không sửa JSON, bóc Markdown/code fence hay trích JSON khỏi văn bản khác.
- churn_prediction: integer 0/1; từ chối bool, float và numeric string.
- churn_probability: int/float hữu hạn trong [0,1], từ chối bool/numeric string,
  NaN/Infinity và số overflow thành infinity.
- reason: Unicode string có nội dung ngoài khoảng trắng, tối đa 300 ký tự.
- Giới hạn toàn response 16.000 ký tự, trước khi parse JSON.
- Giữ nguyên giá trị hợp lệ, kể cả khoảng trắng trong reason. Không tính xác suất
  từ nhãn tham chiếu, không áp threshold để sửa nhãn hoặc xác suất của LLM.

Router bổ sung validator **đồng bộ, theo từng request**, nhận text và raise để từ
chối. Predictor truyền parser qua callback này; validation chạy trước khi publish
response vào hàng đợi chọn winner. Không có cache validator hoặc last-provider
dùng chung giữa request.

```python
text = await router.generate(prompt)  # Hành vi cũ: chuỗi không rỗng là thành công.
result = await router.generate_result(prompt, response_validator=validate)
# validate(text) -> None, raise LLMResponseValidationError để từ chối.
# result.content / result.role / result.provider
```

Qwen JSON hợp lệ trả ngay; API error hoặc JSON sai khởi chạy Gemini ngay. Sau
hedge, JSON sai chỉ loại provider đó, tiếp tục chờ provider còn lại. Không thể để
JSON sai thắng JSON hợp lệ. Gemini được gọi tối đa một lần trong một request.
Lỗi không mong đợi trong validator được chuẩn hóa thành validation error an toàn.

Khi cả hai thất bại, `LLMRouterError.primary_error` và `.fallback_error` phân biệt
`LLMResponseValidationError` với các lỗi API/rate limit/timeout/kết nối hiện có.
`error.to_dict()` trả thông tin cấu trúc gồm loại lỗi, validation code hoặc HTTP
status hợp lệ, cờ deadline. Không chứa raw response, SDK error text hoặc API key.
Validation code thuộc allowlist: invalid_json, invalid_object, invalid_keys,
invalid_prediction, invalid_probability, invalid_reason, response_too_large,
validator_error. Không log prompt, response hoặc dữ liệu khách hàng.

## Hedging và metadata

Không đổi timeout: hedge **60s**, Groq SDK **90s**, Gemini SDK **30s**, deadline
chung **120s** trong factory. Inject thủ công phải giữ timeout primary lớn hơn
hedge như trước. Qwen tiếp tục chạy tại mốc hedge; response hợp lệ đầu tiên thắng.

Winner làm Router cancel task thua và trả ngay, không await cleanup của task thua.
Task còn chạy được giữ strong reference và thu thập exception. Caller cancellation
vẫn được truyền; `aclose()` chờ drain trước khi client bị đóng. Validator không tạo
task riêng. Provider phải hỗ trợ cooperative cancellation như hợp đồng Router cũ.

`RoutedResponse.role` lấy trực tiếp từ outcome thắng. Tên provider do wiring khai
báo: factory đặt Groq/Gemini chính xác; dependency injection không khai báo tên
thì trả `None`. Không đoán tên từ nội dung LLM hoặc class name.

Prompt Builder thêm `build_with_metadata(...) -> BuiltChurnPrompt`, gồm text và
reference_count sau khi cap số ví dụ và bỏ ví dụ cuối để vừa budget. `build()` cũ
vẫn trả đúng string như trước. Predictor trả số ví dụ thực sự này cùng
ReferenceStore.reference_version và `probability_calibrated=False`.

Output không chứa reference list hoặc ID. Ngoài việc prompt chỉ serialize feature
và nhãn reference, Predictor từ chối reason có chứa ID query hoặc reference đã
truy xuất; không sửa reason của LLM. Kiểm tra substring không phân biệt hoa/thường
này có thể từ chối quá mức nếu ID rất ngắn; ID Telco hiện tại dài và có cấu trúc.

## Kiểm thử và giới hạn

```bash
python3 -m pytest tests/test_llm_response.py tests/test_llm_router_validation.py tests/test_llm_predictor.py -q
python3 -m pytest tests --ignore=tests/test_llm_response.py --ignore=tests/test_llm_router_validation.py --ignore=tests/test_llm_predictor.py -q
python3 -m ruff check src tests
python3 -m ruff format --check src tests
python3 -m mypy src/churn_retention/adapters/llm/base.py src/churn_retention/contracts/llm.py src/churn_retention/modules/scoring/llm_response.py src/churn_retention/modules/scoring/llm_predictor.py src/churn_retention/modules/scoring/llm_router.py src/churn_retention/modules/scoring/llm_prompt.py tests/test_llm_response.py tests/test_llm_router_validation.py tests/test_llm_predictor.py
```

Test dùng fake provider, SDK mock và clock mô phỏng 60/120 giây. Fixture tổng hợp
chỉ kiểm tra cơ chế. Ví dụ chứa dữ liệu khách hàng thật được giữ ngoài Git.

Parser xác minh cấu trúc/kiểu/giới hạn, chưa thể chứng minh reason đúng sự thật,
viết bằng tiếng Việt hay probability có chất lượng/calibration. Prompt chỉ dẫn các
yêu cầu ngữ nghĩa đó. Cần đánh giá trên tập tách riêng trước khi dùng nghiệp vụ.
Retrieval hiện chạy đồng bộ trên tập nhỏ trong RAM. Chưa có API/UI, calibration,
đánh giá độ chính xác hoặc kết nối Streamlit trong bước này.

Kết quả kiểm tra bước 5.4:

- Unit test mới: 99 pass, 0 fail.
- Regression toàn bộ 511 test có sẵn: 511 pass, 0 fail. Tổng 610 test khác nhau.
- Ruff và format check `src tests`: pass (97 file Python đã đúng format).
- Ruff toàn repository: còn I001 ở `notebooks/explore_and_split.py:1`, không sửa.
- Mypy 9 file thay đổi/mới: pass. Mypy toàn bộ `src tests`: 11 lỗi ở 4 file test
  cũ không chỉnh (`test_training.py`, `test_bundle.py`, `test_train_workflow.py`,
  `test_tune.py`): thiếu sklearn stubs và hai lỗi pandas iloc index.
- So sánh SHA-256 141 file có sẵn: chỉ base.py, llm_router.py và llm_prompt.py thay
  đổi trong phạm vi được phép; 138 file còn lại giữ nguyên, gồm dữ liệu, model
  bundle, reference artifact, source ML/batch/UI và pyproject.toml.
- Ví dụ dùng artifact thật + fake provider chạy thành công; API thật: 0 request.
