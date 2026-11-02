# Churn Prompt Builder — bước 5.3

`modules/scoring/llm_prompt.py` chỉ tạo prompt string. Không gọi Router/LLM, không
thực hiện retrieval, không kiểm tra JSON phản hồi và không thay đổi reference artifact.

## Interface

```python
builder = ChurnPromptBuilder(
    store.feature_schema,
    max_examples=5,
    max_prompt_chars=16_000,
)
prepared = store.prepare_query(raw_customer)
references = retriever.retrieve(raw_customer)
prompt = builder.build(prepared.features, references)
```

Import class từ `churn_retention.modules.scoring.llm_prompt`.
Constructor nhận `Sequence[FeatureColumn]` của schema chính thức từ ReferenceStore.
`build(query: Mapping[str, object], references: Sequence[SimilarCustomer]) -> str`.
Các kết quả SimilarCustomer được chuyển bằng đúng `to_prompt_data()` hiện có:
`{"features": {...}, "Churn": "Yes/No"}`.

## Dữ liệu và giới hạn

- Chỉ đưa đúng 19 feature vào `customer_to_predict` và feature của từng reference.
- `customerID`, `customer_id`, `Churn` ở cấp query được bỏ qua, không đọc giá trị.
  Trường ngoài schema khác bị từ chối. Feature reference phải khớp schema chính xác.
- Giữ nhãn `Churn` No/Yes của từng reference. ID và distance không được serialize.
- Kiểm tra numerical hữu hạn/không âm, integer tenure, SeniorCitizen 0/1; không nhận
  boolean hoặc numeric string. Categorical phải là string không rỗng, tối đa 128 ký tự.
- Category membership và quy tắc cleaning do `ReferenceStore.prepare_query()` đã
  triển khai xử lý. Builder không có category vocabulary trong feature schema và
  không tự đọc dataset, clean lại hoặc transform. Truyền `prepared.features` khi dùng
  thực tế để category/TotalCharges đã được kiểm tra theo fitted preprocessing.
- Mặc định lấy tối đa 5 ví dụ; `max_examples` cấu hình được từ 0 đến 20. Áp dụng
  giới hạn trước khi đọc/serialize ví dụ, giữ thứ tự nearest-first của Retriever.
- Mặc định tối đa 16.000 ký tự cho **toàn bộ prompt**; giới hạn cấu hình là 1–32.000.
  Bỏ dần ví dụ cuối nếu vượt budget; không cắt query, instructions hoặc JSON.
  Nếu query và instructions vẫn không vừa thì raise ValueError.
- References rỗng hoặc không còn ví dụ vừa budget: prompt nêu rõ không có tham chiếu,
  yêu cầu thể hiện bằng chứng hạn chế/sự không chắc chắn, không tự bịa ví dụ.

Giới hạn là số ký tự Python, không phải token count. String length cũng bị giới hạn
trước serialization. Không cần thêm dependency tokenizer.

## Ranh giới dữ liệu

Feature được serialize bằng JSON với `allow_nan=False`, key order deterministic,
escape Unicode/control characters và thêm escape cho `<`, `>`, `&`, backtick.
JSON nằm trong delimiter theo dòng riêng. Chuỗi có newline, quote hoặc delimiter
giả vẫn là giá trị string trong cùng JSON, không tạo phần chỉ dẫn mới.
Prompt nói rõ không thực thi/làm theo chỉ dẫn nhúng trong dữ liệu.

Đây là bảo vệ cấu trúc prompt; không bảo đảm tuyệt đối hành vi của một LLM.
Builder không thể phát hiện danh tính cố tình nhét vào một feature string sai ngữ
nghĩa. Category validation của Store phải chạy trước khi sử dụng trong luồng thực tế.

## Hợp đồng kết quả được yêu cầu trong prompt

Một object JSON duy nhất, đúng ba khóa:

- `churn_prediction`: integer 0/1, không phải boolean hoặc chuỗi số.
- `churn_probability`: số hữu hạn trong [0, 1].
- `reason`: string tiếng Việt, 1–300 ký tự.

Không có Markdown/nội dung ngoài JSON. Prompt giải thích Yes/No, yêu cầu chỉ dùng
feature và reference được cung cấp, không biến tỷ lệ nhãn K thành xác suất, không
khẳng định calibration, không bịa đặc điểm/số liệu lịch sử và nêu sự không chắc chắn.
Không tự đặt cửa sổ dự đoán theo tháng khi không có dữ liệu thời gian.

JSON Schema chỉ được đưa vào prompt như hợp đồng mô tả kết quả. Bước 5.4 mới xây
validation phản hồi thực tế; module này không parse/validate phản hồi LLM và không
thực hiện retry hoặc fallback.

## Kiểm thử

```bash
PYTHONPATH=src python3 -m pytest tests/test_llm_prompt.py -q
PYTHONPATH=src python3 -m pytest tests --ignore=tests/test_llm_prompt.py -q
python3 -m ruff check src/churn_retention/modules/scoring/llm_prompt.py tests/test_llm_prompt.py
python3 -m ruff format --check src/churn_retention/modules/scoring/llm_prompt.py tests/test_llm_prompt.py
PYTHONPATH=src python3 -m mypy src/churn_retention/modules/scoring/llm_prompt.py tests/test_llm_prompt.py
```

Unit tests kiểm tra pure Prompt Builder, không nạp pickle, fit retrieval hay gọi API.
Ví dụ prompt từ dữ liệu dự án được tạo riêng sau kiểm thử, với ID/nhãn query bị loại.
