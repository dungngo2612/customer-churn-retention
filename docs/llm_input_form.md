# Form LLM/RAG trên main

Chạy app Predict/Retention hiện tại từ repository root:

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit_llm/app.py
```

Trang `apps/streamlit/pages/2_Predict.py` được entrypoint này đăng ký tường minh.
Entrypoint classifier cũ và trang Explain đã bỏ khỏi main; bản trước được giữ trên
nhánh archive ghi trong [README](../README.md).

Submit hợp lệ gọi LLMChurnPredictor qua router Qwen/Groq và Gemini hiện có.
Server cần `GROQ_API_KEY` và `GEMINI_API_KEY` để dự đoán thật. Form không có lựa chọn
classifier hoặc fallback về classifier. Các test chỉ dùng fake providers, không
cần gọi API trả phí.

## Input và validation

19 feature chia thành ba nhóm: thông tin khách hàng (5), dịch vụ (9), hợp đồng
và thanh toán (5). Categorical dùng selectbox với nhãn tiếng Việt, giá trị backend
giữ nguyên. SeniorCitizen dùng selectbox số nguyên 0/1; tenure dùng number_input
số nguyên, các khoản phí dùng number_input số thực. Các widget ban đầu chưa có
giá trị để người dùng lựa chọn/nhập rõ ràng.

Helper `entrypoints/streamlit/llm_input.py` nhận ReferenceStore qua interface hiện
có. Tên feature lấy từ `feature_schema`; các lựa chọn categorical lấy một lần từ
feature của các bản ghi train đã được Store kiểm tra. Chỉ chiếu category values,
không đưa ID hoặc Churn vào dữ liệu của form. Không fit encoder mới, không đọc
CSV để dựng vocabulary và không truy cập classifier/training API.

Danh sách lựa chọn là các giá trị **đã quan sát trong reference**, không phải một
bản sao bộ quy tắc validation hoặc vocabulary tự fit. Với artifact hiện tại,
đây là các category của tập train. Nếu fitted encoder tương lai chấp nhận thêm
category không xuất hiện trong reference, UI sẽ chưa cung cấp các lựa chọn đó.
Fitted preprocessing bên trong Store vẫn là nguồn validation có thẩm quyền.

`validate_customer(resources, values)` yêu cầu đúng 19 key, từ chối cả metadata
ID/Churn, sau đó gọi `ReferenceStore.prepare_query()` đúng một lần. Hàm này tái
sử dụng kiểm tra kiểu, numerical hữu hạn/không âm, integer tenure, SeniorCitizen
0/1, category membership và cleaning TotalCharges hiện có. TotalCharges thiếu
chỉ chuyển thành 0 khi tenure=0; tenure dương thì báo lỗi. Không suy ra TotalCharges
từ hai feature khác. Không bổ sung ràng buộc liên trường chưa có trong backend.

Store có transform query bằng fitted preprocessor để kiểm tra dữ liệu. UI chỉ
giữ `prepared.features` gốc đã clean, không giữ vector làm input và không tự
scaling hoặc one-hot encoding.

Thông báo validation được dịch từ các message đã biết. Exception không nhận diện
được được thay bằng thông báo chung, không xuất raw values hoặc nội dung lỗi
transformer ra giao diện.

## Session, artifact và rerun

Resources dùng `ReferenceStore` v2, mặc định
`artifacts/references/telco-train-v2.zip`, với external SHA-256 pin. Có thể cấu hình
`CHURN_LLM_REFERENCE_ARTIFACT` và `CHURN_LLM_REFERENCE_SHA256`; xem
[cấu hình runtime](llm_runtime_v2.md). Loader restore scaler/encoder từ JSON,
không unpickle classifier hoặc fallback v1.

Cache identity gồm path/stat fingerprint và pin. Đổi path, file hoặc pin sẽ xóa
resources/snapshot/result cũ; load lại phải qua xác thực artifact. Chỉ submit gọi
inference, không tự gọi lại trên rerun. Draft chưa submit không phải snapshot đã
kiểm tra. Lỗi validation/provider/artifact được sanitize.

Session giữ resources, validated input và kết quả; không giữ API key, HTTP client,
Router hoặc task. Không sửa `active_scoring_context`. Kết quả hiển thị probability
chưa hiệu chuẩn, reason, provider và reference version. Xem
[luồng prediction](llm_streamlit_prediction.md) và [app độc lập](llm_standalone_app.md).

## Kiểm thử offline

```bash
python3 -m pytest tests/test_llm_input.py tests/test_llm_input_page.py tests/test_llm_streamlit_e2e.py -q
python3 -m pytest tests/test_llm_runtime_v2.py -q
```

Unit/AppTest dùng v2 tổng hợp, validator/retrieval/router thật và fake providers.
Các kiểm tra artifact private được skip rõ ràng khi thiếu tài nguyên ngoài Git;
`CHURN_GOLDEN_REQUIRE_ASSETS=1` bắt buộc có chúng và không cho skip do thiếu artifact.
Không tái tạo golden expected hoặc sửa artifact/dữ liệu dự án trong test.
