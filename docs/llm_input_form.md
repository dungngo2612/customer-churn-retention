# Form LLM/RAG — bước 6.2

Trang mới `apps/streamlit/pages/2_Predict.py` được phát hiện qua thư mục `pages/`
của entry point hiện có `apps/streamlit/scoring_page.py`.

Chạy ứng dụng bằng lệnh hiện có:

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit/scoring_page.py
```

Trang chỉ nhập và kiểm tra dữ liệu. Nút **Dự đoán bằng LLM** chưa gọi Predictor,
Router, Groq hay Gemini; không cần API key trong bước này. Không có lựa chọn
Logistic Regression và không hiển thị nhãn/probability giả.

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

## Session và rerun

Form gom input tới lúc submit. Store và form specification được giữ trong
`llm_input.resources` theo session, không load artifact hoặc dựng lại danh mục
ở mỗi rerun. Đường dẫn artifact hiện tại được giải từ repository root:
`artifacts/references/telco-train-v1.zip`.

Chỉ nhánh submit gọi validation. Thành công lưu một dict mới tại
`llm_input.validated_customer` và hiển thị thông báo sẵn sàng. Submit sai xóa
snapshot cũ, thay thông báo thành lỗi. Khi rerun, trang chỉ render thông báo của
lần gửi gần nhất, không gọi validation hoặc inference lại. Draft chưa submit
không được coi là snapshot đã kiểm tra.

Không ghi input ra file, không lưu API key, HTTP client, Router, task hoặc LLM
Predictor trong session. Không đọc/ghi `active_scoring_context` của CSV/SHAP.

## Phụ thuộc legacy đã có

UI mới chỉ phụ thuộc vào ReferenceStore và exception retrieval hiện có. Tuy nhiên,
reference artifact hiện chứa `bundle/model.pkl`, feature_schema và metadata của
bundle Logistic Regression. Store loader đang unpickle pipeline, kiểm tra loại
model và lấy fitted preprocessing/schema của bundle này. Đây là phụ thuộc backend
đã có; bước 6.2 không sửa hoặc mở rộng việc dùng classifier.

Nếu bỏ nhánh tự huấn luyện trong tương lai, cần refactor riêng định dạng artifact
và loader để preprocessing/schema độc lập với classifier. Không thực hiện việc
đó hoặc tạo adapter/use case Logistic Regression trong bước này.

## Kiểm thử và bước tiếp theo

```bash
python3 -m pytest tests/test_llm_input.py tests/test_llm_input_page.py -q
python3 -m pytest tests -q
python3 -m ruff check src apps tests
python3 -m ruff format --check src apps tests
python3 -m mypy src/churn_retention/entrypoints/streamlit/llm_input.py apps/streamlit/pages/2_Predict.py tests/test_llm_input.py tests/test_llm_input_page.py
```

Unit test dùng reference fixture tổng hợp và validator thật. AppTest dùng Store
được inject, cấm provider/client constructor, LLM inference và classifier
fit/predict/predict_proba. Trang CSV/SHAP được kiểm tra khởi tạo, không chạy model.
Không gọi API thật hoặc sửa artifact dự án trong test.

Bước 6.3 mới kết nối snapshot hợp lệ với LLMChurnPredictor, quản lý async/client
lifecycle, kiểm tra API key trong môi trường server, hiển thị output và lỗi thật.
Cache hiện tại giữ một version Store trong session; cần xử lý invalidation theo
version/fingerprint trong bước kết nối đó. Chưa triển khai các chức năng này.
