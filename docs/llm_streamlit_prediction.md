# Streamlit LLM/RAG — bước 6.3

Trang `apps/streamlit/pages/2_Predict.py` giữ form 19 feature của bước 6.2 và
kết nối nút **Dự đoán bằng LLM** với backend hiện có. Chạy bằng entry point cũ:

```bash
PYTHONPATH=src python3 -m streamlit run apps/streamlit/scoring_page.py
```

Quản trị viên cấu hình `GROQ_API_KEY` và `GEMINI_API_KEY` trong môi trường server
trước khi chạy Streamlit. Không nhập key vào form, session hoặc repository.
Factory hiện có khởi tạo cả hai provider nên **cần đủ cả hai key**, kể cả khi
Groq thắng. Thiếu một hoặc cả hai key sẽ báo cấu hình thiếu, không tạo client.
Không hỗ trợ chế độ chỉ một provider và không thay đổi chính sách Router.

## Submit và kết quả

1. Form thu đúng 19 feature, không có customerID hoặc nhãn Churn của query.
2. `validate_customer()` gọi `ReferenceStore.prepare_query()` theo quy tắc cleaning
   hiện có. TotalCharges thiếu chỉ hợp lệ với tenure=0 và được chuyển thành 0.
3. Snapshot đã clean được truyền vào
   `predict_customer(store, snapshot) -> LLMChurnPrediction` trong
   `entrypoints/streamlit/llm_prediction_service.py`.
4. Service gọi `LLMChurnPredictor(store, router, k=5).predict()` đúng một lần.
   Predictor thực hiện retrieval, prompt builder và JSON validation hiện có.
5. Trang hiển thị nhãn thực tế, probability, nguyên văn reason bằng `st.text`,
   provider thắng, reference_count thực dùng và reference_version.

Trong khi xử lý, trang hiển thị spinner. Probability luôn đi kèm cảnh báo
**chưa được hiệu chuẩn thành xác suất thống kê**, không dùng progress bar.
Provider hiển thị Groq/Qwen hoặc Gemini theo metadata của kết quả. Gemini thắng
có thể do hedging; giao diện không kết luận Groq đã lỗi. Không hiển thị các
khách hàng tham chiếu, ID, prompt hoặc response JSON thô.

Không có lựa chọn phương pháp, gọi classifier Logistic Regression hay fallback
sang classifier. Không sửa training, batch scoring, SHAP hoặc preprocessing.

## Vòng đời async

Bridge đồng bộ kiểm tra thread hiện tại chưa có event loop đang chạy rồi gọi
`asyncio.run()`. Nếu đã có loop, bridge từ chối bằng thông báo an toàn, không
tạo coroutine hoặc thử chạy loop lồng nhau.

Bên trong coroutine, service dùng **factory `groq_gemini_router()` hiện có**:

- Tạo Groq/Gemini client trong event loop mới của chính lượt dự đoán.
- Giữ nguyên Groq timeout 90 giây, Gemini timeout 30 giây, hedge delay 60 giây,
  overall deadline 120 giây; không thêm retry ở UI/service.
- Context Router kết thúc trước context provider: hủy và drain tác vụ còn chạy,
  thu thập exception, sau đó mới đóng client, rồi `asyncio.run()` mới đóng loop.
- Đường lỗi và cancellation vẫn đi qua context cleanup.

Hai lượt submit dùng hai loop/client lifecycle riêng, cùng Store trong RAM.
Session chỉ giữ tài nguyên reference đồng bộ, metadata file, snapshot và kết quả
đã validate/thông báo an toàn. Không cache hay lưu async client, Router, Predictor,
coroutine hoặc task. Không thêm thread/event-loop worker nền.

Deadline 120 giây là deadline định tuyến của backend; thời gian đóng context
còn bao gồm cleanup. Không hứa toàn bộ lượt UI hoàn tất tuyệt đối trong 120 giây
nếu SDK/transport không hợp tác với cancellation. Giữ nguyên contract hiện có.

## Store và fingerprint

Giữ Store và form specification tại `llm_input.resources`. Mỗi rerun kiểm tra
fingerprint nhẹ: đường dẫn resolve, device/inode, kích thước, mtime_ns và ctime_ns.
Không đọc CSV hay hash toàn bộ ZIP mỗi request. Nếu fingerprint không đổi, dùng
lại Store đã load; không fit preprocessing hoặc transform lại toàn bộ train.
Predictor tạo index từ ma trận đã transform có sẵn và chỉ transform query.

Fingerprint thay đổi sẽ bỏ Store, snapshot và kết quả cũ, sau đó nạp lại artifact
qua `ReferenceStore.load()`: kiểm tra checksum/version/schema/tính tương thích
vẫn do loader hiện có thực hiện. Kiểm tra fingerprint trước và sau lần load để
từ chối file bị thay giữa chừng. Nếu file mất/hỏng/không tương thích, không dùng
lại cache cũ và không dự đoán. Không sửa, ghi đè hoặc tạo artifact.

Stat fingerprint phát hiện chỉnh sửa/thay thế file thông thường, không phải
bằng chứng checksum mật mã cho mỗi rerun. Filesystem metadata không đáng tin có
thể cần cơ chế version/digest mạnh hơn ở bước riêng. Một lượt đang chạy dùng
Store/version đã chọn cho lượt đó; thay đổi artifact giữa lượt được phát hiện
ở rerun tiếp theo.

## Rerun và bản nháp

Chỉ nhánh submit gọi service. Rerun hoặc đổi widget không tự dự đoán. Submit mới
xóa kết quả/lỗi cũ trước khi validate, và chỉ tạo một lượt gọi service. Không có
callback submit thứ hai hoặc UI retry. Kết quả lần gửi gần nhất được giữ trong
`llm_input.prediction`, kèm bản input đã submit để đối chiếu.

Streamlit form gửi bản nháp lên server khi submit; sửa form trên trình duyệt
chưa tạo rerun thì server chưa biết input thay đổi. Vì vậy kết quả luôn có caption
**của lần gửi gần nhất**, không gắn với bản nháp đang chỉnh. Nếu server nhận input
khác trên một rerun không submit, ẩn kết quả cũ và yêu cầu gửi lại. Khi artifact
đổi, kết quả/snapshot cũ cũng bị xóa. Giữ nguyên `active_scoring_context` của trang cũ.

## Lỗi và giới hạn

Service chuyển các loại lỗi chuẩn của Router/provider thành thông báo tĩnh bằng
tiếng Việt: cấu hình, quota/rate limit, timeout/deadline, kết nối, JSON không hợp lệ
hoặc lỗi chung. UI không xuất raw SDK exception, traceback, response hay prompt.
Không log key/input. Fallback/hedging và JSON validation hoàn toàn do backend cũ.

- Reference Store vẫn có phụ thuộc legacy vào fitted preprocessing/schema trong
  Logistic Regression model bundle. Bước này không refactor hoặc mở rộng phụ thuộc.
- Probability do LLM ước lượng chưa hiệu chuẩn; không dùng như xác suất thống kê
  đã được kiểm định hoặc suy ra từ tỷ lệ nhãn trong K ví dụ.
- Vấn đề rate limit Groq đang tạm hoãn; bước này không gọi API thật để thử lại.
- Lựa chọn categorical của form vẫn là các giá trị quan sát trong reference như
  bước 6.2, không tự mở rộng vocabulary của encoder.
- Kiểm thử dùng mock SDK/fake data; không khẳng định API thật hoạt động từ kết quả
  test UI/service này.

## Kiểm thử

```bash
MPLCONFIGDIR=/tmp/churn-step62-matplotlib python3 -m pytest tests/test_llm_prediction_service.py tests/test_llm_input.py tests/test_llm_input_page.py -q
MPLCONFIGDIR=/tmp/churn-step62-matplotlib python3 -m pytest tests -q
python3 -m ruff check src apps tests
python3 -m ruff format --check src apps tests
python3 -m mypy src/churn_retention/entrypoints/streamlit/llm_prediction_service.py src/churn_retention/entrypoints/streamlit/llm_input.py apps/streamlit/pages/2_Predict.py tests/test_llm_prediction_service.py tests/test_llm_input.py tests/test_llm_input_page.py
```

Service tests chạy provider/router/predictor thật với SDK mock và chặn HTTP send.
Kiểm tra fallback do 429/timeout/kết nối/JSON sai, cả hai lỗi, key thiếu, deadline,
cancellation, task drain trước client close, loop riêng mỗi lượt, dùng lại Store
và không đọc CSV/fit model. AppTest kiểm tra form, submit/rerun, kết quả và thông
báo lỗi an toàn; provider/client và classifier calls bị chặn.
