# Scoring feature

Feature tự chứa bằng React + TypeScript và CSS thuần, không kèm khung ứng dụng.
`types.ts` giữ nguyên tên trường JSON của `Prediction` và `PredictionBatch` phía backend;
`scored_at` là chuỗi ISO 8601 có timezone khi truyền qua HTTP.

## Gắn vào khung React của Cường

Từ component trong `src/`, import và truyền run ID lấy từ router của app:

```tsx
import { ScoringScreen } from "./features/scoring/ScoringScreen";

<ScoringScreen runId={runId} />
```

Component tự import `scoring.css`; app cần hỗ trợ TSX và import CSS.
`fetchScoringRun(id)` dùng native `fetch`, gọi GET `/scoring/runs/{id}` cùng origin.
App cần proxy đường dẫn này đến FastAPI hoặc phục vụ frontend/API cùng origin.
Feature không tạo router frontend và không đăng ký router backend.

Khi API chưa sẵn, bật cờ ở nơi gắn component:

```tsx
<ScoringScreen runId="demo-scoring-run" useDemoData />
```

`useDemoData` mặc định là `false`. Khi bật, component dùng `demo.ts`, không gọi API
và luôn hiển thị nhãn **DEMO — Dữ liệu giả** cùng dữ liệu mẫu.
Mọi ID, xác suất và thời gian trong mẫu đều là giả; không có kết quả huấn luyện thật.
Lỗi API không tự chuyển sang demo.

## Hành vi

- Mặc định sắp xếp xác suất giảm dần; có thể đổi sang tăng dần.
- Ngưỡng mặc định 0%; lọc bằng xác suất gốc `>= ngưỡng / 100` trước khi làm tròn hiển thị.
- Hiển thị phần trăm theo `vi-VN`; thời gian theo múi giờ thiết bị, kèm tên múi giờ.
- Có loading, lỗi kèm nút thử lại, dataset rỗng và kết quả lọc rỗng.
- Hủy request cũ khi đổi run ID, đổi chế độ demo hoặc tháo component.

Chưa có `package.json`, cấu hình lint/test hoặc runtime React trong repo tại thời điểm
thêm feature; chạy lint/test của app sau khi Cường gắn vào khung React của nhóm.
