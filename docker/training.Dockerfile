# Dùng đúng phiên bản Python 3.14.2 của môi trường chạy pytest trên host.
FROM python:3.14.2-slim

# Không tạo __pycache__ khi chạy container.
ENV PYTHONDONTWRITEBYTECODE=1
# Đẩy log Python ra ngay để theo dõi quá trình huấn luyện.
ENV PYTHONUNBUFFERED=1
# Ghi MLflow vào SQLite trong thư mục có thể mount; có thể ghi đè bằng -e.
ENV MLFLOW_TRACKING_URI=sqlite:////workspace/mlflow.db

# Cài thư viện OpenMP cần cho CatBoost trên Linux và xóa cache apt.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Tách nơi cài package khỏi thư mục dữ liệu/output sẽ mount.
WORKDIR /opt/churn-retention
# Chép metadata và README mà setuptools cần để đóng gói.
COPY pyproject.toml README.md ./
# Chép mã nguồn Python; không chép dataset hoặc artifact vào image.
COPY src/ ./src/
# Cài bằng pip; pyproject chưa tách nhóm nên cài toàn bộ dependency khai báo.
RUN python -m pip install --no-cache-dir .

# Tạo user thường và thư mục output có quyền ghi trước khi hạ quyền.
RUN useradd --create-home --uid 10001 trainer && mkdir -p /workspace/artifacts/bundles && chown -R trainer:trainer /workspace
# Mọi lệnh runtime chạy bằng user không phải root.
USER trainer
# Bundle mặc định và file MLflow nằm dưới thư mục có thể mount này.
WORKDIR /workspace
# Chỉ gọi CLI để nhận tham số; không tự cung cấp dữ liệu hoặc chạy train khi build.
ENTRYPOINT ["python", "-m", "churn_retention.entrypoints.cli.train"]
