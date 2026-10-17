# src/churn_retention/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Customer Churn Retention"
    MARGIN: float = 0.50
    HORIZON_MONTHS: int = 12

settings = Settings()


# """Application settings and environment configuration."""

# import os
# from pydantic import BaseModel


# class Settings(BaseModel):
#     """Cấu hình ứng dụng đọc từ biến môi trường."""

#     PROJECT_NAME: str = os.getenv("PROJECT_NAME", "customer-churn-retention")
#     ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
#     DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

#     API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
#     API_PORT: int = int(os.getenv("API_PORT", "8000"))

#     DATABASE_URL: str = os.getenv(
#         "DATABASE_URL", "postgresql://user:password@localhost:5432/churn_db"
#     )
#     REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
#     MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

#     DEFAULT_MARGIN: float = float(os.getenv("DEFAULT_MARGIN", "0.50"))
#     DEFAULT_HORIZON_MONTHS: int = int(os.getenv("DEFAULT_HORIZON_MONTHS", "12"))


# settings = Settings()
