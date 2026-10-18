from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Kết nối Database PostgreSQL
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/churn_db"

    # Kết nối S3 / MinIO (Object Storage)
    S3_ENDPOINT_URL: str | None = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "churn-retention-artifacts"
    S3_REGION_NAME: str = "us-east-1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
