from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from churn_retention.adapters.database.session import Base


class DatasetMetadataModel(Base):
    """Bảng lưu trữ thông tin nhận diện và siêu dữ liệu của dataset."""

    __tablename__ = "dataset_metadata"

    dataset_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    total_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    has_labels: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    features_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    labels_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1", nullable=False)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
