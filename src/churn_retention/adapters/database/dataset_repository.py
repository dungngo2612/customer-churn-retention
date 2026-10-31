from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from churn_retention.adapters.database.models import DatasetMetadataModel


class DatasetRepository:
    """Repository quản lý các thao tác truy xuất bảng dataset_metadata."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, dataset: DatasetMetadataModel) -> DatasetMetadataModel:
        """Lưu bản ghi metadata mới vào cơ sở dữ liệu."""
        try:
            self.session.add(dataset)
            self.session.commit()
            self.session.refresh(dataset)
        except Exception:
            self.session.rollback()
            raise
        return dataset

    def get_by_id(self, dataset_id: str) -> Optional[DatasetMetadataModel]:
        """Lấy thông tin metadata theo dataset_id."""
        stmt = select(DatasetMetadataModel).where(DatasetMetadataModel.dataset_id == dataset_id)
        return self.session.scalars(stmt).first()

    def list_all(self) -> List[DatasetMetadataModel]:
        """Lấy danh sách toàn bộ datasets, sắp xếp theo thời gian tạo mới nhất."""
        stmt = select(DatasetMetadataModel).order_by(DatasetMetadataModel.created_at.desc())
        return list(self.session.scalars(stmt).all())
