import os
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Import trực tiếp model để lấy đúng metadata của bảng dataset_metadata
from churn_retention.adapters.database.models import DatasetMetadataModel


@pytest.fixture(scope="module")
def frozen_rag_oracle() -> Path:
    """CI has public CSVs; the immutable original oracle remains a local asset."""
    path = Path(__file__).resolve().parents[1] / "artifacts/references/telco-train-v1.zip"
    if not path.is_file():
        message = "Frozen private RAG oracle is not available; never regenerate expected values."
        if os.environ.get("CHURN_GOLDEN_REQUIRE_ASSETS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return path


@pytest.fixture(scope="function")
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    """Tạo database SQLite tạm thời trên ổ cứng cho từng bài test, tự dọn dẹp khi xong."""
    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )

    # Dùng trực tiếp metadata của DatasetMetadataModel để đảm bảo bảng luôn được tạo
    DatasetMetadataModel.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        DatasetMetadataModel.metadata.drop_all(bind=engine)
        engine.dispose()
