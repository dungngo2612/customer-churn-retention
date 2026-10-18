from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from churn_retention.config import settings

# Khởi tạo engine kết nối PostgreSQL qua psycopg3
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Lớp nền cho toàn bộ các model bảng dữ liệu."""

    pass


def get_db() -> Generator[Session, None, None]:
    """Dependency cung cấp phiên làm việc database cho API."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
