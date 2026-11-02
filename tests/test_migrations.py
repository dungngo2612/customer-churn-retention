from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine


def test_alembic_migrations(tmp_path: Path) -> None:
    """Kiểm tra các file migration upgrade head và downgrade base trên file SQLite độc lập."""
    root_dir = Path(__file__).resolve().parents[1]
    ini_path = root_dir / "alembic.ini"

    if not ini_path.exists():
        pytest.skip("alembic.ini không tồn tại.")

    db_file = tmp_path / "test_migration.db"
    sqlite_url = f"sqlite:///{db_file}"

    config = Config(str(ini_path))
    config.set_main_option("sqlalchemy.url", sqlite_url)

    # Khởi tạo engine SQLite và tiêm kết nối trực tiếp vào Alembic
    engine = create_engine(sqlite_url)
    with engine.connect() as connection:
        config.attributes["connection"] = connection

        # Chạy migration nâng cấp lên head
        command.upgrade(config, "head")

        # Chạy migration hạ cấp về base
        command.downgrade(config, "base")

    engine.dispose()
