import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import churn_retention.adapters.database.models as _models  # noqa: F401  # Nạp các model để Alembic nhận diện schema
from churn_retention.adapters.database.session import Base

# 1. Nạp cấu hình DATABASE_URL và Base model của dự án
from churn_retention.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# 2. Ghi đè URL kết nối bằng DATABASE_URL lấy từ config/.env
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 3. Gán metadata của Base để kích hoạt tính năng tự động phát hiện schema
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # 1. Nếu bài test đã truyền sẵn connection thì dùng luôn kết nối đó
    connectable = config.attributes.get("connection", None)

    if connectable is not None:
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            render_as_batch=True,  # Bắt buộc cho SQLite
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    # 2. Quy trình mặc định khi chạy ngoài test (CLI / Production / CI)
    configuration = config.get_section(config.config_ini_section, {})
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        configuration["sqlalchemy.url"] = database_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
