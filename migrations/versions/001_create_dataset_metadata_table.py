"""create dataset_metadata table

Revision ID: 001_create_dataset_metadata
Revises:
Create Date: 2026-10-21 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_create_dataset_metadata"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_metadata",
        sa.Column("dataset_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False),
        sa.Column("total_columns", sa.Integer(), nullable=False),
        sa.Column("features_path", sa.String(length=500), nullable=True),
        sa.Column("labels_path", sa.String(length=500), nullable=True),
        sa.Column("has_labels", sa.Boolean(), nullable=False, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("dataset_metadata")
