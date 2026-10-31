"""Use local storage paths and record dataset schema/checksum metadata.

Revision ID: 002_dataset_storage_metadata
Revises: 001_create_dataset_metadata
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_dataset_storage_metadata"
down_revision: Union[str, None] = "001_create_dataset_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.alter_column(
        "dataset_metadata",
        "features_s3_uri",
        new_column_name="features_path",
        existing_type=sa.String(length=500),
        existing_nullable=True,
    )
    op.alter_column(
        "dataset_metadata",
        "labels_s3_uri",
        new_column_name="labels_path",
        existing_type=sa.String(length=500),
        existing_nullable=True,
    )
    op.add_column(
        "dataset_metadata",
        sa.Column(
            "schema_version",
            sa.String(length=32),
            server_default="1",
            nullable=False,
        ),
    )
    # Nullable for records created before checksums were recorded.
    op.add_column(
        "dataset_metadata",
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("dataset_metadata", "checksum_sha256")
    op.drop_column("dataset_metadata", "schema_version")
    op.alter_column(
        "dataset_metadata",
        "labels_path",
        new_column_name="labels_s3_uri",
        existing_type=sa.String(length=500),
        existing_nullable=True,
    )
    op.alter_column(
        "dataset_metadata",
        "features_path",
        new_column_name="features_s3_uri",
        existing_type=sa.String(length=500),
        existing_nullable=True,
    )
