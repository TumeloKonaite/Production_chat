"""add knowledge files

Revision ID: 20260707_0012
Revises: 20260707_0011
Create Date: 2026-07-07 19:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260707_0012"
down_revision = "20260707_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_files",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_provider", sa.String(length=50), nullable=False),
        sa.Column("storage_bucket", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="uploaded"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_path"),
    )
    op.create_index(
        op.f("ix_knowledge_files_original_filename"),
        "knowledge_files",
        ["original_filename"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_files_status"),
        "knowledge_files",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_files_status"), table_name="knowledge_files")
    op.drop_index(op.f("ix_knowledge_files_original_filename"), table_name="knowledge_files")
    op.drop_table("knowledge_files")
