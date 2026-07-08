"""add knowledge ingestion jobs

Revision ID: 20260708_0013
Revises: 20260707_0012
Create Date: 2026-07-08 14:35:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260708_0013"
down_revision = "20260707_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_ingestion_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("file_id", sa.String(length=36), nullable=True),
        sa.Column("storage_provider", sa.String(length=50), nullable=True),
        sa.Column("storage_path", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("embedding_provider", sa.String(length=50), nullable=False),
        sa.Column("embedding_model", sa.String(length=300), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("content_checksum", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["knowledge_files.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_knowledge_ingestion_jobs_file_id"),
        "knowledge_ingestion_jobs",
        ["file_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_ingestion_jobs_idempotency_key"),
        "knowledge_ingestion_jobs",
        ["idempotency_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_ingestion_jobs_source_id"),
        "knowledge_ingestion_jobs",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_ingestion_jobs_source_type"),
        "knowledge_ingestion_jobs",
        ["source_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_ingestion_jobs_status"),
        "knowledge_ingestion_jobs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_ingestion_jobs_status"), table_name="knowledge_ingestion_jobs")
    op.drop_index(op.f("ix_knowledge_ingestion_jobs_source_type"), table_name="knowledge_ingestion_jobs")
    op.drop_index(op.f("ix_knowledge_ingestion_jobs_source_id"), table_name="knowledge_ingestion_jobs")
    op.drop_index(op.f("ix_knowledge_ingestion_jobs_idempotency_key"), table_name="knowledge_ingestion_jobs")
    op.drop_index(op.f("ix_knowledge_ingestion_jobs_file_id"), table_name="knowledge_ingestion_jobs")
    op.drop_table("knowledge_ingestion_jobs")
