"""add knowledge chunks

Revision ID: 20260626_0002
Revises: 20260625_0001
Create Date: 2026-06-26 10:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260626_0002"
down_revision = "20260625_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("section", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_chunks_source"), "knowledge_chunks", ["source"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_chunks_source"), table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
