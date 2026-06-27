"""add vector retrieval support

Revision ID: 20260626_0003
Revises: 20260626_0002
Create Date: 2026-06-26 18:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType


class VectorColumn(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_: object) -> str:
        return f"VECTOR({self.dimensions})"

# revision identifiers, used by Alembic.
revision = "20260626_0003"
down_revision = "20260626_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "knowledge_chunks",
        sa.Column("source_type", sa.String(length=50), nullable=False, server_default="markdown"),
    )
    op.execute("UPDATE knowledge_chunks SET source_type = 'markdown' WHERE source_type IS NULL")
    op.alter_column("knowledge_chunks", "source_type", server_default=None)

    if dialect_name == "postgresql":
        op.add_column(
            "knowledge_chunks",
            sa.Column("embedding", VectorColumn(1536), nullable=True),
        )
        op.execute("DELETE FROM knowledge_chunks")
        op.alter_column("knowledge_chunks", "embedding", nullable=False)
    else:
        op.add_column(
            "knowledge_chunks",
            sa.Column("embedding", sa.JSON(), nullable=False, server_default="[]"),
        )
        op.alter_column("knowledge_chunks", "embedding", server_default=None)

    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("retrieved_sources", sa.JSON(), nullable=False),
        sa.Column("similarity_scores", sa.JSON(), nullable=False),
        sa.Column("used_fallback", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_retrieval_logs_conversation_id"), "retrieval_logs", ["conversation_id"], unique=False)
    op.create_index(op.f("ix_retrieval_logs_message_id"), "retrieval_logs", ["message_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_retrieval_logs_message_id"), table_name="retrieval_logs")
    op.drop_index(op.f("ix_retrieval_logs_conversation_id"), table_name="retrieval_logs")
    op.drop_table("retrieval_logs")
    op.drop_column("knowledge_chunks", "embedding")
    op.drop_column("knowledge_chunks", "source_type")
