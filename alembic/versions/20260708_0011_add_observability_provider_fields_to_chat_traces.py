"""add observability provider fields to chat traces

Revision ID: 20260708_0011
Revises: 20260707_0010
Create Date: 2026-07-08 10:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260708_0011"
down_revision = "20260707_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_traces",
        sa.Column("observability_provider", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "chat_traces",
        sa.Column("external_trace_id", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_chat_traces_external_trace_id",
        "chat_traces",
        ["external_trace_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_traces_external_trace_id", table_name="chat_traces")
    op.drop_column("chat_traces", "external_trace_id")
    op.drop_column("chat_traces", "observability_provider")
