"""add message channel and metadata

Revision ID: 20260629_0008
Revises: 20260628_0007
Create Date: 2026-06-29 09:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260629_0008"
down_revision = "20260628_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("channel", sa.String(length=50), nullable=False, server_default="web_chat"),
    )
    op.add_column(
        "messages",
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.alter_column("messages", "channel", server_default=None)
    op.alter_column("messages", "metadata", server_default=None)


def downgrade() -> None:
    op.drop_column("messages", "metadata")
    op.drop_column("messages", "channel")
