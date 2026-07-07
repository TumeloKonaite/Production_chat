"""add message feedback table

Revision ID: 20260707_0010
Revises: 20260706_0009
Create Date: 2026-07-07 10:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260707_0010"
down_revision = "20260706_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=36), nullable=True),
        sa.Column("rating", sa.String(length=10), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "rating IN ('up', 'down')",
            name="ck_message_feedback_rating_valid",
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["trace_id"], ["chat_traces.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", name="uq_message_feedback_message_id"),
    )
    op.alter_column("message_feedback", "metadata", server_default=None)
    op.create_index(
        "ix_message_feedback_conversation_id",
        "message_feedback",
        ["conversation_id"],
        unique=False,
    )
    op.create_index("ix_message_feedback_message_id", "message_feedback", ["message_id"], unique=False)
    op.create_index("ix_message_feedback_trace_id", "message_feedback", ["trace_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_message_feedback_trace_id", table_name="message_feedback")
    op.drop_index("ix_message_feedback_message_id", table_name="message_feedback")
    op.drop_index("ix_message_feedback_conversation_id", table_name="message_feedback")
    op.drop_table("message_feedback")
