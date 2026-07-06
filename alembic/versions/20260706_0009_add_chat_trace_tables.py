"""add chat trace tables

Revision ID: 20260706_0009
Revises: 20260629_0008
Create Date: 2026-07-06 19:45:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260706_0009"
down_revision = "20260629_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_traces",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("request_id", sa.String(length=255), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("llm_provider", sa.String(length=50), nullable=True),
        sa.Column("llm_model", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=50), nullable=True),
        sa.Column("retriever_type", sa.String(length=100), nullable=True),
        sa.Column("embedding_provider", sa.String(length=50), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
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
            "status IN ('started', 'success', 'error', 'cancelled')",
            name="ck_chat_traces_status_valid",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.alter_column("chat_traces", "metadata", server_default=None)
    op.create_index("ix_chat_traces_conversation_id", "chat_traces", ["conversation_id"], unique=False)
    op.create_index("ix_chat_traces_created_at", "chat_traces", ["created_at"], unique=False)
    op.create_index("ix_chat_traces_status", "chat_traces", ["status"], unique=False)
    op.create_index("ix_chat_traces_llm_model", "chat_traces", ["llm_model"], unique=False)

    op.create_table(
        "chat_trace_steps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=36), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column(
            "input_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "output_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "status IN ('started', 'success', 'error', 'cancelled')",
            name="ck_chat_trace_steps_status_valid",
        ),
        sa.CheckConstraint(
            "step_type IN ('request_received', 'retrieval_started', 'retrieval_completed', "
            "'prompt_built', 'llm_call_started', 'llm_call_completed', 'response_generated', 'error')",
            name="ck_chat_trace_steps_type_valid",
        ),
        sa.ForeignKeyConstraint(["trace_id"], ["chat_traces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trace_id", "step_index", name="uq_chat_trace_steps_trace_step_index"),
    )
    op.alter_column("chat_trace_steps", "metadata", server_default=None)
    op.create_index("ix_chat_trace_steps_trace_id", "chat_trace_steps", ["trace_id"], unique=False)
    op.create_index("ix_chat_trace_steps_step_type", "chat_trace_steps", ["step_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chat_trace_steps_step_type", table_name="chat_trace_steps")
    op.drop_index("ix_chat_trace_steps_trace_id", table_name="chat_trace_steps")
    op.drop_table("chat_trace_steps")

    op.drop_index("ix_chat_traces_llm_model", table_name="chat_traces")
    op.drop_index("ix_chat_traces_status", table_name="chat_traces")
    op.drop_index("ix_chat_traces_created_at", table_name="chat_traces")
    op.drop_index("ix_chat_traces_conversation_id", table_name="chat_traces")
    op.drop_table("chat_traces")
