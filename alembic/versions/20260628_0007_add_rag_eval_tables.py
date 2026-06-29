"""add rag eval tables

Revision ID: 20260628_0007
Revises: 20260628_0006
Create Date: 2026-06-28 14:50:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260628_0007"
down_revision = "20260628_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_eval_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_name", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=255), nullable=False),
        sa.Column("retrieval_config", sa.JSON(), nullable=False),
        sa.Column("retrieval_top_k", sa.Integer(), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("total_questions", sa.Integer(), nullable=False),
        sa.Column("avg_precision_at_k", sa.Float(), nullable=False),
        sa.Column("avg_recall_at_k", sa.Float(), nullable=False),
        sa.Column("avg_mrr", sa.Float(), nullable=False),
        sa.Column("avg_ndcg_at_k", sa.Float(), nullable=False),
        sa.Column("avg_context_relevance", sa.Float(), nullable=False),
        sa.Column("avg_faithfulness", sa.Float(), nullable=False),
        sa.Column("avg_answer_relevance", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_rag_eval_runs_run_name"), "rag_eval_runs", ["run_name"], unique=False)

    op.create_table(
        "rag_eval_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("question_id", sa.String(length=255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("expected_source_documents", sa.JSON(), nullable=False),
        sa.Column("retrieved_source_documents", sa.JSON(), nullable=False),
        sa.Column("precision_at_k", sa.Float(), nullable=False),
        sa.Column("recall_at_k", sa.Float(), nullable=False),
        sa.Column("mrr", sa.Float(), nullable=False),
        sa.Column("ndcg_at_k", sa.Float(), nullable=False),
        sa.Column("context_relevance_score", sa.Integer(), nullable=False),
        sa.Column("context_relevance_reason", sa.Text(), nullable=False),
        sa.Column("faithfulness_score", sa.Integer(), nullable=False),
        sa.Column("faithfulness_reason", sa.Text(), nullable=False),
        sa.Column("answer_relevance_score", sa.Integer(), nullable=False),
        sa.Column("answer_relevance_reason", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("token_usage", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["rag_eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_rag_eval_results_question_id"), "rag_eval_results", ["question_id"], unique=False)
    op.create_index(op.f("ix_rag_eval_results_run_id"), "rag_eval_results", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_rag_eval_results_run_id"), table_name="rag_eval_results")
    op.drop_index(op.f("ix_rag_eval_results_question_id"), table_name="rag_eval_results")
    op.drop_table("rag_eval_results")
    op.drop_index(op.f("ix_rag_eval_runs_run_name"), table_name="rag_eval_runs")
    op.drop_table("rag_eval_runs")
