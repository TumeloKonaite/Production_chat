"""add message model metadata

Revision ID: 20260628_0006
Revises: 20260627_0005
Create Date: 2026-06-28 10:20:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260628_0006"
down_revision = "20260627_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("model_provider", sa.String(length=50), nullable=True))
    op.add_column("messages", sa.Column("model_name", sa.String(length=255), nullable=True))
    op.add_column("messages", sa.Column("model_config_id", sa.String(length=255), nullable=True))
    op.add_column("messages", sa.Column("retrieval_config", sa.String(length=100), nullable=True))
    op.add_column("messages", sa.Column("estimated_cost_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "estimated_cost_usd")
    op.drop_column("messages", "retrieval_config")
    op.drop_column("messages", "model_config_id")
    op.drop_column("messages", "model_name")
    op.drop_column("messages", "model_provider")
