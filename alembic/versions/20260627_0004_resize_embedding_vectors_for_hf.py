"""resize embedding vectors for huggingface model

Revision ID: 20260627_0004
Revises: 20260626_0003
Create Date: 2026-06-27 11:30:00
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260627_0004"
down_revision = "20260626_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DELETE FROM knowledge_chunks")
    op.execute("ALTER TABLE knowledge_chunks ALTER COLUMN embedding TYPE VECTOR(384)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DELETE FROM knowledge_chunks")
    op.execute("ALTER TABLE knowledge_chunks ALTER COLUMN embedding TYPE VECTOR(1536)")
