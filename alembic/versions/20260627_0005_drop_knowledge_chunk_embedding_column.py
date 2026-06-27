"""drop knowledge chunk embedding column

Revision ID: 20260627_0005
Revises: 20260627_0004
Create Date: 2026-06-27 12:15:00
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
revision = "20260627_0005"
down_revision = "20260627_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("knowledge_chunks", "embedding")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.add_column("knowledge_chunks", sa.Column("embedding", VectorColumn(384), nullable=True))
        op.execute("DELETE FROM knowledge_chunks")
        op.alter_column("knowledge_chunks", "embedding", nullable=False)
    else:
        op.add_column(
            "knowledge_chunks",
            sa.Column("embedding", sa.JSON(), nullable=False, server_default="[]"),
        )
        op.alter_column("knowledge_chunks", "embedding", server_default=None)
