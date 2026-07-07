"""ensure pgvector extension and vector indexes

Revision ID: 20260707_0011
Revises: 20260707_0010
Create Date: 2026-07-07 18:45:00
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260707_0011"
down_revision = "20260707_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('langchain_pg_embedding') IS NOT NULL THEN
                EXECUTE '
                    CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id
                    ON langchain_pg_embedding (collection_id)
                ';
                EXECUTE '
                    CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_embedding_cosine_ivfflat
                    ON langchain_pg_embedding
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                ';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_langchain_pg_embedding_embedding_cosine_ivfflat")
    op.execute("DROP INDEX IF EXISTS ix_langchain_pg_embedding_collection_id")
