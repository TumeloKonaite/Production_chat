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
        DECLARE
            embedding_type text;
        BEGIN
            IF to_regclass('langchain_pg_embedding') IS NOT NULL THEN
                EXECUTE '
                    CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id
                    ON langchain_pg_embedding (collection_id)
                ';

                SELECT format_type(a.atttypid, a.atttypmod)
                INTO embedding_type
                FROM pg_attribute AS a
                JOIN pg_class AS c
                  ON a.attrelid = c.oid
                JOIN pg_namespace AS n
                  ON c.relnamespace = n.oid
                WHERE c.relname = 'langchain_pg_embedding'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END, n.nspname
                LIMIT 1;

                IF embedding_type ~ '^vector\\([0-9]+\\)$' THEN
                    EXECUTE '
                        CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_embedding_cosine_ivfflat
                        ON langchain_pg_embedding
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    ';
                END IF;
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
