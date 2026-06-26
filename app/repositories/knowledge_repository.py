from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.knowledge.ingestion.chunker import ChunkedDocument
from app.repositories.models import KnowledgeChunk


class KnowledgeRepositoryError(Exception):
    """Raised when knowledge-chunk persistence fails."""


class KnowledgeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_source_chunks(
        self,
        *,
        source: str,
        chunks: Sequence[ChunkedDocument],
        ingested_at: datetime | None = None,
    ) -> list[KnowledgeChunk]:
        chunk_timestamp = ingested_at or datetime.now(timezone.utc)
        records = [
            KnowledgeChunk(
                source=chunk.source,
                section=chunk.section,
                content=chunk.content,
                chunk_metadata=chunk.metadata,
                updated_at=chunk_timestamp,
            )
            for chunk in chunks
        ]

        try:
            self._session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source == source))
            self._session.add_all(records)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeRepositoryError() from exc

        return records

    def list_by_source(self, source: str) -> Sequence[KnowledgeChunk]:
        statement = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.source == source)
            .order_by(KnowledgeChunk.section.asc(), KnowledgeChunk.created_at.asc(), KnowledgeChunk.id.asc())
        )
        return self._run_scalar_query(statement)

    def list_all(self) -> Sequence[KnowledgeChunk]:
        statement = select(KnowledgeChunk).order_by(
            KnowledgeChunk.source.asc(),
            KnowledgeChunk.section.asc(),
            KnowledgeChunk.created_at.asc(),
            KnowledgeChunk.id.asc(),
        )
        return self._run_scalar_query(statement)

    def search(self, query: str, *, limit: int = 20) -> Sequence[KnowledgeChunk]:
        search_text = query.strip()
        if not search_text:
            return []

        pattern = f"%{search_text}%"
        statement = (
            select(KnowledgeChunk)
            .where(
                or_(
                    KnowledgeChunk.content.ilike(pattern),
                    KnowledgeChunk.section.ilike(pattern),
                    KnowledgeChunk.source.ilike(pattern),
                )
            )
            .order_by(KnowledgeChunk.updated_at.desc(), KnowledgeChunk.created_at.desc())
            .limit(limit)
        )
        return self._run_scalar_query(statement)

    def _run_scalar_query(self, statement) -> Sequence[KnowledgeChunk]:
        try:
            return list(self._session.scalars(statement))
        except SQLAlchemyError as exc:
            raise KnowledgeRepositoryError() from exc
