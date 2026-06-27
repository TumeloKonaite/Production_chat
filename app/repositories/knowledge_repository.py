from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import KnowledgeChunk, RetrievalLog

if TYPE_CHECKING:
    from app.knowledge.ingestion.chunker import ChunkedDocument


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
                source_type=chunk.source_type,
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

    def log_retrieval(
        self,
        *,
        conversation_id: str,
        message_id: str,
        query: str,
        top_k: int,
        retrieved_chunk_ids: Sequence[str],
        retrieved_sources: Sequence[str],
        similarity_scores: Sequence[float],
        used_fallback: bool,
    ) -> RetrievalLog:
        log_entry = RetrievalLog(
            conversation_id=conversation_id,
            message_id=message_id,
            query=query,
            top_k=top_k,
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            retrieved_sources=list(retrieved_sources),
            similarity_scores=[float(score) for score in similarity_scores],
            used_fallback=used_fallback,
        )

        try:
            self._session.add(log_entry)
            self._session.commit()
            self._session.refresh(log_entry)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeRepositoryError() from exc

        return log_entry

    def _run_scalar_query(self, statement) -> Sequence[KnowledgeChunk]:
        try:
            return list(self._session.scalars(statement))
        except SQLAlchemyError as exc:
            raise KnowledgeRepositoryError() from exc
