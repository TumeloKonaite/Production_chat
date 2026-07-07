from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy.orm import Session

from app.config import DEFAULT_KNOWLEDGE_CHUNK_OVERLAP, DEFAULT_KNOWLEDGE_CHUNK_SIZE
from app.knowledge.ingestion.chunker import chunk_markdown_document
from app.knowledge.ingestion.cleaner import clean_markdown_text
from app.knowledge.ingestion.loader import SourceDocument, load_source_documents
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.retrieval import RetrievalService


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source: str
    chunk_count: int


def ingest_documents(
    session: Session,
    retrieval_service: RetrievalService,
    *,
    documents: Sequence[SourceDocument],
    ingested_at: datetime | None = None,
    chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
) -> tuple[list[SourceDocument], list[IngestionResult]]:
    repository = KnowledgeRepository(session)
    # Use one ingestion timestamp across the full run so all replaced chunks can
    # be traced back to the same refresh operation.
    ingestion_time = ingested_at or datetime.now(timezone.utc)
    results: list[IngestionResult] = []
    indexed_chunks = []

    for document in documents:
        # Clean first so the chunker works from predictable markdown instead of
        # whatever spacing happened to be in the source file.
        cleaned_text = clean_markdown_text(document.text)
        cleaned_document = SourceDocument(
            source=document.source,
            text=cleaned_text,
            updated_at=document.updated_at,
            chunk_source_type=document.chunk_source_type,
            metadata=dict(document.metadata),
        )

        # Chunk the cleaned document, then replace any previously stored chunks
        # for that source so re-ingestion updates instead of duplicating rows.
        chunks = chunk_markdown_document(
            cleaned_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        stored_chunks = repository.replace_source_chunks(
            source=document.source,
            chunks=chunks,
            ingested_at=ingestion_time,
        )
        indexed_chunks.extend(stored_chunks)
        results.append(IngestionResult(source=document.source, chunk_count=len(chunks)))

    retrieval_service.replace_all_chunks(indexed_chunks)
    return list(documents), results


def ingest_knowledge(
    session: Session,
    retrieval_service: RetrievalService,
    *,
    source_dir: Path | None = None,
    ingested_at: datetime | None = None,
    chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
) -> tuple[list[SourceDocument], list[IngestionResult]]:
    documents = load_source_documents(source_dir)
    return ingest_documents(
        session,
        retrieval_service,
        documents=documents,
        ingested_at=ingested_at,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
