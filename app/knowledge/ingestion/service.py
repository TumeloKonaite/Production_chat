from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.knowledge.ingestion.errors import KnowledgeIngestionServiceError
from app.knowledge.ingestion.ingest import ingest_knowledge
from app.repositories.db.base import Base
from app.services.retrieval import RetrievalService


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionDocumentResult:
    source: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionRunResult:
    status: str
    documents_loaded: int
    results: list[KnowledgeIngestionDocumentResult]


class KnowledgeIngestionService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        source_dir: Path | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._source_dir = source_dir

    def run(self, session: Session) -> KnowledgeIngestionRunResult:
        try:
            documents, results = ingest_knowledge(
                session,
                self._retrieval_service,
                source_dir=self._source_dir,
            )
        except Exception as exc:
            raise KnowledgeIngestionServiceError() from exc

        return KnowledgeIngestionRunResult(
            status="ok",
            documents_loaded=len(documents),
            results=[
                KnowledgeIngestionDocumentResult(
                    source=result.source,
                    chunk_count=result.chunk_count,
                )
                for result in results
            ],
        )


def prepare_knowledge_ingestion_storage(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
