from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import DEFAULT_KNOWLEDGE_CHUNK_OVERLAP, DEFAULT_KNOWLEDGE_CHUNK_SIZE
from app.knowledge.ingestion.errors import (
    KnowledgeIngestionConflictError,
    KnowledgeIngestionGoneError,
    KnowledgeIngestionNotFoundError,
    KnowledgeIngestionServiceError,
)
from app.knowledge.ingestion.ingest import ingest_documents, ingest_knowledge
from app.knowledge.ingestion.uploaded_file_loader import UploadedKnowledgeFileLoader
from app.repositories.db.base import Base
from app.repositories.knowledge_file_repository import (
    KnowledgeFileRepository,
    KnowledgeFileRepositoryError,
)
from app.repositories.models import KnowledgeFile
from app.services.retrieval import RetrievalService

if TYPE_CHECKING:
    from app.api.knowledge.schemas import KnowledgeIngestionRequest


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionDocumentResult:
    source: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionRunResult:
    status: str
    source_type: str
    file_id: str | None
    documents_loaded: int
    chunks_created: int
    chunks_updated: int
    chunks_skipped: int
    results: list[KnowledgeIngestionDocumentResult]


class KnowledgeIngestionService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        source_dir: Path | None = None,
        uploaded_file_loader: UploadedKnowledgeFileLoader | None = None,
        knowledge_file_repository_factory: Callable[
            [Session], KnowledgeFileRepository
        ] = KnowledgeFileRepository,
        chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._source_dir = source_dir
        self._uploaded_file_loader = uploaded_file_loader
        self._knowledge_file_repository_factory = knowledge_file_repository_factory
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def run(
        self,
        session: Session,
        request: KnowledgeIngestionRequest | None = None,
    ) -> KnowledgeIngestionRunResult:
        source_type = "local_directory"
        file_id: str | None = None
        if request is not None:
            source_type = request.source_type
            if request.file_id is not None:
                file_id = str(request.file_id)

        if source_type == "uploaded_file":
            if file_id is None:
                raise KnowledgeIngestionServiceError("Uploaded file ingestion requires a file_id.")
            return self._run_uploaded_file(session, file_id=file_id)

        try:
            documents, results = ingest_knowledge(
                session,
                self._retrieval_service,
                source_dir=self._source_dir,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        except KnowledgeIngestionServiceError:
            raise
        except Exception as exc:
            raise KnowledgeIngestionServiceError() from exc

        chunks_created = sum(result.chunk_count for result in results)
        return KnowledgeIngestionRunResult(
            status="ok",
            source_type="local_directory",
            file_id=None,
            documents_loaded=len(documents),
            chunks_created=chunks_created,
            chunks_updated=0,
            chunks_skipped=0,
            results=[
                KnowledgeIngestionDocumentResult(
                    source=result.source,
                    chunk_count=result.chunk_count,
                )
                for result in results
            ],
        )

    def _run_uploaded_file(self, session: Session, *, file_id: str) -> KnowledgeIngestionRunResult:
        if self._uploaded_file_loader is None:
            raise KnowledgeIngestionServiceError("Uploaded file ingestion is not configured.")

        repository = self._knowledge_file_repository_factory(session)
        knowledge_file = self._get_knowledge_file(repository=repository, file_id=file_id)
        self._validate_ingestable_status(knowledge_file)
        self._persist_file_status(
            repository=repository,
            knowledge_file=knowledge_file,
            status_value="ingesting",
            error_message=None,
            ingested_at=None,
        )

        ingestion_time = datetime.now(timezone.utc)
        try:
            document = self._uploaded_file_loader.load_file(knowledge_file)
            documents, results = ingest_documents(
                session,
                self._retrieval_service,
                documents=[document],
                ingested_at=ingestion_time,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        except KnowledgeIngestionServiceError as exc:
            self._persist_failure(repository=repository, knowledge_file=knowledge_file, error=exc)
            raise
        except Exception as exc:
            wrapped_error = KnowledgeIngestionServiceError() if not str(exc) else KnowledgeIngestionServiceError(str(exc))
            self._persist_failure(
                repository=repository,
                knowledge_file=knowledge_file,
                error=wrapped_error,
            )
            raise wrapped_error from exc

        self._persist_file_status(
            repository=repository,
            knowledge_file=knowledge_file,
            status_value="ingested",
            error_message=None,
            ingested_at=ingestion_time,
        )

        chunks_created = sum(result.chunk_count for result in results)
        return KnowledgeIngestionRunResult(
            status="ingested",
            source_type="uploaded_file",
            file_id=knowledge_file.id,
            documents_loaded=len(documents),
            chunks_created=chunks_created,
            chunks_updated=0,
            chunks_skipped=0,
            results=[
                KnowledgeIngestionDocumentResult(
                    source=knowledge_file.original_filename,
                    chunk_count=result.chunk_count,
                )
                for result in results
            ],
        )

    def _get_knowledge_file(
        self,
        *,
        repository: KnowledgeFileRepository,
        file_id: str,
    ) -> KnowledgeFile:
        try:
            knowledge_file = repository.get_by_id(file_id)
        except KnowledgeFileRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to load knowledge file metadata.") from exc

        if knowledge_file is None:
            raise KnowledgeIngestionNotFoundError("Knowledge file not found.")
        return knowledge_file

    def _validate_ingestable_status(self, knowledge_file: KnowledgeFile) -> None:
        status_value = knowledge_file.status.casefold()
        if status_value in {"uploaded", "failed"}:
            return
        if status_value == "ingesting":
            raise KnowledgeIngestionConflictError("Knowledge file is already being ingested.")
        if status_value == "ingested":
            raise KnowledgeIngestionConflictError("Knowledge file has already been ingested.")
        if status_value == "deleted":
            raise KnowledgeIngestionGoneError("Knowledge file has been deleted.")
        raise KnowledgeIngestionServiceError("Knowledge file is in an unsupported status.")

    def _persist_failure(
        self,
        *,
        repository: KnowledgeFileRepository,
        knowledge_file: KnowledgeFile,
        error: Exception,
    ) -> None:
        message = str(error) or "Knowledge ingestion failed."
        self._persist_file_status(
            repository=repository,
            knowledge_file=knowledge_file,
            status_value="failed",
            error_message=message,
            ingested_at=None,
        )

    def _persist_file_status(
        self,
        *,
        repository: KnowledgeFileRepository,
        knowledge_file: KnowledgeFile,
        status_value: str,
        error_message: str | None,
        ingested_at: datetime | None,
    ) -> None:
        knowledge_file.status = status_value
        knowledge_file.error_message = error_message
        knowledge_file.ingested_at = ingested_at
        try:
            repository.save(knowledge_file)
        except KnowledgeFileRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to update knowledge file status.") from exc


def prepare_knowledge_ingestion_storage(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
