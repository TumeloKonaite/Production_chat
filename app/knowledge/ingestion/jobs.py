from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import logging
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.repositories.knowledge_file_repository import (
    KnowledgeFileRepository,
    KnowledgeFileRepositoryError,
)
from app.repositories.knowledge_ingestion_job_repository import (
    KnowledgeIngestionJobRepository,
    KnowledgeIngestionJobRepositoryError,
)
from app.repositories.models import KnowledgeIngestionJob

from .errors import (
    KnowledgeIngestionConflictError,
    KnowledgeIngestionNotFoundError,
    KnowledgeIngestionServiceError,
)
from .loader import DEFAULT_SOURCE_DIR
from .service import KnowledgeIngestionService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.api.knowledge.schemas import KnowledgeIngestionRequest


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionTriggerResult:
    job_id: str
    status: str
    source_type: str
    file_id: str | None


@dataclass(frozen=True, slots=True)
class KnowledgeIngestionJobResult:
    job_id: str
    source_id: str
    status: str
    chunk_count: int | None
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int


class KnowledgeIngestionRunner:
    def submit(self, *, job_id: str) -> None:
        raise NotImplementedError


class LocalKnowledgeIngestionRunner(KnowledgeIngestionRunner):
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        worker: KnowledgeIngestionJobWorker,
    ) -> None:
        self._session_factory = session_factory
        self._worker = worker

    def submit(self, *, job_id: str) -> None:
        worker_thread = Thread(
            target=self._run_job_in_thread,
            kwargs={"job_id": job_id},
            name=f"knowledge-ingestion-{job_id}",
            daemon=True,
        )
        worker_thread.start()

    def _run_job_in_thread(self, *, job_id: str) -> None:
        with self._session_factory() as session:
            try:
                self._worker.run_job(session, job_id=job_id)
            except Exception:
                logger.exception(
                    "Background knowledge ingestion job failed.",
                    extra={"job_id": job_id},
                )


class ModalKnowledgeIngestionRunner(KnowledgeIngestionRunner):
    def __init__(
        self,
        *,
        app_name: str,
        function_name: str,
    ) -> None:
        self._app_name = app_name
        self._function_name = function_name

    def submit(self, *, job_id: str) -> None:
        try:
            import modal
        except ImportError as exc:
            raise KnowledgeIngestionServiceError(
                "Modal ingestion backend requires the modal package to be installed."
            ) from exc

        function = modal.Function.from_name(self._app_name, self._function_name)
        function.spawn(job_id=job_id)


class KnowledgeIngestionJobWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        ingestion_service_factory: Callable[[Settings], KnowledgeIngestionService],
        job_repository_factory: Callable[[Session], KnowledgeIngestionJobRepository] = (
            KnowledgeIngestionJobRepository
        ),
    ) -> None:
        self._settings = settings
        self._ingestion_service_factory = ingestion_service_factory
        self._job_repository_factory = job_repository_factory

    def run_job(self, session: Session, *, job_id: str) -> KnowledgeIngestionJobResult:
        repository = self._job_repository_factory(session)
        job = self._get_job(repository=repository, job_id=job_id)
        if job.status in {"completed", "skipped"}:
            return _build_job_result(job)

        started_at = datetime.now(timezone.utc)
        if not repository.mark_running(job_id=job.id, started_at=started_at):
            refreshed_job = self._get_job(repository=repository, job_id=job_id)
            return _build_job_result(refreshed_job)

        effective_settings = replace(
            self._settings,
            embedding_provider=job.embedding_provider,
            knowledge_embedding_model=job.embedding_model,
            embedding_dimension=job.embedding_dimension,
            knowledge_chunk_size=job.chunk_size,
            knowledge_chunk_overlap=job.chunk_overlap,
        )
        ingestion_service = self._ingestion_service_factory(effective_settings)
        request = _build_ingestion_request(job)

        try:
            result = ingestion_service.run(session, request=request)
        except Exception as exc:
            completed_at = datetime.now(timezone.utc)
            repository.mark_failed(
                job_id=job.id,
                completed_at=completed_at,
                error_message=str(exc) or "Knowledge ingestion failed.",
            )
            raise

        completed_at = datetime.now(timezone.utc)
        repository.mark_completed(
            job_id=job.id,
            completed_at=completed_at,
            chunk_count=result.chunks_created,
        )
        completed_job = self._get_job(repository=repository, job_id=job.id)
        return _build_job_result(completed_job)

    def _get_job(
        self,
        *,
        repository: KnowledgeIngestionJobRepository,
        job_id: str,
    ) -> KnowledgeIngestionJob:
        try:
            job = repository.get_by_id(job_id)
        except KnowledgeIngestionJobRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to load ingestion job.") from exc

        if job is None:
            raise KnowledgeIngestionNotFoundError("Knowledge ingestion job not found.")
        return job


class KnowledgeIngestionOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        runner: KnowledgeIngestionRunner,
        file_repository_factory: Callable[[Session], KnowledgeFileRepository] = KnowledgeFileRepository,
        job_repository_factory: Callable[[Session], KnowledgeIngestionJobRepository] = (
            KnowledgeIngestionJobRepository
        ),
    ) -> None:
        self._runner = runner
        self._file_repository_factory = file_repository_factory
        self._job_repository_factory = job_repository_factory

    def trigger(
        self,
        session: Session,
        *,
        request: KnowledgeIngestionRequest,
        effective_settings: Settings,
    ) -> KnowledgeIngestionTriggerResult:
        source = self._resolve_source(session=session, request=request)
        repository = self._job_repository_factory(session)
        idempotency_key = _build_idempotency_key(source=source, settings=effective_settings)

        try:
            active_job = repository.find_latest_active_by_idempotency_key(idempotency_key)
            terminal_job = repository.find_latest_terminal_by_idempotency_key(idempotency_key)
        except KnowledgeIngestionJobRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to query ingestion jobs.") from exc

        if active_job is not None:
            return KnowledgeIngestionTriggerResult(
                job_id=active_job.id,
                status="queued",
                source_type=active_job.source_type,
                file_id=active_job.file_id,
            )

        if terminal_job is not None:
            skipped_job = self._create_job(
                repository=repository,
                source=source,
                effective_settings=effective_settings,
                idempotency_key=idempotency_key,
                initial_status="skipped",
                chunk_count=terminal_job.chunk_count,
                error_message=None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            return KnowledgeIngestionTriggerResult(
                job_id=skipped_job.id,
                status="skipped",
                source_type=skipped_job.source_type,
                file_id=skipped_job.file_id,
            )

        pending_job = self._create_job(
            repository=repository,
            source=source,
            effective_settings=effective_settings,
            idempotency_key=idempotency_key,
            initial_status="pending",
            chunk_count=None,
            error_message=None,
            started_at=None,
            completed_at=None,
        )
        self._runner.submit(job_id=pending_job.id)
        return KnowledgeIngestionTriggerResult(
            job_id=pending_job.id,
            status="queued",
            source_type=pending_job.source_type,
            file_id=pending_job.file_id,
        )

    def _create_job(
        self,
        *,
        repository: KnowledgeIngestionJobRepository,
        source: ResolvedIngestionSource,
        effective_settings: Settings,
        idempotency_key: str,
        initial_status: str,
        chunk_count: int | None,
        error_message: str | None,
        started_at: datetime | None,
        completed_at: datetime | None,
    ) -> KnowledgeIngestionJob:
        job = KnowledgeIngestionJob(
            source_type=source.source_type,
            source_id=source.source_id,
            file_id=source.file_id,
            storage_provider=source.storage_provider,
            storage_path=source.storage_path,
            status=initial_status,
            chunk_count=chunk_count,
            embedding_provider=effective_settings.embedding_provider,
            embedding_model=effective_settings.knowledge_embedding_model,
            embedding_dimension=effective_settings.embedding_dimension,
            chunk_size=effective_settings.knowledge_chunk_size,
            chunk_overlap=effective_settings.knowledge_chunk_overlap,
            content_checksum=source.content_checksum,
            error_message=error_message,
            idempotency_key=idempotency_key,
            started_at=started_at,
            completed_at=completed_at,
        )
        try:
            return repository.create(job)
        except KnowledgeIngestionJobRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to persist ingestion job.") from exc

    def _resolve_source(
        self,
        *,
        session: Session,
        request: KnowledgeIngestionRequest,
    ) -> "ResolvedIngestionSource":
        if request.source_type == "uploaded_file":
            return self._resolve_uploaded_file_source(session=session, file_id=request.file_id)

        source_dir = DEFAULT_SOURCE_DIR
        return ResolvedIngestionSource(
            source_type="local_directory",
            source_id=str(source_dir),
            file_id=None,
            storage_provider="local",
            storage_path=str(source_dir),
            content_checksum=_build_directory_checksum(source_dir),
        )

    def _resolve_uploaded_file_source(
        self,
        *,
        session: Session,
        file_id: UUID | None,
    ) -> "ResolvedIngestionSource":
        if file_id is None:
            raise KnowledgeIngestionNotFoundError("Knowledge file not found.")

        repository = self._file_repository_factory(session)
        try:
            knowledge_file = repository.get_by_id(str(file_id))
        except KnowledgeFileRepositoryError as exc:
            raise KnowledgeIngestionServiceError("Unable to load knowledge file metadata.") from exc

        if knowledge_file is None:
            raise KnowledgeIngestionNotFoundError("Knowledge file not found.")
        if knowledge_file.status.casefold() == "ingesting":
            raise KnowledgeIngestionConflictError("Knowledge file is already being ingested.")

        return ResolvedIngestionSource(
            source_type="uploaded_file",
            source_id=knowledge_file.id,
            file_id=knowledge_file.id,
            storage_provider=knowledge_file.storage_provider,
            storage_path=knowledge_file.storage_path,
            content_checksum=knowledge_file.checksum,
        )


@dataclass(frozen=True, slots=True)
class ResolvedIngestionSource:
    source_type: str
    source_id: str
    file_id: str | None
    storage_provider: str | None
    storage_path: str | None
    content_checksum: str | None


def _build_idempotency_key(
    *,
    source: ResolvedIngestionSource,
    settings: Settings,
) -> str:
    digest = sha256()
    parts = (
        source.source_type,
        source.source_id,
        source.content_checksum or "",
        settings.embedding_provider,
        settings.knowledge_embedding_model,
        str(settings.embedding_dimension),
        str(settings.knowledge_chunk_size),
        str(settings.knowledge_chunk_overlap),
    )
    digest.update(":".join(parts).encode("utf-8"))
    return digest.hexdigest()


def _build_ingestion_request(job: KnowledgeIngestionJob) -> KnowledgeIngestionRequest:
    from app.api.knowledge.schemas import KnowledgeIngestionRequest

    file_id = UUID(job.file_id) if job.file_id is not None else None
    return KnowledgeIngestionRequest(source_type=job.source_type, file_id=file_id)


def _build_job_result(job: KnowledgeIngestionJob) -> KnowledgeIngestionJobResult:
    return KnowledgeIngestionJobResult(
        job_id=job.id,
        source_id=job.source_id,
        status=job.status,
        chunk_count=job.chunk_count,
        embedding_provider=job.embedding_provider,
        embedding_model=job.embedding_model,
        embedding_dimension=job.embedding_dimension,
    )


def _build_directory_checksum(source_dir: Path) -> str:
    digest = sha256()
    for path in sorted(source_dir.rglob("*.md")):
        digest.update(str(path.relative_to(source_dir)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
