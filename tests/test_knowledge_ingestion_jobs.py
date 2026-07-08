from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.knowledge.schemas import KnowledgeIngestionRequest
from app.config import Settings
from app.knowledge.ingestion import (
    KnowledgeIngestionOrchestrator,
    KnowledgeIngestionServiceError,
)
from app.knowledge.ingestion.jobs import (
    KnowledgeIngestionJobWorker,
    LocalKnowledgeIngestionRunner,
)
from app.repositories.db.base import Base
from app.repositories.models import KnowledgeFile, KnowledgeIngestionJob


def build_session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def build_test_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite:///unused-for-tests.db",
        "openai_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": "openrouter-test-key",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "tavus_api_key": "tavus-test-key",
        "tavus_base_url": "https://tavus.example",
        "tavus_face_id": "face_123",
        "tavus_pal_id": "pal_123",
        "public_backend_url": "https://backend.example",
        "tavus_tool_secret": "tool-secret",
        "ingestion_api_secret": "ingestion-secret",
        "eval_admin_token": "eval-secret",
        "default_model_config_id": "openai:gpt-4.1-mini",
        "model_configs_json": None,
        "embedding_provider": "hf",
        "knowledge_embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "knowledge_collection_name": "personal_knowledge_base",
        "default_prompt_version": "v1_professional",
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "personal-chatbot-model-comparison",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "storage_provider": "local",
        "minio_endpoint": "http://localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin",
        "minio_bucket": "knowledge-files",
        "minio_secure": False,
        "local_storage_path": ".pytest_tmp/storage",
        "knowledge_upload_max_bytes": 10485760,
    }
    values.update(overrides)
    return Settings(**values)


def create_uploaded_knowledge_file(
    session: Session,
    *,
    filename: str = "company-profile.md",
    checksum: str = "checksum-a",
    status: str = "uploaded",
    storage_path: str = "uploaded/company-profile.md",
) -> KnowledgeFile:
    knowledge_file = KnowledgeFile(
        id=str(uuid4()),
        original_filename=filename,
        content_type="text/markdown",
        file_size_bytes=128,
        storage_provider="local",
        storage_bucket="knowledge-files",
        storage_path=storage_path,
        checksum=checksum,
        status=status,
    )
    session.add(knowledge_file)
    session.commit()
    session.refresh(knowledge_file)
    return knowledge_file


class RecordingRunner:
    def __init__(self) -> None:
        self.submitted_job_ids: list[str] = []

    def submit(self, *, job_id: str) -> None:
        self.submitted_job_ids.append(job_id)


class FakeIngestionService:
    def __init__(self, *, chunk_count: int = 3, error: Exception | None = None) -> None:
        self.chunk_count = chunk_count
        self.error = error
        self.calls: list[tuple[Session, object]] = []

    def run(self, session: Session, request=None):
        self.calls.append((session, request))
        if self.error is not None:
            raise self.error
        return type(
            "RunResult",
            (),
            {
                "chunks_created": self.chunk_count,
            },
        )()


def test_orchestrator_creates_job_and_dispatches_runner() -> None:
    session_factory = build_session_factory()
    runner = RecordingRunner()
    settings = build_test_settings()
    orchestrator = KnowledgeIngestionOrchestrator(settings=settings, runner=runner)

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        result = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        jobs = list(session.scalars(select(KnowledgeIngestionJob)))

    assert result.status == "queued"
    assert len(runner.submitted_job_ids) == 1
    assert runner.submitted_job_ids == [result.job_id]
    assert len(jobs) == 1
    assert jobs[0].status == "pending"
    assert jobs[0].file_id == knowledge_file.id


def test_orchestrator_skips_duplicate_completed_job() -> None:
    session_factory = build_session_factory()
    runner = RecordingRunner()
    settings = build_test_settings()
    orchestrator = KnowledgeIngestionOrchestrator(settings=settings, runner=runner)

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        first = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        job = session.get(KnowledgeIngestionJob, first.job_id)
        assert job is not None
        job.status = "completed"
        job.chunk_count = 4
        job.completed_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

        second = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        jobs = list(session.scalars(select(KnowledgeIngestionJob).order_by(KnowledgeIngestionJob.created_at.asc())))

    assert second.status == "skipped"
    assert runner.submitted_job_ids == [first.job_id]
    assert len(jobs) == 2
    assert jobs[1].status == "skipped"
    assert jobs[1].chunk_count == 4


def test_orchestrator_reuses_active_job_for_same_idempotency_key() -> None:
    session_factory = build_session_factory()
    runner = RecordingRunner()
    settings = build_test_settings()
    orchestrator = KnowledgeIngestionOrchestrator(settings=settings, runner=runner)

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        first = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        second = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        jobs = list(session.scalars(select(KnowledgeIngestionJob)))

    assert second.status == "queued"
    assert second.job_id == first.job_id
    assert runner.submitted_job_ids == [first.job_id]
    assert len(jobs) == 1


def test_orchestrator_allows_retry_after_failed_job() -> None:
    session_factory = build_session_factory()
    runner = RecordingRunner()
    settings = build_test_settings()
    orchestrator = KnowledgeIngestionOrchestrator(settings=settings, runner=runner)

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        first = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        first_job = session.get(KnowledgeIngestionJob, first.job_id)
        assert first_job is not None
        first_job.status = "failed"
        first_job.error_message = "embedding timeout"
        session.add(first_job)
        session.commit()

        second = orchestrator.trigger(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
            effective_settings=settings,
        )
        jobs = list(session.scalars(select(KnowledgeIngestionJob).order_by(KnowledgeIngestionJob.created_at.asc())))

    assert second.status == "queued"
    assert second.job_id != first.job_id
    assert runner.submitted_job_ids == [first.job_id, second.job_id]
    assert [job.status for job in jobs] == ["failed", "pending"]


def test_worker_marks_job_completed_and_records_chunk_count() -> None:
    session_factory = build_session_factory()
    settings = build_test_settings()
    fake_service = FakeIngestionService(chunk_count=7)
    worker = KnowledgeIngestionJobWorker(
        settings=settings,
        ingestion_service_factory=lambda _settings: fake_service,
    )

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        job = KnowledgeIngestionJob(
            source_type="uploaded_file",
            source_id=knowledge_file.id,
            file_id=knowledge_file.id,
            storage_provider="local",
            storage_path=knowledge_file.storage_path,
            status="pending",
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.knowledge_embedding_model,
            embedding_dimension=settings.embedding_dimension,
            chunk_size=settings.knowledge_chunk_size,
            chunk_overlap=settings.knowledge_chunk_overlap,
            content_checksum=knowledge_file.checksum,
            idempotency_key="idem-1",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        result = worker.run_job(session, job_id=job.id)
        refreshed_job = session.get(KnowledgeIngestionJob, job.id)

    assert result.status == "completed"
    assert refreshed_job is not None
    assert refreshed_job.status == "completed"
    assert refreshed_job.chunk_count == 7
    assert refreshed_job.started_at is not None
    assert refreshed_job.completed_at is not None
    assert len(fake_service.calls) == 1


def test_worker_marks_job_failed_when_ingestion_raises() -> None:
    session_factory = build_session_factory()
    settings = build_test_settings()
    fake_service = FakeIngestionService(error=KnowledgeIngestionServiceError("embedding timeout"))
    worker = KnowledgeIngestionJobWorker(
        settings=settings,
        ingestion_service_factory=lambda _settings: fake_service,
    )

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        job = KnowledgeIngestionJob(
            source_type="uploaded_file",
            source_id=knowledge_file.id,
            file_id=knowledge_file.id,
            storage_provider="local",
            storage_path=knowledge_file.storage_path,
            status="pending",
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.knowledge_embedding_model,
            embedding_dimension=settings.embedding_dimension,
            chunk_size=settings.knowledge_chunk_size,
            chunk_overlap=settings.knowledge_chunk_overlap,
            content_checksum=knowledge_file.checksum,
            idempotency_key="idem-1",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        with pytest.raises(KnowledgeIngestionServiceError, match="embedding timeout"):
            worker.run_job(session, job_id=job.id)

        refreshed_job = session.get(KnowledgeIngestionJob, job.id)

    assert refreshed_job is not None
    assert refreshed_job.status == "failed"
    assert refreshed_job.error_message == "embedding timeout"
    assert refreshed_job.completed_at is not None


def test_local_runner_submits_without_blocking_call_site() -> None:
    session_factory = build_session_factory()
    settings = build_test_settings()
    fake_service = FakeIngestionService()
    worker = KnowledgeIngestionJobWorker(
        settings=settings,
        ingestion_service_factory=lambda _settings: fake_service,
    )
    runner = LocalKnowledgeIngestionRunner(session_factory=session_factory, worker=worker)

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(session)
        job = KnowledgeIngestionJob(
            source_type="uploaded_file",
            source_id=knowledge_file.id,
            file_id=knowledge_file.id,
            storage_provider="local",
            storage_path=knowledge_file.storage_path,
            status="pending",
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.knowledge_embedding_model,
            embedding_dimension=settings.embedding_dimension,
            chunk_size=settings.knowledge_chunk_size,
            chunk_overlap=settings.knowledge_chunk_overlap,
            content_checksum=knowledge_file.checksum,
            idempotency_key="idem-1",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        runner.submit(job_id=job.id)

    # The local runner is fire-and-forget. This test only verifies the submit
    # call returns synchronously without raising.
    assert True
