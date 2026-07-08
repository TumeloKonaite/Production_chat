from collections.abc import Callable

from fastapi import Depends

from app.api.dependencies.common_dependencies import get_app_settings
from app.config import Settings
from app.infrastructure.storage import create_knowledge_file_storage
from app.knowledge.ingestion import (
    KnowledgeIngestionJobWorker,
    KnowledgeIngestionOrchestrator,
    KnowledgeIngestionService,
    LocalKnowledgeIngestionRunner,
    ModalKnowledgeIngestionRunner,
    UploadedKnowledgeFileLoader,
)
from app.repositories.db.session import get_session_factory
from app.services.knowledge_files import KnowledgeFileUploadService
from app.services.retrieval import RetrievalService


def build_knowledge_ingestion_service(settings: Settings) -> KnowledgeIngestionService:
    return KnowledgeIngestionService(
        retrieval_service=RetrievalService(settings=settings),
        uploaded_file_loader=UploadedKnowledgeFileLoader(
            storage=create_knowledge_file_storage(settings),
            storage_provider=settings.storage_provider,
        ),
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )


def get_knowledge_ingestion_service_factory(
    settings: Settings = Depends(get_app_settings),
) -> Callable[[Settings | None], KnowledgeIngestionService]:
    def build_service(effective_settings: Settings | None = None) -> KnowledgeIngestionService:
        resolved_settings = effective_settings or settings
        return build_knowledge_ingestion_service(resolved_settings)

    return build_service


def build_knowledge_ingestion_job_worker(settings: Settings) -> KnowledgeIngestionJobWorker:
    return KnowledgeIngestionJobWorker(
        settings=settings,
        ingestion_service_factory=build_knowledge_ingestion_service,
    )


def get_knowledge_ingestion_orchestrator(
    settings: Settings = Depends(get_app_settings),
) -> KnowledgeIngestionOrchestrator:
    worker = build_knowledge_ingestion_job_worker(settings)
    if settings.ingestion_backend == "modal":
        runner = ModalKnowledgeIngestionRunner(
            app_name=settings.modal_ingestion_app_name,
            function_name=settings.modal_ingestion_function_name,
        )
    else:
        runner = LocalKnowledgeIngestionRunner(
            session_factory=get_session_factory(),
            worker=worker,
        )
    return KnowledgeIngestionOrchestrator(settings=settings, runner=runner)


def get_knowledge_file_upload_service(
    settings: Settings = Depends(get_app_settings),
) -> KnowledgeFileUploadService:
    return KnowledgeFileUploadService(
        settings=settings,
        storage=create_knowledge_file_storage(settings),
    )
