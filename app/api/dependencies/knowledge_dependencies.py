from collections.abc import Callable

from fastapi import Depends

from app.api.dependencies.common_dependencies import get_app_settings
from app.config import Settings
from app.knowledge.ingestion import KnowledgeIngestionService
from app.services.retrieval import RetrievalService


def build_knowledge_ingestion_service(settings: Settings) -> KnowledgeIngestionService:
    return KnowledgeIngestionService(
        retrieval_service=RetrievalService(settings=settings),
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
