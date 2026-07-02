from collections.abc import Callable

from fastapi import Depends

from app.api.dependencies.common_dependencies import get_app_settings
from app.config import Settings
from app.knowledge.ingestion import KnowledgeIngestionService
from app.services.retrieval import RetrievalService


def get_knowledge_ingestion_service_factory(
    settings: Settings = Depends(get_app_settings),
) -> Callable[[], KnowledgeIngestionService]:
    def build_service() -> KnowledgeIngestionService:
        return KnowledgeIngestionService(
            retrieval_service=RetrievalService(settings=settings),
            chunk_size=settings.knowledge_chunk_size,
            chunk_overlap=settings.knowledge_chunk_overlap,
        )

    return build_service
