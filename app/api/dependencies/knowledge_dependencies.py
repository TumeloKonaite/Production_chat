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
        )

    return build_service
