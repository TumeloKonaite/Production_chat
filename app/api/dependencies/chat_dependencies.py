from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.repositories import ConversationRepository, KnowledgeRepository
from app.services.retrieval import RetrievalService
from app.services.chat import ChatService
from app.services.llm import LLMService


def get_llm_service(settings: Settings = Depends(get_app_settings)) -> LLMService:
    return LLMService(settings=settings)


def get_chat_repository(
    session: Session = Depends(get_db_session),
) -> ConversationRepository:
    return ConversationRepository(session=session)


def get_knowledge_repository(
    session: Session = Depends(get_db_session),
) -> KnowledgeRepository:
    return KnowledgeRepository(session=session)


def get_retrieval_service(
    settings: Settings = Depends(get_app_settings),
) -> RetrievalService:
    return RetrievalService(settings=settings)


def get_chat_service(
    settings: Settings = Depends(get_app_settings),
    llm_service: LLMService = Depends(get_llm_service),
    repository: ConversationRepository = Depends(get_chat_repository),
    knowledge_repository: KnowledgeRepository = Depends(get_knowledge_repository),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> ChatService:
    return ChatService(
        llm_service=llm_service,
        repository=repository,
        knowledge_repository=knowledge_repository,
        retrieval_service=retrieval_service,
        history_limit=settings.conversation_history_limit,
        retrieval_top_k=settings.retrieval_top_k,
    )
