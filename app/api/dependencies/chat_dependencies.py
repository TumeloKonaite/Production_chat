from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.repositories import ConversationRepository
from app.services.chat import ChatService
from app.services.llm import LLMService


def get_llm_service(settings: Settings = Depends(get_app_settings)) -> LLMService:
    return LLMService(settings=settings)


def get_chat_repository(
    session: Session = Depends(get_db_session),
) -> ConversationRepository:
    return ConversationRepository(session=session)


def get_chat_service(
    settings: Settings = Depends(get_app_settings),
    llm_service: LLMService = Depends(get_llm_service),
    repository: ConversationRepository = Depends(get_chat_repository),
) -> ChatService:
    return ChatService(
        llm_service=llm_service,
        repository=repository,
        history_limit=settings.conversation_history_limit,
    )
