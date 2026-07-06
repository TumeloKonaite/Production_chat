from pathlib import Path

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.infrastructure.prompts import PromptLoader
from app.repositories import ConversationRepository, KnowledgeRepository
from app.services.retrieval import RetrievalService
from app.services.chat import ChatService
from app.services.llm import LLMService
from app.services.tracing import TraceService


def get_llm_service(settings: Settings = Depends(get_app_settings)) -> LLMService:
    return LLMService(settings=settings)


def get_prompt_loader() -> PromptLoader:
    prompts_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "infrastructure"
        / "prompts"
        / "templates"
    )
    return PromptLoader(prompts_dir=prompts_dir)


def get_chat_repository(
    session: Session = Depends(get_db_session),
) -> ConversationRepository:
    return ConversationRepository(session=session)


def get_knowledge_repository(
    session: Session = Depends(get_db_session),
) -> KnowledgeRepository:
    return KnowledgeRepository(session=session)


def get_trace_service(
    session: Session = Depends(get_db_session),
) -> TraceService:
    trace_session_factory = sessionmaker(
        bind=session.get_bind(),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )
    return TraceService(session_factory=trace_session_factory)


def get_retrieval_service(
    settings: Settings = Depends(get_app_settings),
    knowledge_repository: KnowledgeRepository = Depends(get_knowledge_repository),
) -> RetrievalService:
    return RetrievalService(
        settings=settings,
        knowledge_repository=knowledge_repository,
    )


def get_chat_service(
    settings: Settings = Depends(get_app_settings),
    llm_service: LLMService = Depends(get_llm_service),
    prompt_loader: PromptLoader = Depends(get_prompt_loader),
    repository: ConversationRepository = Depends(get_chat_repository),
    knowledge_repository: KnowledgeRepository = Depends(get_knowledge_repository),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    trace_service: TraceService = Depends(get_trace_service),
) -> ChatService:
    return ChatService(
        llm_service=llm_service,
        prompt_loader=prompt_loader,
        repository=repository,
        knowledge_repository=knowledge_repository,
        retrieval_service=retrieval_service,
        trace_service=trace_service,
        history_limit=settings.conversation_history_limit,
        retrieval_top_k=settings.retrieval_top_k,
        settings=settings,
    )
