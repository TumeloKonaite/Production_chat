from functools import lru_cache
import logging
from pathlib import Path

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.infrastructure.cache import CacheClient, NullCacheClient, build_cache_client
from app.infrastructure.observability import ObservabilityTracer, get_tracer
from app.infrastructure.prompts import PromptLoader
from app.repositories import ConversationRepository, KnowledgeRepository, MessageFeedbackRepository
from app.services.cache import NoOpResponseCache, RedisResponseCache, RequestLock, ResponseCache
from app.services.chat import ChatService
from app.services.feedback import MessageFeedbackService
from app.services.llm import LLMService
from app.services.rate_limiting.service import RateLimitingService
from app.services.retrieval import RetrievalService
from app.services.tracing import TraceService

logger = logging.getLogger(__name__)


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


def get_feedback_repository(
    session: Session = Depends(get_db_session),
) -> MessageFeedbackRepository:
    return MessageFeedbackRepository(session=session)


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


def get_observability_tracer(
    settings: Settings = Depends(get_app_settings),
) -> ObservabilityTracer:
    return get_tracer(settings)


def get_retrieval_service(
    settings: Settings = Depends(get_app_settings),
    knowledge_repository: KnowledgeRepository = Depends(get_knowledge_repository),
) -> RetrievalService:
    return RetrievalService(
        settings=settings,
        knowledge_repository=knowledge_repository,
    )


@lru_cache
def _build_cache_layer(settings: Settings) -> CacheClient:
    try:
        return build_cache_client(settings)
    except Exception:
        logger.warning(
            "Redis cache layer initialization failed. Continuing with disabled cache layer.",
            exc_info=True,
        )
        return NullCacheClient()


def get_cache_client(
    settings: Settings = Depends(get_app_settings),
) -> CacheClient:
    return _build_cache_layer(settings)


@lru_cache
def _build_response_cache(settings: Settings) -> ResponseCache:
    if not settings.enable_redis or not settings.exact_cache_enabled:
        return NoOpResponseCache()

    try:
        return RedisResponseCache(
            settings=settings,
            redis_client=_build_cache_layer(settings),
        )
    except Exception:
        logger.warning(
            "Response cache initialization failed. Continuing with disabled cache.",
            exc_info=True,
        )
        return NoOpResponseCache()


def get_response_cache(
    settings: Settings = Depends(get_app_settings),
) -> ResponseCache:
    return _build_response_cache(settings)


@lru_cache
def _build_request_lock(settings: Settings) -> RequestLock:
    return RequestLock(
        settings=settings,
        cache_client=_build_cache_layer(settings),
    )


def get_request_lock(
    settings: Settings = Depends(get_app_settings),
) -> RequestLock:
    return _build_request_lock(settings)


@lru_cache
def _build_rate_limiting_service(settings: Settings) -> RateLimitingService:
    return RateLimitingService(
        settings=settings,
        cache_client=_build_cache_layer(settings),
    )


def get_rate_limiting_service(
    settings: Settings = Depends(get_app_settings),
) -> RateLimitingService:
    return _build_rate_limiting_service(settings)


def get_chat_service(
    settings: Settings = Depends(get_app_settings),
    llm_service: LLMService = Depends(get_llm_service),
    prompt_loader: PromptLoader = Depends(get_prompt_loader),
    repository: ConversationRepository = Depends(get_chat_repository),
    knowledge_repository: KnowledgeRepository = Depends(get_knowledge_repository),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    response_cache: ResponseCache = Depends(get_response_cache),
    request_lock: RequestLock = Depends(get_request_lock),
    rate_limiting_service: RateLimitingService = Depends(get_rate_limiting_service),
    trace_service: TraceService = Depends(get_trace_service),
    observability_tracer: ObservabilityTracer = Depends(get_observability_tracer),
) -> ChatService:
    return ChatService(
        llm_service=llm_service,
        prompt_loader=prompt_loader,
        repository=repository,
        knowledge_repository=knowledge_repository,
        retrieval_service=retrieval_service,
        response_cache=response_cache,
        request_lock=request_lock,
        rate_limiting_service=rate_limiting_service,
        trace_service=trace_service,
        observability_tracer=observability_tracer,
        history_limit=settings.conversation_history_limit,
        retrieval_top_k=settings.retrieval_top_k,
        settings=settings,
    )


def get_feedback_service(
    repository: MessageFeedbackRepository = Depends(get_feedback_repository),
) -> MessageFeedbackService:
    return MessageFeedbackService(repository=repository)
