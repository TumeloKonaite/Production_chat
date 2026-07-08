import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.evals import router as evals_router
from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.api.tavus import router as tavus_router
from app.config import Settings, get_settings
from app.infrastructure.llm import UnknownModelError
from app.infrastructure.prompts import UnknownPromptVersionError
from app.knowledge.ingestion import (
    KnowledgeIngestionConflictError,
    KnowledgeIngestionGoneError,
    KnowledgeIngestionNotFoundError,
    KnowledgeIngestionServiceError,
    KnowledgeIngestionValidationError,
)
from app.services.chat import (
    ChatPersistenceError,
    ChatServiceError,
    ConversationNotFoundError,
    InvalidChatMessageError,
    InvalidConversationIdError,
)
from app.services.cache import DuplicateRequestInProgressError
from app.services.feedback import (
    InvalidFeedbackTargetError,
    MessageFeedbackPersistenceError,
    MessageFeedbackTargetNotFoundError,
)
from app.services.knowledge_files import KnowledgeFileUploadError, KnowledgeFileValidationError
from app.services.llm import LLMConfigurationError, LLMServiceError
from app.services.rate_limiting import (
    RateLimitExceededError,
    RateLimitingBackendUnavailableError,
)
from app.services.retrieval import EmbeddingConfigurationError, VectorIndexConfigurationError
from app.services.tavus import TavusConfigurationError, TavusServiceError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI, settings: Settings):
    logger.info("App environment: %s", settings.app_env)
    logger.info("Vector store provider: %s", settings.vector_store_provider)
    logger.info("Langfuse enabled: %s", settings.enable_langfuse_observability)
    logger.info("MLflow tracking enabled: %s", settings.enable_mlflow_tracking)
    logger.info("Redis configured: %s", settings.redis_configured)
    logger.info("Supabase configured: %s", settings.supabase_configured)
    logger.info("OpenAI base URL configured: %s", settings.openai_base_url_configured)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="Production Chatbot",
        lifespan=lambda app: _lifespan(app, resolved_settings),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(evals_router)
    app.include_router(knowledge_router)
    app.include_router(tavus_router)

    @app.exception_handler(InvalidChatMessageError)
    async def handle_invalid_message(
        _: Request,
        exc: InvalidChatMessageError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(InvalidConversationIdError)
    async def handle_invalid_conversation_id(
        _: Request,
        exc: InvalidConversationIdError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(UnknownPromptVersionError)
    async def handle_unknown_prompt_version(
        _: Request,
        exc: UnknownPromptVersionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(UnknownModelError)
    async def handle_unknown_model(
        _: Request,
        exc: UnknownModelError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ConversationNotFoundError)
    async def handle_missing_conversation(
        _: Request,
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(MessageFeedbackTargetNotFoundError)
    async def handle_missing_feedback_message(
        _: Request,
        exc: MessageFeedbackTargetNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(InvalidFeedbackTargetError)
    async def handle_invalid_feedback_target(
        _: Request,
        exc: InvalidFeedbackTargetError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(LLMConfigurationError)
    async def handle_configuration_error(
        _: Request,
        __: LLMConfigurationError,
    ) -> JSONResponse:
        # Configuration failures are kept generic so deployment details are not exposed.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Chat service is not configured correctly."},
        )

    @app.exception_handler(EmbeddingConfigurationError)
    async def handle_embedding_configuration_error(
        _: Request,
        exc: EmbeddingConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    @app.exception_handler(VectorIndexConfigurationError)
    async def handle_vector_index_configuration_error(
        _: Request,
        exc: VectorIndexConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    @app.exception_handler(TavusConfigurationError)
    async def handle_tavus_configuration_error(
        _: Request,
        __: TavusConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Tavus integration is not configured correctly."},
        )

    @app.exception_handler(KnowledgeIngestionServiceError)
    async def handle_knowledge_ingestion_error(
        _: Request,
        __: KnowledgeIngestionServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unable to ingest knowledge. Please try again."},
        )

    @app.exception_handler(KnowledgeIngestionValidationError)
    async def handle_knowledge_ingestion_validation_error(
        _: Request,
        exc: KnowledgeIngestionValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(KnowledgeIngestionNotFoundError)
    async def handle_knowledge_ingestion_not_found(
        _: Request,
        exc: KnowledgeIngestionNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(KnowledgeIngestionConflictError)
    async def handle_knowledge_ingestion_conflict(
        _: Request,
        exc: KnowledgeIngestionConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(KnowledgeIngestionGoneError)
    async def handle_knowledge_ingestion_gone(
        _: Request,
        exc: KnowledgeIngestionGoneError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(KnowledgeFileValidationError)
    async def handle_knowledge_file_validation_error(
        _: Request,
        exc: KnowledgeFileValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(KnowledgeFileUploadError)
    async def handle_knowledge_file_upload_error(
        _: Request,
        __: KnowledgeFileUploadError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unable to upload knowledge file. Please try again."},
        )

    @app.exception_handler(LLMServiceError)
    async def handle_llm_error(_: Request, __: LLMServiceError) -> JSONResponse:
        # Upstream provider failures are normalized to one safe client-facing message.
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "Unable to generate assistant response. Please try again."},
        )

    @app.exception_handler(RateLimitExceededError)
    async def handle_rate_limit_exceeded(
        _: Request,
        exc: RateLimitExceededError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": exc.detail,
                "retry_after_seconds": exc.retry_after_seconds,
            },
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(RateLimitingBackendUnavailableError)
    async def handle_rate_limiting_backend_unavailable(
        _: Request,
        exc: RateLimitingBackendUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": exc.detail},
        )

    @app.exception_handler(DuplicateRequestInProgressError)
    async def handle_duplicate_request_in_progress(
        _: Request,
        exc: DuplicateRequestInProgressError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ChatPersistenceError)
    async def handle_persistence_error(
        _: Request,
        __: ChatPersistenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unable to save chat conversation. Please try again."},
        )

    @app.exception_handler(MessageFeedbackPersistenceError)
    async def handle_feedback_persistence_error(
        _: Request,
        __: MessageFeedbackPersistenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unable to save message feedback. Please try again."},
        )

    @app.exception_handler(ChatServiceError)
    async def handle_chat_service_error(
        _: Request,
        __: ChatServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unable to generate assistant response. Please try again."},
        )

    @app.exception_handler(TavusServiceError)
    async def handle_tavus_service_error(
        _: Request,
        __: TavusServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "Unable to complete Tavus request. Please try again."},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unexpected backend error. Please try again."},
        )

    return app


app = create_app()
