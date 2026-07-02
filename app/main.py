from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.tavus import router as tavus_router
from app.infrastructure.llm import UnknownModelError
from app.infrastructure.prompts import UnknownPromptVersionError
from app.services.chat import (
    ChatPersistenceError,
    ChatServiceError,
    ConversationNotFoundError,
    InvalidChatMessageError,
    InvalidConversationIdError,
)
from app.services.llm import LLMConfigurationError, LLMServiceError
from app.services.tavus import TavusConfigurationError, TavusServiceError


def create_app() -> FastAPI:
    app = FastAPI(title="Production Chatbot")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)
    app.include_router(tavus_router)

    @app.get("/health", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

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

    @app.exception_handler(TavusConfigurationError)
    async def handle_tavus_configuration_error(
        _: Request,
        __: TavusConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Tavus integration is not configured correctly."},
        )

    @app.exception_handler(LLMServiceError)
    async def handle_llm_error(_: Request, __: LLMServiceError) -> JSONResponse:
        # Upstream provider failures are normalized to one safe client-facing message.
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "Unable to generate assistant response. Please try again."},
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
