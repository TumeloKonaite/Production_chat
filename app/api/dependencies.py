from app.config import get_settings
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService


def get_chat_service() -> ChatService:
    # Build the request-scoped chat service that FastAPI injects into the route.
    settings = get_settings()
    llm_service = LLMService(settings=settings)
    return ChatService(llm_service=llm_service)
