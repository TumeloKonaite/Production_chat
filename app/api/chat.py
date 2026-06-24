from fastapi import APIRouter, Depends

from app.config import get_settings
from app.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService

router = APIRouter(tags=["chat"])


def get_chat_service() -> ChatService:
    settings = get_settings()
    llm_service = LLMService(settings=settings)
    return ChatService(llm_service=llm_service)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    response = await chat_service.generate_reply(request.message)
    return ChatResponse(response=response)
