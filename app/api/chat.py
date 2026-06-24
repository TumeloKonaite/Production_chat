from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    response = await chat_service.generate_reply(request.message)
    return ChatResponse(response=response)
