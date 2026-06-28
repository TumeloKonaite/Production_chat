from fastapi import APIRouter, Depends

from app.api.dependencies.chat_dependencies import get_chat_service
from app.api.schema import ChatRequest, ChatResponse, TokenUsageResponse
from app.services.chat import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    response = await chat_service.generate_reply(
        message=request.message,
        conversation_id=request.conversation_id,
        prompt_version=request.prompt_version,
    )
    return ChatResponse(
        conversation_id=response.conversation_id,
        message=response.message,
        model=response.model,
        prompt_version=response.prompt_version,
        latency_ms=response.latency_ms,
        token_usage=TokenUsageResponse(
            input_tokens=response.token_usage.input_tokens,
            output_tokens=response.token_usage.output_tokens,
            total_tokens=response.token_usage.total_tokens,
        ),
    )
