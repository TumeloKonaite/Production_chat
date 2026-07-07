from fastapi import APIRouter, Depends

from app.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackCreate,
    FeedbackResponse,
    TokenUsageResponse,
)
from app.api.dependencies.chat_dependencies import get_chat_service, get_feedback_service
from app.services.chat import ChatService
from app.services.feedback import MessageFeedbackService

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
        model_config_id=request.model_config_id,
    )
    return ChatResponse(
        conversation_id=response.conversation_id,
        message_id=response.message_id,
        message=response.message,
        model=response.model,
        model_provider=response.model_provider,
        model_name=response.model_name,
        model_config_id=response.model_config_id,
        prompt_version=response.prompt_version,
        retrieval_config=response.retrieval_config,
        latency_ms=response.latency_ms,
        token_usage=TokenUsageResponse(
            input_tokens=response.token_usage.input_tokens,
            output_tokens=response.token_usage.output_tokens,
            total_tokens=response.token_usage.total_tokens,
        ),
        estimated_cost_usd=response.estimated_cost_usd,
    )


@router.post("/api/chat/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_message_feedback(
    message_id: str,
    request: FeedbackCreate,
    feedback_service: MessageFeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    feedback = feedback_service.submit_feedback(
        message_id=message_id,
        rating=request.rating,
        comment=request.comment,
    )
    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )
