from __future__ import annotations

from typing import Any
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.dependencies.chat_dependencies import get_chat_service
from app.api.dependencies.common_dependencies import get_app_settings
from app.api.dependencies.tavus_dependencies import get_tavus_service
from app.api.tavus.schemas import (
    TavusConversationCreateRequest,
    TavusConversationEndRequest,
    TavusConversationResponse,
    TavusToolRequest,
    TavusToolResponse,
)
from app.config import Settings
from app.services.chat import ChatService
from app.services.tavus import TavusService

router = APIRouter(prefix="/api/tavus", tags=["Tavus"])


@router.post("/conversations", response_model=TavusConversationResponse)
async def create_tavus_conversation(
    request: TavusConversationCreateRequest,
    tavus_service: TavusService = Depends(get_tavus_service),
) -> TavusConversationResponse:
    session = await tavus_service.create_conversation(
        visitor_name=request.visitor_name,
        backend_conversation_id=request.conversation_id,
    )
    return TavusConversationResponse(
        conversation_id=session.conversation_id,
        conversation_url=session.conversation_url,
    )


@router.post("/tools/ask-tumelo", response_model=TavusToolResponse)
async def ask_tumelo_tool(
    request: TavusToolRequest,
    chat_service: ChatService = Depends(get_chat_service),
    settings: Settings = Depends(get_app_settings),
    tavus_tool_secret: str | None = Header(default=None, alias="x-tavus-tool-secret"),
) -> TavusToolResponse:
    _validate_tavus_tool_secret(
        provided_secret=tavus_tool_secret,
        expected_secret=settings.tavus_tool_secret,
    )
    answer = await chat_service.generate_answer(
        user_message=request.message,
        conversation_id=request.tavus_conversation_id,
        channel="tavus_video",
        metadata={
            "visitor_name": request.visitor_name or "Website visitor",
            "source": "tavus_tool_call",
            "tavus_conversation_id": request.tavus_conversation_id,
        },
    )
    return TavusToolResponse(response=answer.message)


@router.post("/conversations/end")
async def end_tavus_conversation(
    request: TavusConversationEndRequest,
    tavus_service: TavusService = Depends(get_tavus_service),
) -> dict[str, Any]:
    return await tavus_service.end_conversation(conversation_id=request.conversation_id)


def _validate_tavus_tool_secret(
    *,
    provided_secret: str | None,
    expected_secret: str | None,
) -> None:
    if not expected_secret or not provided_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Tavus tool secret.",
        )
    if not secrets.compare_digest(provided_secret, expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Tavus tool secret.",
        )
