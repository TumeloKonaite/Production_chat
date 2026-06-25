from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.repositories import ConversationRepository, ConversationRepositoryError
from app.services.chat.errors import (
    ChatPersistenceError,
    ConversationNotFoundError,
    InvalidChatMessageError,
    InvalidConversationIdError,
)
from app.services.llm.service import LLMChatMessage, LLMGeneratedResponse, TokenUsage


@dataclass(frozen=True, slots=True)
class ChatReply:
    conversation_id: str
    message: str
    model: str
    prompt_version: str
    latency_ms: int | None
    token_usage: TokenUsage


class ChatService:
    def __init__(
        self,
        llm_service,
        repository: ConversationRepository,
        history_limit: int,
    ) -> None:
        self.llm_service = llm_service
        self.repository = repository
        self.history_limit = history_limit

    async def generate_reply(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> ChatReply:
        normalized_message = message.strip()
        if not normalized_message:
            raise InvalidChatMessageError("Chat message cannot be empty.")

        conversation = self._get_or_create_conversation(conversation_id)

        try:
            # Persist the user turn before calling the LLM so failed generations still leave a trace.
            self.repository.add_message(
                conversation=conversation,
                role="user",
                content=normalized_message,
            )
            recent_messages = self.repository.list_recent_messages(
                conversation.id,
                limit=self.history_limit,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        llm_messages = [
            LLMChatMessage(role=stored_message.role, content=stored_message.content)
            for stored_message in recent_messages
        ]
        llm_response = await self.llm_service.generate_response(llm_messages)
        # Only persist the assistant turn after the upstream call succeeds.
        self._store_assistant_message(conversation=conversation, llm_response=llm_response)

        return ChatReply(
            conversation_id=conversation.id,
            message=llm_response.message,
            model=llm_response.model,
            prompt_version=llm_response.prompt_version,
            latency_ms=llm_response.latency_ms,
            token_usage=llm_response.token_usage,
        )

    def _get_or_create_conversation(self, conversation_id: str | None):
        if conversation_id is None:
            return self._create_conversation()

        self._validate_conversation_id(conversation_id)
        try:
            conversation = self.repository.get_conversation(conversation_id)
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")

        return conversation

    def _create_conversation(self):
        try:
            return self.repository.create_conversation(
                model=self.llm_service.model,
                prompt_version=self.llm_service.prompt_version,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _store_assistant_message(
        self,
        *,
        conversation,
        llm_response: LLMGeneratedResponse,
    ) -> None:
        try:
            self.repository.add_message(
                conversation=conversation,
                role="assistant",
                content=llm_response.message,
                model=llm_response.model,
                prompt_version=llm_response.prompt_version,
                latency_ms=llm_response.latency_ms,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _validate_conversation_id(self, conversation_id: str) -> None:
        try:
            uuid.UUID(conversation_id)
        except ValueError as exc:
            raise InvalidConversationIdError("conversation_id must be a valid UUID.") from exc
