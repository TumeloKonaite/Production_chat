from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.config import Settings
from app.infrastructure.prompts import PromptLoader
from app.repositories import (
    ConversationRepository,
    ConversationRepositoryError,
    KnowledgeRepository,
    KnowledgeRepositoryError,
)
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    should_use_direct_fallback,
)
from app.services.chat.errors import (
    ChatPersistenceError,
    ChatServiceError,
    ConversationNotFoundError,
    InvalidChatMessageError,
    InvalidConversationIdError,
)
from app.services.llm.service import LLMChatMessage, LLMGeneratedResponse, TokenUsage
from app.services.retrieval import RetrievedChunk, RetrievalService


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
        prompt_loader: PromptLoader,
        repository: ConversationRepository,
        knowledge_repository: KnowledgeRepository,
        retrieval_service: RetrievalService,
        history_limit: int,
        retrieval_top_k: int,
        settings: Settings,
    ) -> None:
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.repository = repository
        self.knowledge_repository = knowledge_repository
        self.retrieval_service = retrieval_service
        self.history_limit = history_limit
        self.retrieval_top_k = retrieval_top_k
        self.settings = settings

    async def generate_reply(
        self,
        message: str,
        conversation_id: str | None = None,
        prompt_version: str | None = None,
    ) -> ChatReply:
        normalized_message = message.strip()
        if not normalized_message:
            raise InvalidChatMessageError("Chat message cannot be empty.")

        selected_prompt_version = prompt_version or self.settings.default_prompt_version
        base_prompt = self.prompt_loader.load(selected_prompt_version)

        conversation = self._get_or_create_conversation(
            conversation_id,
            prompt_version=selected_prompt_version,
        )
        conversation.prompt_version = selected_prompt_version

        try:
            # Persist the user turn before calling the LLM so failed generations still leave a trace.
            user_message = self.repository.add_message(
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

        retrieved_chunks = self._retrieve_chunks(normalized_message)
        use_direct_fallback = should_use_direct_fallback(normalized_message, retrieved_chunks)
        self._log_retrieval(
            conversation_id=conversation.id,
            message_id=user_message.id,
            query=normalized_message,
            retrieved_chunks=retrieved_chunks,
            used_fallback=use_direct_fallback,
        )

        if use_direct_fallback:
            llm_response = self._build_direct_response(
                normalized_message,
                prompt_version=selected_prompt_version,
            )
        else:
            llm_messages = [
                LLMChatMessage(role=stored_message.role, content=stored_message.content)
                for stored_message in recent_messages
            ]
            system_prompt = build_chat_system_prompt(
                base_prompt=base_prompt,
                message=normalized_message,
                retrieved_chunks=retrieved_chunks,
            )
            llm_response = await self.llm_service.generate_response(
                llm_messages,
                system_prompt=system_prompt,
                prompt_version=selected_prompt_version,
            )

        # Only persist the assistant turn after the final response succeeds.
        self._store_assistant_message(conversation=conversation, llm_response=llm_response)

        return ChatReply(
            conversation_id=conversation.id,
            message=llm_response.message,
            model=llm_response.model,
            prompt_version=llm_response.prompt_version,
            latency_ms=llm_response.latency_ms,
            token_usage=llm_response.token_usage,
        )

    def _get_or_create_conversation(
        self,
        conversation_id: str | None,
        *,
        prompt_version: str,
    ):
        if conversation_id is None:
            return self._create_conversation(prompt_version=prompt_version)

        self._validate_conversation_id(conversation_id)
        try:
            conversation = self.repository.get_conversation(conversation_id)
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")

        return conversation

    def _create_conversation(self, *, prompt_version: str):
        try:
            return self.repository.create_conversation(
                model=self.llm_service.model,
                prompt_version=prompt_version,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _retrieve_chunks(self, message: str) -> list[RetrievedChunk]:
        try:
            return self.retrieval_service.retrieve(message, top_k=self.retrieval_top_k)
        except KnowledgeRepositoryError as exc:
            raise ChatServiceError() from exc

    def _log_retrieval(
        self,
        *,
        conversation_id: str,
        message_id: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        used_fallback: bool,
    ) -> None:
        try:
            self.knowledge_repository.log_retrieval(
                conversation_id=conversation_id,
                message_id=message_id,
                query=query,
                top_k=self.retrieval_top_k,
                retrieved_chunk_ids=[item.id for item in retrieved_chunks],
                retrieved_sources=[item.source for item in retrieved_chunks],
                similarity_scores=[item.similarity for item in retrieved_chunks],
                used_fallback=used_fallback,
            )
        except KnowledgeRepositoryError as exc:
            raise ChatServiceError() from exc

    def _build_direct_response(
        self,
        message: str,
        *,
        prompt_version: str,
    ) -> LLMGeneratedResponse:
        return LLMGeneratedResponse(
            message=build_direct_fallback_text(message),
            model=self.llm_service.model,
            prompt_version=prompt_version,
            latency_ms=None,
            token_usage=TokenUsage(),
        )

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
