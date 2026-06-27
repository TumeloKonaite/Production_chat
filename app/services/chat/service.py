from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.repositories import (
    ConversationRepository,
    ConversationRepositoryError,
    KnowledgeRepository,
    KnowledgeRepositoryError,
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

PERSONAL_QUERY_MARKERS = {
    "tumelo",
    "you",
    "your",
    "yours",
    "experience",
    "background",
    "education",
    "contact",
    "email",
    "linkedin",
    "github",
    "portfolio",
    "project",
    "projects",
    "skill",
    "skills",
    "worked",
    "career",
    "resume",
    "cv",
    "employer",
    "employment",
    "degree",
    "certification",
    "location",
}
GENERAL_TECH_MARKERS = {
    "api",
    "backend",
    "chatbot",
    "database",
    "docker",
    "embedding",
    "fastapi",
    "llm",
    "postgres",
    "python",
    "rag",
    "retrieval",
    "sql",
    "sqlalchemy",
    "vector",
}


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
        knowledge_repository: KnowledgeRepository,
        retrieval_service: RetrievalService,
        history_limit: int,
        retrieval_top_k: int,
    ) -> None:
        self.llm_service = llm_service
        self.repository = repository
        self.knowledge_repository = knowledge_repository
        self.retrieval_service = retrieval_service
        self.history_limit = history_limit
        self.retrieval_top_k = retrieval_top_k

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
        use_direct_fallback = not retrieved_chunks and self._should_use_direct_fallback(normalized_message)
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
                personal_query=self._is_personal_query(normalized_message),
            )
        else:
            llm_messages = [
                LLMChatMessage(role=stored_message.role, content=stored_message.content)
                for stored_message in recent_messages
            ]
            system_prompt = self._build_system_prompt(
                message=normalized_message,
                retrieved_chunks=retrieved_chunks,
            )
            llm_response = await self.llm_service.generate_response(
                llm_messages,
                system_prompt=system_prompt,
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

    def _build_system_prompt(
        self,
        *,
        message: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        base_prompt = self.llm_service.load_system_prompt()
        context_block = self._format_retrieved_context(retrieved_chunks)
        guidance = [
            "Use the retrieved context to answer the visitor's question when it is relevant to Tumelo.",
            "Do not invent experience, projects, employers, dates, tools, certifications, or achievements.",
            "If the approved context does not contain enough Tumelo-specific information, say that you do not have that information available.",
            "If the user is asking a general technical question, you may answer generally, but do not present general knowledge as Tumelo's personal experience.",
        ]
        if not retrieved_chunks:
            guidance.append(
                "No relevant approved Tumelo context was retrieved for this turn, so avoid personal claims unless they are already established in the conversation."
            )

        return "\n\n".join(
            [
                base_prompt,
                "Approved Tumelo knowledge base context:\n" + context_block,
                "Additional rules:\n" + "\n".join(f"- {rule}" for rule in guidance),
                f"Current user question:\n{message}",
            ]
        )

    def _format_retrieved_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No approved context retrieved."

        formatted_chunks = []
        for item in retrieved_chunks:
            formatted_chunks.append(
                "\n".join(
                    [
                        f"Source: {item.source}",
                        f"Section: {item.section}",
                        f"Similarity: {item.similarity:.3f}",
                        item.content,
                    ]
                )
            )
        return "\n\n---\n\n".join(formatted_chunks)

    def _should_use_direct_fallback(self, message: str) -> bool:
        if self._is_personal_query(message):
            return True
        if self._is_general_technical_query(message):
            return False
        return True

    def _is_personal_query(self, message: str) -> bool:
        normalized_message = message.casefold()
        return any(marker in normalized_message for marker in PERSONAL_QUERY_MARKERS)

    def _is_general_technical_query(self, message: str) -> bool:
        normalized_message = message.casefold()
        if self._is_personal_query(message):
            return False
        return any(marker in normalized_message for marker in GENERAL_TECH_MARKERS)

    def _build_direct_response(
        self,
        message: str,
        *,
        personal_query: bool,
    ) -> LLMGeneratedResponse:
        if personal_query:
            response_text = (
                "I do not have enough approved information about that in Tumelo's knowledge base yet."
            )
        else:
            response_text = (
                "Could you clarify whether you're asking about Tumelo's background or a general technical topic?"
            )

        return LLMGeneratedResponse(
            message=response_text,
            model=self.llm_service.model,
            prompt_version=self.llm_service.prompt_version,
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
