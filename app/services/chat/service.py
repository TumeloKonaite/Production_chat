from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from time import perf_counter
import uuid

from app.config import Settings
from app.domain.tracing import TraceStatus, TraceStepType
from app.infrastructure.prompts import PromptLoader, normalize_prompt_version
from app.repositories import (
    ConversationRepository,
    ConversationRepositoryError,
    KnowledgeRepository,
    KnowledgeRepositoryError,
)
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    is_broad_project_query,
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
from app.services.tracing import TraceService, TraceServiceError

logger = logging.getLogger(__name__)
TRACE_PROMPT_PREVIEW_LIMIT = 4000

@dataclass(frozen=True, slots=True)
class ChatReply:
    conversation_id: str
    message: str
    model: str
    model_provider: str
    model_name: str
    model_config_id: str
    prompt_version: str
    retrieval_config: str
    latency_ms: int | None
    token_usage: TokenUsage
    estimated_cost_usd: float | None


class ChatService:
    def __init__(
        self,
        llm_service,
        prompt_loader: PromptLoader,
        repository: ConversationRepository,
        knowledge_repository: KnowledgeRepository,
        retrieval_service: RetrievalService,
        trace_service: TraceService | None,
        history_limit: int,
        retrieval_top_k: int,
        settings: Settings,
    ) -> None:
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.repository = repository
        self.knowledge_repository = knowledge_repository
        self.retrieval_service = retrieval_service
        self.trace_service = trace_service
        self.history_limit = history_limit
        self.retrieval_top_k = retrieval_top_k
        self.settings = settings

    async def generate_reply(
        self,
        message: str,
        conversation_id: str | None = None,
        prompt_version: str | None = None,
        model_config_id: str | None = None,
    ) -> ChatReply:
        return await self._generate_reply(
            message=message,
            conversation_id=conversation_id,
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            channel="web_chat",
            message_metadata={},
            allow_external_conversation_id=False,
        )

    async def generate_answer(
        self,
        *,
        user_message: str,
        conversation_id: str | None = None,
        channel: str = "web_chat",
        metadata: dict[str, object] | None = None,
        prompt_version: str | None = None,
        model_config_id: str | None = None,
    ) -> ChatReply:
        return await self._generate_reply(
            message=user_message,
            conversation_id=conversation_id,
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            channel=channel,
            message_metadata=dict(metadata or {}),
            allow_external_conversation_id=channel == "tavus_video",
        )

    async def _generate_reply(
        self,
        *,
        message: str,
        conversation_id: str | None,
        prompt_version: str | None,
        model_config_id: str | None,
        channel: str,
        message_metadata: dict[str, object],
        allow_external_conversation_id: bool,
    ) -> ChatReply:
        normalized_message = message.strip()
        if not normalized_message:
            raise InvalidChatMessageError("Chat message cannot be empty.")

        selected_prompt_version = normalize_prompt_version(
            prompt_version or self.settings.default_prompt_version
        )
        base_prompt = self.prompt_loader.load(selected_prompt_version)

        conversation = self._get_or_create_conversation(
            conversation_id,
            prompt_version=selected_prompt_version,
            model_config_id=model_config_id,
            allow_external_conversation_id=allow_external_conversation_id,
            external_title=self._extract_title_from_metadata(message_metadata),
        )
        conversation.prompt_version = selected_prompt_version
        selected_model_config = self.llm_service.get_model_config(
            model_config_id or conversation.model
        )
        conversation.model = selected_model_config.config_id
        request_started_at = perf_counter()
        trace_id = self._start_trace(
            conversation_id=conversation.id,
            message=normalized_message,
            channel=channel,
            message_metadata=message_metadata,
            prompt_version=selected_prompt_version,
            model_config_id=selected_model_config.config_id,
        )
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.REQUEST_RECEIVED,
            name="Chat request received",
            input_payload={
                "message": normalized_message,
                "channel": channel,
            },
            metadata=self._build_trace_metadata(channel, message_metadata),
        )

        try:
            # Persist the user turn before calling the LLM so failed generations still leave a trace.
            user_message = self.repository.add_message(
                conversation=conversation,
                role="user",
                content=normalized_message,
                channel=channel,
                message_metadata=message_metadata,
            )
            recent_messages = self.repository.list_recent_messages(
                conversation.id,
                limit=self.history_limit,
            )
            retrieved_chunks = self._trace_retrieve_chunks(normalized_message, trace_id=trace_id)
            use_direct_fallback = should_use_direct_fallback(normalized_message, retrieved_chunks)
            self._log_retrieval(
                conversation_id=conversation.id,
                message_id=user_message.id,
                query=normalized_message,
                retrieved_chunks=retrieved_chunks,
                used_fallback=use_direct_fallback,
            )

            if use_direct_fallback:
                self._record_trace_step(
                    trace_id=trace_id,
                    step_type=TraceStepType.PROMPT_BUILT,
                    name="Direct fallback selected",
                    output_payload={
                        "direct_fallback": True,
                        "retrieved_chunk_count": len(retrieved_chunks),
                        "prompt_version": selected_prompt_version,
                    },
                )
                llm_response = self._build_direct_response(
                    normalized_message,
                    prompt_version=selected_prompt_version,
                    model_config_id=selected_model_config.config_id,
                )
            else:
                llm_messages = [
                    LLMChatMessage(role=stored_message.role, content=stored_message.content)
                    for stored_message in recent_messages
                ]
                prompt_started_at = perf_counter()
                system_prompt = build_chat_system_prompt(
                    base_prompt=base_prompt,
                    message=normalized_message,
                    retrieved_chunks=retrieved_chunks,
                )
                self._record_trace_step(
                    trace_id=trace_id,
                    step_type=TraceStepType.PROMPT_BUILT,
                    name="System prompt built",
                    output_payload={
                        "prompt_version": selected_prompt_version,
                        "history_message_count": len(llm_messages),
                        "retrieved_chunk_count": len(retrieved_chunks),
                        "system_prompt": self._truncate_text(system_prompt),
                        "truncated": len(system_prompt) > TRACE_PROMPT_PREVIEW_LIMIT,
                    },
                    latency_ms=self._elapsed_ms(prompt_started_at),
                )
                llm_started_at = self._utcnow()
                llm_timer = perf_counter()
                self._record_trace_step(
                    trace_id=trace_id,
                    step_type=TraceStepType.LLM_CALL_STARTED,
                    status=TraceStatus.STARTED,
                    name="LLM call started",
                    input_payload={
                        "model_config_id": selected_model_config.config_id,
                        "provider": selected_model_config.provider,
                        "model": selected_model_config.model,
                        "message_count": len(llm_messages),
                    },
                    started_at=llm_started_at,
                )
                llm_response = await self.llm_service.generate_response(
                    llm_messages,
                    system_prompt=system_prompt,
                    prompt_version=selected_prompt_version,
                    retrieval_config=self.settings.default_retrieval_config,
                    model_config_id=selected_model_config.config_id,
                )
                llm_latency_ms = llm_response.latency_ms or self._elapsed_ms(llm_timer)
                self._record_trace_step(
                    trace_id=trace_id,
                    step_type=TraceStepType.LLM_CALL_COMPLETED,
                    name="LLM call completed",
                    output_payload={
                        "model_config_id": llm_response.model_config_id,
                        "provider": llm_response.model_provider,
                        "model": llm_response.model_name,
                        "token_usage": {
                            "input_tokens": llm_response.token_usage.input_tokens,
                            "output_tokens": llm_response.token_usage.output_tokens,
                            "total_tokens": llm_response.token_usage.total_tokens,
                        },
                        "estimated_cost_usd": llm_response.estimated_cost_usd,
                    },
                    latency_ms=llm_latency_ms,
                    started_at=llm_started_at,
                    completed_at=self._utcnow(),
                )

            # Only persist the assistant turn after the final response succeeds.
            self._store_assistant_message(
                conversation=conversation,
                llm_response=llm_response,
                channel=channel,
                message_metadata=message_metadata,
            )
        except ConversationRepositoryError as exc:
            self._fail_trace(
                trace_id=trace_id,
                exc=exc,
                started_at=request_started_at,
                metadata={"conversation_id": conversation.id},
            )
            raise ChatPersistenceError() from exc
        except Exception as exc:
            self._fail_trace(
                trace_id=trace_id,
                exc=exc,
                started_at=request_started_at,
                metadata={"conversation_id": conversation.id},
            )
            raise

        total_latency_ms = self._elapsed_ms(request_started_at)
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.RESPONSE_GENERATED,
            name="Response generated",
            output_payload={
                "conversation_id": conversation.id,
                "model_config_id": llm_response.model_config_id,
                "estimated_cost_usd": llm_response.estimated_cost_usd,
            },
            latency_ms=total_latency_ms,
            completed_at=self._utcnow(),
        )
        self._complete_trace(
            trace_id=trace_id,
            output_text=llm_response.message,
            llm_response=llm_response,
            latency_ms=total_latency_ms,
            channel=channel,
            message_metadata=message_metadata,
        )

        return ChatReply(
            conversation_id=conversation.id,
            message=llm_response.message,
            model=llm_response.model,
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            model_config_id=llm_response.model_config_id,
            prompt_version=llm_response.prompt_version,
            retrieval_config=llm_response.retrieval_config,
            latency_ms=llm_response.latency_ms,
            token_usage=llm_response.token_usage,
            estimated_cost_usd=llm_response.estimated_cost_usd,
        )

    def _get_or_create_conversation(
        self,
        conversation_id: str | None,
        *,
        prompt_version: str,
        model_config_id: str | None,
        allow_external_conversation_id: bool = False,
        external_title: str | None = None,
    ):
        if conversation_id is None:
            return self._create_conversation(
                prompt_version=prompt_version,
                model_config_id=model_config_id,
                title=external_title,
            )

        if allow_external_conversation_id and not self._is_valid_conversation_id(conversation_id):
            return self._get_or_create_external_conversation(
                external_conversation_id=conversation_id,
                prompt_version=prompt_version,
                model_config_id=model_config_id,
                title=external_title,
            )

        self._validate_conversation_id(conversation_id)
        try:
            conversation = self.repository.get_conversation(conversation_id)
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")

        if model_config_id is not None:
            conversation.model = self.llm_service.get_model_config(model_config_id).config_id

        return conversation

    def _create_conversation(
        self,
        *,
        prompt_version: str,
        model_config_id: str | None,
        visitor_id: str | None = None,
        title: str | None = None,
    ):
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        try:
            return self.repository.create_conversation(
                visitor_id=visitor_id,
                title=title,
                model=selected_model_config.config_id,
                prompt_version=prompt_version,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _get_or_create_external_conversation(
        self,
        *,
        external_conversation_id: str,
        prompt_version: str,
        model_config_id: str | None,
        title: str | None,
    ):
        try:
            conversation = self.repository.get_conversation_by_visitor_id(external_conversation_id)
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        if conversation is not None:
            if title and conversation.title != title:
                try:
                    conversation = self.repository.update_conversation(
                        conversation,
                        title=title,
                    )
                except ConversationRepositoryError as exc:
                    raise ChatPersistenceError() from exc
            if model_config_id is not None:
                conversation.model = self.llm_service.get_model_config(model_config_id).config_id
            return conversation

        return self._create_conversation(
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            visitor_id=external_conversation_id,
            title=title,
        )

    def _retrieve_chunks(self, message: str) -> list[RetrievedChunk]:
        try:
            broad_project_chunks = self._retrieve_broad_project_chunks(message)
            if broad_project_chunks:
                return broad_project_chunks
            return self.retrieval_service.retrieve(message, top_k=self.retrieval_top_k)
        except KnowledgeRepositoryError as exc:
            raise ChatServiceError() from exc

    def _trace_retrieve_chunks(
        self,
        message: str,
        *,
        trace_id: str | None,
    ) -> list[RetrievedChunk]:
        started_at = self._utcnow()
        retrieval_timer = perf_counter()
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.RETRIEVAL_STARTED,
            status=TraceStatus.STARTED,
            name="Retrieval started",
            input_payload={
                "query": message,
                "top_k": self.retrieval_top_k,
                "retriever_type": self.settings.retriever_type,
            },
            started_at=started_at,
        )
        retrieved_chunks = self._retrieve_chunks(message)
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.RETRIEVAL_COMPLETED,
            name="Retrieval completed",
            input_payload={
                "query": message,
                "top_k": self.retrieval_top_k,
            },
            output_payload={
                "retrieved_chunks": [
                    {
                        "chunk_id": item.metadata.get("chunk_id", item.id),
                        "source": item.source,
                        "section": item.section,
                        "score": item.similarity,
                    }
                    for item in retrieved_chunks
                ]
            },
            latency_ms=self._elapsed_ms(retrieval_timer),
            started_at=started_at,
            completed_at=self._utcnow(),
        )
        return retrieved_chunks

    def _retrieve_broad_project_chunks(self, message: str) -> list[RetrievedChunk]:
        if not is_broad_project_query(message):
            return []

        seen_sections: set[str] = set()
        project_chunks = []
        for chunk in self.knowledge_repository.list_by_source("projects.md"):
            if chunk.section in seen_sections:
                continue
            if int(chunk.chunk_metadata.get("section_chunk_index", 0)) != 0:
                continue
            seen_sections.add(chunk.section)
            project_chunks.append(chunk)

        project_chunks.sort(
            key=lambda item: int(item.chunk_metadata.get("chunk_index", 0))
        )
        selected_chunks = project_chunks[: self.retrieval_top_k]
        return [
            RetrievedChunk(
                id=chunk.id,
                source=chunk.source,
                section=chunk.section,
                content=chunk.content,
                similarity=max(0.9 - (index * 0.01), 0.75),
                metadata={
                    **chunk.chunk_metadata,
                    "chunk_id": chunk.id,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "section": chunk.section,
                },
            )
            for index, chunk in enumerate(selected_chunks)
        ]

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
        model_config_id: str,
    ) -> LLMGeneratedResponse:
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        return LLMGeneratedResponse(
            message=build_direct_fallback_text(message),
            model=selected_model_config.model,
            model_provider=selected_model_config.provider,
            model_name=selected_model_config.model,
            model_config_id=selected_model_config.config_id,
            prompt_version=prompt_version,
            retrieval_config=self.settings.default_retrieval_config,
            latency_ms=0,
            token_usage=TokenUsage(),
            estimated_cost_usd=None,
        )

    def _store_assistant_message(
        self,
        *,
        conversation,
        llm_response: LLMGeneratedResponse,
        channel: str,
        message_metadata: dict[str, object],
    ) -> None:
        try:
            self.repository.add_message(
                conversation=conversation,
                role="assistant",
                content=llm_response.message,
                channel=channel,
                model=llm_response.model,
                model_provider=llm_response.model_provider,
                model_name=llm_response.model_name,
                model_config_id=llm_response.model_config_id,
                prompt_version=llm_response.prompt_version,
                retrieval_config=llm_response.retrieval_config,
                latency_ms=llm_response.latency_ms,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
                estimated_cost_usd=llm_response.estimated_cost_usd,
                message_metadata=message_metadata,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _validate_conversation_id(self, conversation_id: str) -> None:
        if not self._is_valid_conversation_id(conversation_id):
            raise InvalidConversationIdError("conversation_id must be a valid UUID.")

    def _is_valid_conversation_id(self, conversation_id: str) -> bool:
        try:
            uuid.UUID(conversation_id)
        except ValueError:
            return False
        return True

    def _extract_title_from_metadata(self, metadata: dict[str, object]) -> str | None:
        visitor_name = metadata.get("visitor_name")
        if not isinstance(visitor_name, str):
            return None
        normalized_visitor_name = visitor_name.strip()
        return normalized_visitor_name or None

    def _start_trace(
        self,
        *,
        conversation_id: str,
        message: str,
        channel: str,
        message_metadata: dict[str, object],
        prompt_version: str,
        model_config_id: str,
    ) -> str | None:
        if self.trace_service is None:
            return None

        try:
            trace = self.trace_service.start_trace(
                conversation_id=conversation_id,
                user_id=self._extract_optional_string(message_metadata, "user_id"),
                request_id=self._extract_optional_string(message_metadata, "request_id"),
                session_id=self._extract_optional_string(message_metadata, "session_id"),
                input_text=message,
                status=TraceStatus.STARTED,
                llm_provider=self._extract_model_provider(model_config_id),
                llm_model=self._extract_model_name(model_config_id),
                prompt_version=prompt_version,
                retriever_type=self.settings.retriever_type,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                metadata=self._build_trace_metadata(channel, message_metadata),
            )
        except TraceServiceError:
            logger.warning("Trace start failed.", exc_info=True)
            return None

        return trace.id

    def _record_trace_step(
        self,
        *,
        trace_id: str | None,
        step_type: TraceStepType,
        status: TraceStatus = TraceStatus.SUCCESS,
        name: str | None = None,
        input_payload: dict[str, object] | None = None,
        output_payload: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if self.trace_service is None or trace_id is None:
            return

        try:
            self.trace_service.add_step(
                trace_id=trace_id,
                step_type=step_type,
                status=status,
                name=name,
                input_payload=input_payload,
                output_payload=output_payload,
                metadata=metadata,
                latency_ms=latency_ms,
                error_message=error_message,
                started_at=started_at,
                completed_at=completed_at,
            )
        except TraceServiceError:
            logger.warning("Trace step write failed.", exc_info=True)

    def _complete_trace(
        self,
        *,
        trace_id: str | None,
        output_text: str,
        llm_response: LLMGeneratedResponse,
        latency_ms: int,
        channel: str,
        message_metadata: dict[str, object],
    ) -> None:
        if self.trace_service is None or trace_id is None:
            return

        try:
            self.trace_service.complete_trace(
                trace_id,
                output_text=output_text,
                status=TraceStatus.SUCCESS,
                llm_provider=llm_response.model_provider,
                llm_model=llm_response.model_name,
                prompt_version=llm_response.prompt_version,
                retriever_type=self.settings.retriever_type,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
                estimated_cost_usd=llm_response.estimated_cost_usd,
                latency_ms=latency_ms,
                metadata=self._build_trace_metadata(channel, message_metadata),
            )
        except TraceServiceError:
            logger.warning("Trace completion failed.", exc_info=True)

    def _fail_trace(
        self,
        *,
        trace_id: str | None,
        exc: Exception,
        started_at: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        safe_error_message = self._safe_trace_error_message(exc)
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.ERROR,
            status=TraceStatus.ERROR,
            name="Request failed",
            metadata=metadata,
            latency_ms=self._elapsed_ms(started_at),
            error_message=safe_error_message,
            completed_at=self._utcnow(),
        )
        if self.trace_service is None or trace_id is None:
            return

        try:
            self.trace_service.fail_trace(
                trace_id,
                error_message=safe_error_message,
                latency_ms=self._elapsed_ms(started_at),
                metadata=metadata,
            )
        except TraceServiceError:
            logger.warning("Trace failure write failed.", exc_info=True)

    def _build_trace_metadata(
        self,
        channel: str,
        message_metadata: dict[str, object],
    ) -> dict[str, object]:
        metadata = {
            "channel": channel,
            "route": "/api/tavus/tools/ask-tumelo" if channel == "tavus_video" else "/chat",
        }
        metadata.update(message_metadata)
        return metadata

    def _safe_trace_error_message(self, exc: Exception) -> str:
        if isinstance(exc, (ChatPersistenceError, ConversationRepositoryError)):
            return "Unable to save chat conversation. Please try again."
        if exc.__class__.__name__ == "LLMServiceError":
            return "Unable to generate assistant response. Please try again."
        if isinstance(exc, ChatServiceError):
            return "Unable to generate assistant response. Please try again."
        return exc.__class__.__name__

    def _extract_optional_string(
        self,
        metadata: dict[str, object],
        key: str,
    ) -> str | None:
        value = metadata.get(key)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _extract_model_provider(self, model_config_id: str) -> str | None:
        if ":" not in model_config_id:
            return None
        provider, _model = model_config_id.split(":", 1)
        normalized_provider = provider.strip()
        return normalized_provider or None

    def _extract_model_name(self, model_config_id: str) -> str:
        if ":" not in model_config_id:
            return model_config_id
        _provider, model = model_config_id.split(":", 1)
        return model.strip() or model_config_id

    def _truncate_text(self, value: str) -> str:
        if len(value) <= TRACE_PROMPT_PREVIEW_LIMIT:
            return value
        return value[:TRACE_PROMPT_PREVIEW_LIMIT]

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)
