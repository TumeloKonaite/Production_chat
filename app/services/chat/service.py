from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
from time import perf_counter
import uuid

from app.config import Settings
from app.domain.tracing import TraceStatus, TraceStepType
from app.infrastructure.cache import normalize_whitespace, stable_json_hash
from app.infrastructure.observability import ObservabilityTrace, ObservabilityTracer
from app.infrastructure.prompts import PromptLoader, normalize_prompt_version
from app.repositories import (
    ConversationRepository,
    ConversationRepositoryError,
    KnowledgeRepository,
    KnowledgeRepositoryError,
)
from app.services.cache import (
    CacheLookupRequest,
    CacheLookupResult,
    CacheScope,
    CacheStoreEntry,
    DuplicateRequestInProgressError,
    ResponseCache,
    RequestLock,
    hash_scope,
    normalize_question,
    stable_hash,
)
from app.services.chat.prompting import (
    build_chat_system_prompt,
)
from app.services.chat.context_resolution import ConversationContextResolver
from app.services.chat.models import (
    DirectResponseDecision,
    PortfolioKnowledgeDecision,
    QueryRoute,
    RetrievalMode,
    RoutingDecision,
)
from app.services.chat.routing import PortfolioScopeRouter, build_direct_response_text
from app.services.chat.errors import (
    ChatPersistenceError,
    ChatServiceError,
    ConversationNotFoundError,
    InvalidChatMessageError,
    InvalidConversationIdError,
)
from app.services.llm.service import LLMChatMessage, LLMGeneratedResponse, TokenUsage
from app.services.rate_limiting.schemas import RateLimitActor
from app.services.rate_limiting.service import RateLimitingService
from app.services.retrieval import RetrievedChunk, RetrievalService
from app.services.tracing import TraceService, TraceServiceError

logger = logging.getLogger(__name__)
TRACE_PROMPT_PREVIEW_LIMIT = 4000


@dataclass(frozen=True, slots=True)
class ChatReply:
    conversation_id: str
    message_id: str
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
    response_cache_hit: bool
    response_cache_type: str | None
    response_cache_reason: str
    response_cache_distance: float | None


@dataclass(slots=True)
class ResponseCacheContext:
    enabled: bool
    provider: str | None
    exact_enabled: bool
    semantic_enabled: bool
    hit: bool = False
    cache_type: str | None = None
    reason: str = "disabled"
    write_reason: str | None = None
    entry_id: str | None = None
    distance: float | None = None
    threshold: float | None = None
    lookup_latency_ms: int = 0
    exact_lookup_latency_ms: int = 0
    semantic_lookup_latency_ms: int = 0
    embedding_latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    llm_latency_ms: int | None = None
    total_latency_ms: int | None = None
    latency_saved_estimate_ms: int | None = None


@dataclass(slots=True)
class RoutingExecutionContext:
    decision: RoutingDecision | None = None
    context_resolution_attempted: bool = False
    context_resolution_succeeded: bool = False
    routing_latency_ms: int = 0
    retrieval_attempted: bool = False
    retrieval_result_count: int = 0
    retrieval_used: bool = False
    generation_used: bool = False
    grounded_not_found: bool = False

    def metadata(self) -> dict[str, object]:
        decision = self.decision
        return {
            "query_route": decision.route.value if decision is not None else None,
            "reason_code": decision.reason_code if decision is not None else None,
            "resolved_query": decision.resolved_query if decision is not None else None,
            "context_resolution_attempted": self.context_resolution_attempted,
            "context_resolution_succeeded": self.context_resolution_succeeded,
            "retrieval_mode": (
                decision.retrieval_mode.value
                if isinstance(decision, PortfolioKnowledgeDecision)
                else None
            ),
            "direct_response_kind": (
                decision.direct_response_kind.value
                if isinstance(decision, DirectResponseDecision)
                else None
            ),
            "retrieval_attempted": self.retrieval_attempted,
            "retrieval_result_count": self.retrieval_result_count,
            "retrieval_used": self.retrieval_used,
            "generation_used": self.generation_used,
            "grounded_not_found": self.grounded_not_found,
            "routing_latency_ms": self.routing_latency_ms,
        }


class ChatService:
    def __init__(
        self,
        llm_service,
        prompt_loader: PromptLoader,
        repository: ConversationRepository,
        knowledge_repository: KnowledgeRepository,
        retrieval_service: RetrievalService,
        response_cache: ResponseCache,
        request_lock: RequestLock,
        rate_limiting_service: RateLimitingService,
        trace_service: TraceService | None,
        observability_tracer: ObservabilityTracer,
        history_limit: int,
        retrieval_top_k: int,
        settings: Settings,
    ) -> None:
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.repository = repository
        self.knowledge_repository = knowledge_repository
        self.retrieval_service = retrieval_service
        self.response_cache = response_cache
        self.request_lock = request_lock
        self.rate_limiting_service = rate_limiting_service
        self.trace_service = trace_service
        self.observability_tracer = observability_tracer
        self.history_limit = history_limit
        self.retrieval_top_k = retrieval_top_k
        self.settings = settings
        self.context_resolver = ConversationContextResolver()
        self.portfolio_scope_router = PortfolioScopeRouter(self.context_resolver)

    async def generate_reply(
        self,
        message: str,
        conversation_id: str | None = None,
        prompt_version: str | None = None,
        model_config_id: str | None = None,
        rate_limit_actor: RateLimitActor | None = None,
    ) -> ChatReply:
        return await self._generate_reply(
            message=message,
            conversation_id=conversation_id,
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            rate_limit_actor=rate_limit_actor,
            channel="web_chat",
            message_metadata={},
        )

    async def _generate_reply(
        self,
        *,
        message: str,
        conversation_id: str | None,
        prompt_version: str | None,
        model_config_id: str | None,
        rate_limit_actor: RateLimitActor | None,
        channel: str,
        message_metadata: dict[str, object],
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
        )
        conversation.prompt_version = selected_prompt_version
        selected_model_config = self.llm_service.get_model_config(
            model_config_id or conversation.model
        )
        conversation.model = selected_model_config.config_id
        request_started_at = perf_counter()
        response_cache_context = self._build_response_cache_context()
        routing_context = RoutingExecutionContext()
        response_message_metadata = dict(message_metadata)
        trace_id = self._start_trace(
            conversation_id=conversation.id,
            message=normalized_message,
            channel=channel,
            message_metadata=message_metadata,
            prompt_version=selected_prompt_version,
            model_config_id=selected_model_config.config_id,
        )
        observability_trace = self._start_observability_trace(
            conversation_id=conversation.id,
            message=normalized_message,
            channel=channel,
            message_metadata=message_metadata,
            prompt_version=selected_prompt_version,
            model_config_id=selected_model_config.config_id,
            internal_trace_id=trace_id,
            response_cache_context=response_cache_context,
        )
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.REQUEST_RECEIVED,
            name="Chat request received",
            input_payload={
                "message": normalized_message,
                "channel": channel,
            },
            metadata=self._build_trace_metadata(
                conversation_id=conversation.id,
                channel=channel,
                message_metadata=message_metadata,
                prompt_version=selected_prompt_version,
                model_config_id=selected_model_config.config_id,
                internal_trace_id=trace_id,
                observability_trace=observability_trace,
            ),
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
            llm_response: LLMGeneratedResponse
            cached_response = None
            cache_request: CacheLookupRequest | None = None
            if self._is_response_cache_candidate(
                message=normalized_message,
                conversation_id=conversation_id,
                channel=channel,
                message_metadata=message_metadata,
            ):
                cache_request = self._build_cache_lookup_request(
                    message=normalized_message,
                    prompt_version=selected_prompt_version,
                    model_config_id=selected_model_config.config_id,
                )
                cached_response = await self._lookup_cached_response(
                    cache_request=cache_request,
                    response_cache_context=response_cache_context,
                )

            retrieved_chunks: list[RetrievedChunk] = []
            use_direct_fallback = False
            if cached_response is not None:
                llm_response = cached_response
            else:
                lock_acquired = False
                try:
                    if cache_request is not None:
                        lock_acquired = await self.request_lock.acquire(
                            cache_request.request_hash
                        )
                        if not lock_acquired:
                            cached_response = await self._lookup_cached_response(
                                cache_request=cache_request,
                                response_cache_context=response_cache_context,
                            )
                            if cached_response is not None:
                                llm_response = cached_response
                            else:
                                raise DuplicateRequestInProgressError(
                                    "An identical request is already being processed. Please retry shortly."
                                )

                    if cached_response is not None:
                        llm_response = cached_response
                    else:
                        recent_messages = self.repository.list_recent_messages(
                            conversation.id,
                            limit=self.history_limit,
                        )
                        routing_started_at = perf_counter()
                        resolved_query = None
                        if self.context_resolver.requires_resolution(
                            normalized_message
                        ):
                            routing_context.context_resolution_attempted = True
                            resolved_query = self.context_resolver.resolve(
                                message=normalized_message,
                                recent_messages=recent_messages,
                            )
                            routing_context.context_resolution_succeeded = (
                                resolved_query is not None
                            )
                        decision = self.portfolio_scope_router.route(
                            message=normalized_message,
                            resolved_query=resolved_query,
                            recent_messages=list(recent_messages),
                        )
                        routing_context.decision = decision
                        routing_context.routing_latency_ms = self._elapsed_ms(
                            routing_started_at
                        )
                        if decision.route == QueryRoute.DIRECT_RESPONSE:
                            use_direct_fallback = True
                            self._record_trace_step(
                                trace_id=trace_id,
                                step_type=TraceStepType.PROMPT_BUILT,
                                name="Deterministic direct response selected",
                                output_payload={
                                    "direct_fallback": True,
                                    **routing_context.metadata(),
                                    "prompt_version": selected_prompt_version,
                                },
                            )
                            llm_response = self._build_direct_response(
                                normalized_message,
                                decision=decision,
                                prompt_version=selected_prompt_version,
                                model_config_id=selected_model_config.config_id,
                            )
                            response_cache_context.llm_latency_ms = 0
                        else:
                            retrieval_observation = self._start_observability_retrieval(
                                observability_trace,
                                message=decision.resolved_query,
                            )
                            routing_context.retrieval_attempted = True
                            retrieved_chunks, retrieval_latency_ms = (
                                self._trace_retrieve_chunks(
                                    decision.resolved_query,
                                    retrieval_mode=decision.retrieval_mode,
                                    trace_id=trace_id,
                                    query_embedding=None,
                                )
                            )
                            routing_context.retrieval_result_count = len(
                                retrieved_chunks
                            )
                            routing_context.retrieval_used = bool(retrieved_chunks)
                            response_cache_context.retrieval_latency_ms = (
                                retrieval_latency_ms
                            )
                            self._complete_observability_retrieval(
                                observability_trace,
                                retrieval_observation=retrieval_observation,
                                retrieved_chunks=retrieved_chunks,
                                latency_ms=retrieval_latency_ms,
                            )
                            self._log_retrieval(
                                conversation_id=conversation.id,
                                message_id=user_message.id,
                                query=decision.resolved_query,
                                retrieved_chunks=retrieved_chunks,
                                used_fallback=not retrieved_chunks,
                            )

                            if not retrieved_chunks:
                                use_direct_fallback = True
                                routing_context.grounded_not_found = True
                                llm_response = self._build_grounded_not_found_response(
                                    prompt_version=selected_prompt_version,
                                    model_config_id=selected_model_config.config_id,
                                )
                                response_cache_context.llm_latency_ms = 0
                            else:
                                await self.rate_limiting_service.enforce_chat_budget(
                                    actor=rate_limit_actor,
                                )
                                routing_context.generation_used = True
                                llm_messages = [
                                    LLMChatMessage(
                                        role=stored_message.role,
                                        content=stored_message.content,
                                    )
                                    for stored_message in recent_messages
                                ]
                                prompt_started_at = perf_counter()
                                system_prompt = build_chat_system_prompt(
                                    base_prompt=base_prompt,
                                    message=decision.resolved_query,
                                    retrieved_chunks=retrieved_chunks,
                                )
                                self._record_trace_step(
                                    trace_id=trace_id,
                                    step_type=TraceStepType.PROMPT_BUILT,
                                    name="Grounded system prompt built",
                                    output_payload={
                                        "prompt_version": selected_prompt_version,
                                        "history_message_count": len(llm_messages),
                                        "retrieved_chunk_count": len(retrieved_chunks),
                                        "system_prompt": self._truncate_text(
                                            system_prompt
                                        ),
                                        "truncated": len(system_prompt)
                                        > TRACE_PROMPT_PREVIEW_LIMIT,
                                        **routing_context.metadata(),
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
                                llm_observation = self._start_observability_llm_call(
                                    observability_trace,
                                    provider=selected_model_config.provider,
                                    model=selected_model_config.model,
                                )
                                try:
                                    llm_response = await self.llm_service.generate_response(
                                        llm_messages,
                                        system_prompt=system_prompt,
                                        prompt_version=selected_prompt_version,
                                        retrieval_config=self.settings.default_retrieval_config,
                                        model_config_id=selected_model_config.config_id,
                                    )
                                except Exception as exc:
                                    self._complete_observability_llm_call(
                                        observability_trace,
                                        llm_observation=llm_observation,
                                        provider=selected_model_config.provider,
                                        model=selected_model_config.model,
                                        latency_ms=self._elapsed_ms(llm_timer),
                                        input_tokens=None,
                                        output_tokens=None,
                                        total_tokens=None,
                                        estimated_cost_usd=None,
                                        error_message=self._safe_trace_error_message(
                                            exc
                                        ),
                                    )
                                    raise
                                llm_latency_ms = (
                                    llm_response.latency_ms
                                    or self._elapsed_ms(llm_timer)
                                )
                                response_cache_context.llm_latency_ms = llm_latency_ms
                                self._complete_observability_llm_call(
                                    observability_trace,
                                    llm_observation=llm_observation,
                                    provider=llm_response.model_provider,
                                    model=llm_response.model_name,
                                    latency_ms=llm_latency_ms,
                                    input_tokens=llm_response.token_usage.input_tokens,
                                    output_tokens=llm_response.token_usage.output_tokens,
                                    total_tokens=llm_response.token_usage.total_tokens,
                                    estimated_cost_usd=llm_response.estimated_cost_usd,
                                )
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
                                await self.rate_limiting_service.record_llm_usage(
                                    actor=rate_limit_actor,
                                    total_tokens=llm_response.token_usage.total_tokens,
                                    estimated_cost_usd=llm_response.estimated_cost_usd,
                                )

                        if self._should_store_response_cache(
                            message=normalized_message,
                            conversation_id=conversation_id,
                            channel=channel,
                            message_metadata=message_metadata,
                            retrieved_chunks=retrieved_chunks,
                            llm_response=llm_response,
                            use_direct_fallback=use_direct_fallback,
                        ):
                            await self._store_response_cache_entry(
                                cache_request=cache_request,
                                llm_response=llm_response,
                                retrieved_chunks=retrieved_chunks,
                                response_cache_context=response_cache_context,
                            )
                        else:
                            response_cache_context.write_reason = "write_skipped"
                finally:
                    if lock_acquired and cache_request is not None:
                        await self.request_lock.release(cache_request.request_hash)

            if routing_context.decision is not None:
                routing_metadata = routing_context.metadata()
                response_message_metadata.update(routing_metadata)
                response_message_metadata["routing"] = routing_metadata

            # Only persist the assistant turn after the final response succeeds.
            assistant_message = self._store_assistant_message(
                conversation=conversation,
                llm_response=llm_response,
                channel=channel,
                message_metadata=response_message_metadata,
                trace_id=trace_id,
                observability_trace=observability_trace,
                response_cache_context=response_cache_context,
            )
        except ConversationRepositoryError as exc:
            self._fail_trace(
                trace_id=trace_id,
                conversation_id=conversation.id,
                channel=channel,
                message_metadata=message_metadata,
                prompt_version=selected_prompt_version,
                model_config_id=selected_model_config.config_id,
                observability_trace=observability_trace,
                exc=exc,
                started_at=request_started_at,
                metadata={
                    "conversation_id": conversation.id,
                    "response_cache": self._response_cache_metadata(
                        response_cache_context
                    ),
                },
            )
            self._fail_observability_trace(
                observability_trace,
                exc=exc,
                started_at=request_started_at,
            )
            raise ChatPersistenceError() from exc
        except Exception as exc:
            self._fail_trace(
                trace_id=trace_id,
                conversation_id=conversation.id,
                channel=channel,
                message_metadata=message_metadata,
                prompt_version=selected_prompt_version,
                model_config_id=selected_model_config.config_id,
                observability_trace=observability_trace,
                exc=exc,
                started_at=request_started_at,
                metadata={
                    "conversation_id": conversation.id,
                    "response_cache": self._response_cache_metadata(
                        response_cache_context
                    ),
                },
            )
            self._fail_observability_trace(
                observability_trace,
                exc=exc,
                started_at=request_started_at,
            )
            raise

        total_latency_ms = self._elapsed_ms(request_started_at)
        response_cache_context.total_latency_ms = total_latency_ms
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.RESPONSE_GENERATED,
            name="Response generated",
            output_payload={
                "conversation_id": conversation.id,
                "model_config_id": llm_response.model_config_id,
                "estimated_cost_usd": llm_response.estimated_cost_usd,
                **routing_context.metadata(),
            },
            latency_ms=total_latency_ms,
            completed_at=self._utcnow(),
        )
        self._complete_trace(
            trace_id=trace_id,
            output_text=llm_response.message,
            llm_response=llm_response,
            latency_ms=total_latency_ms,
            conversation_id=conversation.id,
            channel=channel,
            message_metadata=response_message_metadata,
            assistant_message_id=assistant_message.id,
            observability_trace=observability_trace,
            response_cache_context=response_cache_context,
        )
        self._complete_observability_trace(
            observability_trace,
            llm_response=llm_response,
            latency_ms=total_latency_ms,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message.id,
            response_cache_context=response_cache_context,
            routing_context=routing_context,
        )
        self._log_response_cache_context(response_cache_context)
        self._flush_observability_tracer()

        return ChatReply(
            conversation_id=conversation.id,
            message_id=assistant_message.id,
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
            response_cache_hit=response_cache_context.hit,
            response_cache_type=response_cache_context.cache_type,
            response_cache_reason=response_cache_context.reason,
            response_cache_distance=response_cache_context.distance,
        )

    def _get_or_create_conversation(
        self,
        conversation_id: str | None,
        *,
        prompt_version: str,
        model_config_id: str | None,
    ):
        if conversation_id is None:
            return self._create_conversation(
                prompt_version=prompt_version,
                model_config_id=model_config_id,
            )

        self._validate_conversation_id(conversation_id)
        try:
            conversation = self.repository.get_conversation(conversation_id)
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")

        if model_config_id is not None:
            conversation.model = self.llm_service.get_model_config(
                model_config_id
            ).config_id

        return conversation

    def _create_conversation(
        self,
        *,
        prompt_version: str,
        model_config_id: str | None,
    ):
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        try:
            return self.repository.create_conversation(
                visitor_id=None,
                title=None,
                model=selected_model_config.config_id,
                prompt_version=prompt_version,
            )
        except ConversationRepositoryError as exc:
            raise ChatPersistenceError() from exc

    def _retrieve_chunks(
        self,
        message: str,
        *,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        try:
            if retrieval_mode == RetrievalMode.PROJECT_OVERVIEW:
                return self._retrieve_broad_project_chunks()
            return self.retrieval_service.retrieve(
                message,
                top_k=self.retrieval_top_k,
                query_embedding=query_embedding,
            )
        except KnowledgeRepositoryError as exc:
            raise ChatServiceError() from exc

    def _trace_retrieve_chunks(
        self,
        message: str,
        *,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        trace_id: str | None,
        query_embedding: list[float] | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
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
                "retrieval_mode": retrieval_mode.value,
                "similarity_threshold": self.settings.retrieval_min_similarity,
                "reranker_enabled": self.retrieval_service.reranker_enabled,
                "reranker_type": self.retrieval_service.reranker_type,
            },
            started_at=started_at,
        )
        retrieved_chunks = self._retrieve_chunks(
            message,
            retrieval_mode=retrieval_mode,
            query_embedding=query_embedding,
        )
        latency_ms = self._elapsed_ms(retrieval_timer)
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
                        "source_document_id": (
                            item.metadata.get("source_document_id")
                            or item.metadata.get("document_id")
                            or item.metadata.get("knowledge_file_id")
                        ),
                        "source": item.source,
                        "source_type": item.metadata.get("source_type"),
                        "section": item.section,
                        "retrieval_rank": item.metadata.get("retrieval_rank"),
                        "final_rank": item.metadata.get("final_rank"),
                        "score": item.similarity,
                    }
                    for item in retrieved_chunks
                ]
            },
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=self._utcnow(),
        )
        return retrieved_chunks, latency_ms

    def _retrieve_broad_project_chunks(self) -> list[RetrievedChunk]:
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
        decision: DirectResponseDecision,
        prompt_version: str,
        model_config_id: str,
    ) -> LLMGeneratedResponse:
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        return LLMGeneratedResponse(
            message=build_direct_response_text(decision.direct_response_kind, message),
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

    def _build_grounded_not_found_response(
        self,
        *,
        prompt_version: str,
        model_config_id: str,
    ) -> LLMGeneratedResponse:
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        return LLMGeneratedResponse(
            message=(
                "I do not have enough approved information to answer that accurately. "
                "You could ask about Tumelo's listed projects, technical skills, education, "
                "or work experience."
            ),
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
        trace_id: str | None,
        observability_trace: ObservabilityTrace,
        response_cache_context: ResponseCacheContext,
    ):
        assistant_message_metadata = dict(message_metadata)
        if trace_id is not None:
            assistant_message_metadata["trace_id"] = trace_id
        if observability_trace.provider is not None:
            assistant_message_metadata["observability_provider"] = (
                observability_trace.provider
            )
        if observability_trace.external_trace_id is not None:
            assistant_message_metadata["external_trace_id"] = (
                observability_trace.external_trace_id
            )
            if observability_trace.provider == "langfuse":
                assistant_message_metadata["langfuse_trace_id"] = (
                    observability_trace.external_trace_id
                )
        if response_cache_context.enabled:
            assistant_message_metadata["response_cache"] = (
                self._response_cache_message_metadata(response_cache_context)
            )
        try:
            return self.repository.add_message(
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
                message_metadata=assistant_message_metadata,
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
                request_id=self._extract_optional_string(
                    message_metadata, "request_id"
                ),
                session_id=self._extract_optional_string(
                    message_metadata, "session_id"
                ),
                input_text=message,
                status=TraceStatus.STARTED,
                llm_provider=self._extract_model_provider(model_config_id),
                llm_model=self._extract_model_name(model_config_id),
                prompt_version=prompt_version,
                retriever_type=self.settings.retriever_type,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                metadata=self._build_trace_metadata(
                    conversation_id=conversation_id,
                    channel=channel,
                    message_metadata=message_metadata,
                    prompt_version=prompt_version,
                    model_config_id=model_config_id,
                    internal_trace_id=None,
                ),
            )
        except TraceServiceError:
            logger.warning("Trace start failed.")
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
            logger.warning("Trace step write failed.")

    def _complete_trace(
        self,
        *,
        trace_id: str | None,
        output_text: str,
        llm_response: LLMGeneratedResponse,
        latency_ms: int,
        conversation_id: str,
        channel: str,
        message_metadata: dict[str, object],
        assistant_message_id: str,
        observability_trace: ObservabilityTrace,
        response_cache_context: ResponseCacheContext,
    ) -> None:
        if self.trace_service is None or trace_id is None:
            return

        try:
            trace_metadata = self._build_trace_metadata(
                conversation_id=conversation_id,
                channel=channel,
                message_metadata=message_metadata,
                prompt_version=llm_response.prompt_version,
                model_config_id=llm_response.model_config_id,
                internal_trace_id=trace_id,
                observability_trace=observability_trace,
            )
            trace_metadata["assistant_message_id"] = assistant_message_id
            trace_metadata["response_cache"] = self._response_cache_metadata(
                response_cache_context
            )
            self.trace_service.complete_trace(
                trace_id,
                output_text=output_text,
                status=TraceStatus.SUCCESS,
                llm_provider=llm_response.model_provider,
                llm_model=llm_response.model_name,
                observability_provider=observability_trace.provider,
                external_trace_id=observability_trace.external_trace_id,
                prompt_version=llm_response.prompt_version,
                retriever_type=self.settings.retriever_type,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
                estimated_cost_usd=llm_response.estimated_cost_usd,
                latency_ms=latency_ms,
                metadata=trace_metadata,
            )
        except TraceServiceError:
            logger.warning("Trace completion failed.")

    def _fail_trace(
        self,
        *,
        trace_id: str | None,
        conversation_id: str,
        channel: str,
        message_metadata: dict[str, object],
        prompt_version: str,
        model_config_id: str,
        observability_trace: ObservabilityTrace,
        exc: Exception,
        started_at: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        safe_error_message = self._safe_trace_error_message(exc)
        trace_metadata = self._build_trace_metadata(
            conversation_id=conversation_id,
            channel=channel,
            message_metadata=message_metadata,
            prompt_version=prompt_version,
            model_config_id=model_config_id,
            internal_trace_id=trace_id,
            observability_trace=observability_trace,
        )
        if metadata:
            trace_metadata.update(metadata)
        self._record_trace_step(
            trace_id=trace_id,
            step_type=TraceStepType.ERROR,
            status=TraceStatus.ERROR,
            name="Request failed",
            metadata=trace_metadata,
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
                observability_provider=observability_trace.provider,
                external_trace_id=observability_trace.external_trace_id,
                metadata=trace_metadata,
            )
        except TraceServiceError:
            logger.warning("Trace failure write failed.")

    def _build_trace_metadata(
        self,
        *,
        conversation_id: str,
        channel: str,
        message_metadata: dict[str, object],
        prompt_version: str | None,
        model_config_id: str | None,
        internal_trace_id: str | None,
        observability_trace: ObservabilityTrace | None = None,
    ) -> dict[str, object]:
        metadata = {
            "app_environment": self.settings.app_env,
            "channel": channel,
            "route": self._get_endpoint(channel),
            "endpoint_name": self._get_endpoint_name(channel),
            "conversation_id": conversation_id,
            "prompt_version": prompt_version,
            "model_config_id": model_config_id,
            "llm_provider": self._extract_model_provider(model_config_id),
            "llm_model": self._extract_model_name(model_config_id)
            if model_config_id is not None
            else None,
            "retriever_type": self.settings.retriever_type,
            "vector_store_provider": self.settings.vector_store_provider,
            "vector_store_name": self._get_vector_store_name(),
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.knowledge_embedding_model,
            "embedding_dimension": self.settings.embedding_dimension,
            "top_k": self.retrieval_top_k,
            "similarity_threshold": self.settings.retrieval_min_similarity,
            "query_rewriting_enabled": self.settings.enable_query_rewriting,
            "reranker_enabled": self.retrieval_service.reranker_enabled,
            "reranker_type": self.retrieval_service.reranker_type,
            "reranker_model": self._get_reranker_model(),
        }
        metadata.update(self._sanitize_metadata_mapping(message_metadata))
        metadata["internal_trace_id"] = internal_trace_id
        if observability_trace is not None:
            if observability_trace.provider is not None:
                metadata["observability_provider"] = observability_trace.provider
            if observability_trace.external_trace_id is not None:
                metadata["external_trace_id"] = observability_trace.external_trace_id
                if observability_trace.provider == "langfuse":
                    metadata["langfuse_trace_id"] = (
                        observability_trace.external_trace_id
                    )
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

    def _extract_model_provider(self, model_config_id: str | None) -> str | None:
        if model_config_id is None:
            return None
        if ":" not in model_config_id:
            return None
        provider, _model = model_config_id.split(":", 1)
        normalized_provider = provider.strip()
        return normalized_provider or None

    def _extract_model_name(self, model_config_id: str | None) -> str | None:
        if model_config_id is None:
            return None
        if ":" not in model_config_id:
            return model_config_id
        _provider, model = model_config_id.split(":", 1)
        return model.strip() or model_config_id

    def _get_reranker_model(self) -> str | None:
        value = getattr(self.settings, "reranker_model", None)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _sanitize_metadata_mapping(
        self,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        return {
            str(key): self._sanitize_metadata_value(value)
            for key, value in metadata.items()
            if value is not None
        }

    def _sanitize_metadata_value(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            normalized = value
            if normalized.tzinfo is None:
                normalized = normalized.replace(tzinfo=UTC)
            return normalized.isoformat()
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, list):
            return [self._sanitize_metadata_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_metadata_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._sanitize_metadata_value(item)
                for key, item in value.items()
                if item is not None
            }
        return str(value)

    def _log_observability_failure(self, stage: str, exc: Exception) -> None:
        logger.warning(
            "Observability %s failed: %s (%s)",
            stage,
            exc.__class__.__name__,
            self._safe_observability_error_message(exc),
        )

    def _safe_observability_error_message(self, exc: Exception) -> str:
        raw_message = str(exc).strip()
        if not raw_message:
            return "no-details"

        redacted = raw_message
        for secret in (
            self.settings.langfuse_public_key,
            self.settings.langfuse_secret_key,
            self.settings.openai_api_key,
            self.settings.openrouter_api_key,
            self.settings.llm_api_key,
        ):
            if secret:
                redacted = redacted.replace(secret, "[redacted]")
        return redacted

    def _truncate_text(self, value: str) -> str:
        if len(value) <= TRACE_PROMPT_PREVIEW_LIMIT:
            return value
        return value[:TRACE_PROMPT_PREVIEW_LIMIT]

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)

    def _start_observability_trace(
        self,
        *,
        conversation_id: str,
        message: str,
        channel: str,
        message_metadata: dict[str, object],
        prompt_version: str,
        model_config_id: str,
        internal_trace_id: str | None,
        response_cache_context: ResponseCacheContext,
    ) -> ObservabilityTrace:
        try:
            return self.observability_tracer.start_chat_request(
                question=message,
                conversation_id=conversation_id,
                session_id=(
                    self._extract_optional_string(message_metadata, "session_id")
                    or conversation_id
                ),
                user_id=self._extract_optional_string(message_metadata, "user_id"),
                endpoint=self._get_endpoint(channel),
                endpoint_name=self._get_endpoint_name(channel),
                channel=channel,
                llm_provider=self._extract_model_provider(model_config_id),
                llm_model=self._extract_model_name(model_config_id),
                metadata=self._sanitize_metadata_mapping(
                    {
                        **self._build_trace_metadata(
                            conversation_id=conversation_id,
                            channel=channel,
                            message_metadata=message_metadata,
                            prompt_version=prompt_version,
                            model_config_id=model_config_id,
                            internal_trace_id=internal_trace_id,
                        ),
                        **self._response_cache_config_metadata(response_cache_context),
                    }
                ),
            )
        except Exception as exc:
            self._log_observability_failure("trace start", exc)
            return ObservabilityTrace()

    def _start_observability_retrieval(
        self,
        observability_trace: ObservabilityTrace,
        *,
        message: str,
    ):
        try:
            return self.observability_tracer.start_retrieval(
                observability_trace,
                original_query=message,
                rewritten_query=None,
                retriever_type=self.settings.retriever_type,
                top_k=self.retrieval_top_k,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                embedding_dimension=self.settings.embedding_dimension,
                similarity_threshold=self.settings.retrieval_min_similarity,
                vector_store_provider=self.settings.vector_store_provider,
                vector_store_name=self._get_vector_store_name(),
                reranker_enabled=self.retrieval_service.reranker_enabled,
                reranker_type=self.retrieval_service.reranker_type,
                reranker_model=self._get_reranker_model(),
            )
        except Exception as exc:
            self._log_observability_failure("retrieval trace start", exc)
            return None

    def _complete_observability_retrieval(
        self,
        observability_trace: ObservabilityTrace,
        *,
        retrieval_observation,
        retrieved_chunks: list[RetrievedChunk],
        latency_ms: int,
    ) -> None:
        try:
            self.observability_tracer.complete_retrieval(
                observability_trace,
                observation=retrieval_observation,
                retrieved_chunks=retrieved_chunks,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            self._log_observability_failure("retrieval trace completion", exc)

    def _start_observability_llm_call(
        self,
        observability_trace: ObservabilityTrace,
        *,
        provider: str | None,
        model: str | None,
    ):
        try:
            return self.observability_tracer.start_llm_call(
                observability_trace,
                provider=provider,
                model=model,
                temperature=None,
                max_tokens=None,
            )
        except Exception as exc:
            self._log_observability_failure("LLM trace start", exc)
            return None

    def _complete_observability_llm_call(
        self,
        observability_trace: ObservabilityTrace,
        *,
        llm_observation,
        provider: str | None,
        model: str | None,
        latency_ms: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
        error_message: str | None = None,
    ) -> None:
        try:
            self.observability_tracer.complete_llm_call(
                observability_trace,
                observation=llm_observation,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost_usd,
                error_message=error_message,
            )
        except Exception as exc:
            self._log_observability_failure("LLM trace completion", exc)

    def _complete_observability_trace(
        self,
        observability_trace: ObservabilityTrace,
        *,
        llm_response: LLMGeneratedResponse,
        latency_ms: int,
        conversation_id: str,
        assistant_message_id: str,
        response_cache_context: ResponseCacheContext,
        routing_context: RoutingExecutionContext,
    ) -> None:
        try:
            self.observability_tracer.complete_chat_request(
                observability_trace,
                final_answer=llm_response.message,
                conversation_id=conversation_id,
                latency_ms=latency_ms,
                llm_provider=llm_response.model_provider,
                llm_model=llm_response.model_name,
                input_tokens=llm_response.token_usage.input_tokens,
                output_tokens=llm_response.token_usage.output_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
                estimated_cost_usd=llm_response.estimated_cost_usd,
                metadata=self._sanitize_metadata_mapping(
                    {
                        "assistant_message_id": assistant_message_id,
                        "final_response_metadata": {
                            "response_cache": self._response_cache_metadata(
                                response_cache_context
                            ),
                            "routing": routing_context.metadata(),
                        },
                    }
                ),
            )
        except Exception as exc:
            self._log_observability_failure("trace completion", exc)

    def _fail_observability_trace(
        self,
        observability_trace: ObservabilityTrace,
        *,
        exc: Exception,
        started_at: float,
    ) -> None:
        try:
            self.observability_tracer.capture_error(
                observability_trace,
                error_message=self._safe_trace_error_message(exc),
                latency_ms=self._elapsed_ms(started_at),
            )
        except Exception as captured_exc:
            self._log_observability_failure("trace failure capture", captured_exc)
        finally:
            self._flush_observability_tracer()

    def _flush_observability_tracer(self) -> None:
        try:
            self.observability_tracer.flush()
        except Exception as exc:
            self._log_observability_failure("tracer flush", exc)

    def _get_vector_store_name(self) -> str | None:
        value = getattr(self.retrieval_service, "vector_store_name", None)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _get_endpoint(self, channel: str) -> str:
        return "/chat"

    def _get_endpoint_name(self, channel: str) -> str:
        return "chat"

    def _build_response_cache_context(self) -> ResponseCacheContext:
        return ResponseCacheContext(
            enabled=self.settings.enable_redis,
            provider="upstash" if self.settings.enable_redis else None,
            exact_enabled=(
                self.settings.enable_redis and self.settings.exact_cache_enabled
            ),
            semantic_enabled=False,
            threshold=None,
        )

    async def _lookup_cached_response(
        self,
        *,
        cache_request: CacheLookupRequest,
        response_cache_context: ResponseCacheContext,
    ) -> LLMGeneratedResponse | None:
        exact_outcome = await self.response_cache.get_exact(cache_request)
        response_cache_context.exact_lookup_latency_ms = exact_outcome.latency_ms
        response_cache_context.lookup_latency_ms += exact_outcome.latency_ms
        response_cache_context.reason = exact_outcome.reason
        validated_exact_entry = self._validated_cache_entry(
            exact_outcome.entry,
            cache_request=cache_request,
        )
        if exact_outcome.hit and validated_exact_entry is not None:
            response_cache_context.hit = True
            response_cache_context.cache_type = "exact"
            response_cache_context.entry_id = validated_exact_entry.entry_id
            response_cache_context.latency_saved_estimate_ms = (
                validated_exact_entry.total_latency_ms
            )
            return self._build_cached_response(validated_exact_entry)
        if exact_outcome.hit and validated_exact_entry is None:
            response_cache_context.reason = (
                self._cache_entry_miss_reason(
                    exact_outcome.entry,
                    cache_request=cache_request,
                )
                or "miss_metadata_mismatch"
            )
        return None

    def _validated_cache_entry(
        self,
        entry: CacheLookupResult | None,
        *,
        cache_request: CacheLookupRequest,
    ) -> CacheLookupResult | None:
        return (
            entry
            if self._cache_entry_miss_reason(entry, cache_request=cache_request) is None
            else None
        )

    def _cache_entry_miss_reason(
        self,
        entry: CacheLookupResult | None,
        *,
        cache_request: CacheLookupRequest,
    ) -> str | None:
        if entry is None:
            return "miss_no_exact_entry"
        if entry.metadata_scope_hash != cache_request.metadata_scope_hash:
            return "miss_metadata_mismatch"
        if entry.prompt_version != cache_request.metadata_scope.prompt_version:
            return "miss_metadata_mismatch"
        if entry.llm_provider != cache_request.metadata_scope.llm_provider:
            return "miss_metadata_mismatch"
        if entry.llm_model != cache_request.metadata_scope.llm_model:
            return "miss_metadata_mismatch"
        if entry.embedding_provider != cache_request.metadata_scope.embedding_provider:
            return "miss_metadata_mismatch"
        if entry.embedding_model != cache_request.metadata_scope.embedding_model:
            return "miss_metadata_mismatch"
        if (
            entry.knowledge_base_version
            != cache_request.metadata_scope.knowledge_base_version
        ):
            return "miss_metadata_mismatch"
        if entry.retriever_type != cache_request.metadata_scope.retriever_type:
            return "miss_metadata_mismatch"
        if entry.top_k != cache_request.metadata_scope.top_k:
            return "miss_metadata_mismatch"
        if (
            entry.query_rewrite_enabled
            != cache_request.metadata_scope.query_rewrite_enabled
        ):
            return "miss_metadata_mismatch"
        if entry.reranker_enabled != cache_request.metadata_scope.reranker_enabled:
            return "miss_metadata_mismatch"
        if (
            entry.retriever_config_hash
            != cache_request.metadata_scope.retriever_config_hash
        ):
            return "miss_metadata_mismatch"
        if entry.expires_at is not None and entry.expires_at <= self._utcnow():
            return "miss_expired"
        return None

    def _build_cache_lookup_request(
        self,
        *,
        message: str,
        prompt_version: str,
        model_config_id: str,
    ) -> CacheLookupRequest:
        normalized = normalize_question(message)
        scope = self._build_cache_scope(
            prompt_version=prompt_version,
            model_config_id=model_config_id,
        )
        request_hash = stable_json_hash(
            {
                "message": normalize_whitespace(message),
                "prompt_version": prompt_version,
                "llm_provider": scope.llm_provider,
                "llm_model": scope.llm_model,
                "knowledge_base_version": scope.knowledge_base_version,
                "retriever_type": scope.retriever_type,
                "top_k": scope.top_k,
                "query_rewrite_enabled": scope.query_rewrite_enabled,
                "reranker_enabled": scope.reranker_enabled,
                "retriever_config_hash": scope.retriever_config_hash,
            }
        )
        return CacheLookupRequest(
            request_hash=request_hash,
            normalized_question=normalized,
            question_hash=stable_hash(normalized),
            metadata_scope_hash=hash_scope(scope),
            metadata_scope=scope,
        )

    def _build_cache_scope(
        self,
        *,
        prompt_version: str,
        model_config_id: str,
    ) -> CacheScope:
        selected_model_config = self.llm_service.get_model_config(model_config_id)
        return CacheScope(
            knowledge_base_version=self.settings.response_cache_knowledge_base_version,
            prompt_version=prompt_version,
            llm_provider=selected_model_config.provider,
            llm_model=selected_model_config.model,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.knowledge_embedding_model,
            retriever_type=self.settings.retriever_type,
            top_k=self.retrieval_top_k,
            query_rewrite_enabled=self.settings.enable_query_rewriting,
            reranker_enabled=self.retrieval_service.reranker_enabled,
            retriever_config_hash=self._build_retriever_config_hash(),
        )

    def _build_retriever_config_hash(self) -> str:
        payload = {
            "retriever_type": self.settings.retriever_type,
            "top_k": self.retrieval_top_k,
            "retrieval_min_similarity": self.settings.retrieval_min_similarity,
            "default_retrieval_config": self.settings.default_retrieval_config,
            "query_rewrite_enabled": self.settings.enable_query_rewriting,
            "query_rewrite_model": self.settings.query_rewrite_model,
            "query_rewrite_prompt_version": self.settings.query_rewrite_prompt_version,
            "query_rewrite_temperature": self.settings.query_rewrite_temperature,
            "reranker_enabled": self.retrieval_service.reranker_enabled,
            "reranker_type": self.retrieval_service.reranker_type,
            "reranker_model": self.settings.reranker_model,
            "reranker_initial_top_k": self.settings.reranker_initial_top_k,
            "reranker_final_top_k": self.settings.reranker_final_top_k,
        }
        return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _is_response_cache_candidate(
        self,
        *,
        message: str,
        conversation_id: str | None,
        channel: str,
        message_metadata: dict[str, object],
    ) -> bool:
        if not self.settings.enable_redis or not self.settings.exact_cache_enabled:
            return False
        if channel != "web_chat":
            return False
        if conversation_id is not None:
            return False
        if message_metadata:
            return False
        if self._is_time_sensitive_query(message):
            return False
        return True

    def _should_store_response_cache(
        self,
        *,
        message: str,
        conversation_id: str | None,
        channel: str,
        message_metadata: dict[str, object],
        retrieved_chunks: list[RetrievedChunk],
        llm_response: LLMGeneratedResponse,
        use_direct_fallback: bool,
    ) -> bool:
        if not self._is_response_cache_candidate(
            message=message,
            conversation_id=conversation_id,
            channel=channel,
            message_metadata=message_metadata,
        ):
            return False
        if use_direct_fallback:
            return False
        if not llm_response.message.strip():
            return False
        if self._looks_like_error_response(llm_response.message):
            return False
        return True

    async def _store_response_cache_entry(
        self,
        *,
        cache_request: CacheLookupRequest | None,
        llm_response: LLMGeneratedResponse,
        retrieved_chunks: list[RetrievedChunk],
        response_cache_context: ResponseCacheContext,
    ) -> None:
        if cache_request is None:
            response_cache_context.write_reason = "write_skipped"
            return
        created_at = self._utcnow()
        ttl_seconds = self.settings.exact_cache_ttl_seconds
        store_outcome = await self.response_cache.store(
            CacheStoreEntry(
                entry_id=cache_request.request_hash,
                normalized_question=cache_request.normalized_question,
                question_hash=cache_request.question_hash,
                question_embedding=None,
                answer_text=llm_response.message,
                source_documents=[
                    {
                        "chunk_id": chunk.metadata.get("chunk_id", chunk.id),
                        "source": chunk.source,
                        "section": chunk.section,
                        "score": chunk.similarity,
                        "content": chunk.content,
                    }
                    for chunk in retrieved_chunks
                ],
                llm_provider=llm_response.model_provider,
                # Cache scope should use the configured model identifier, not the
                # provider-returned versioned runtime name.
                llm_model=cache_request.metadata_scope.llm_model,
                prompt_version=cache_request.metadata_scope.prompt_version,
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.knowledge_embedding_model,
                knowledge_base_version=cache_request.metadata_scope.knowledge_base_version,
                retriever_type=cache_request.metadata_scope.retriever_type,
                top_k=cache_request.metadata_scope.top_k,
                query_rewrite_enabled=cache_request.metadata_scope.query_rewrite_enabled,
                reranker_enabled=cache_request.metadata_scope.reranker_enabled,
                retriever_config_hash=cache_request.metadata_scope.retriever_config_hash,
                metadata_scope_hash=cache_request.metadata_scope_hash,
                retrieval_config=llm_response.retrieval_config,
                created_at=created_at,
                expires_at=created_at + timedelta(seconds=ttl_seconds),
                total_latency_ms=self._estimated_response_total_latency_ms(
                    response_cache_context
                ),
                embedding_latency_ms=response_cache_context.embedding_latency_ms,
                retrieval_latency_ms=response_cache_context.retrieval_latency_ms,
                llm_latency_ms=response_cache_context.llm_latency_ms,
            )
        )
        response_cache_context.write_reason = store_outcome.reason

    def _build_cached_response(
        self,
        entry: CacheLookupResult,
    ) -> LLMGeneratedResponse:
        return LLMGeneratedResponse(
            message=entry.answer_text,
            model=entry.llm_model,
            model_provider=entry.llm_provider,
            model_name=entry.llm_model,
            model_config_id=f"{entry.llm_provider}:{entry.llm_model}",
            prompt_version=entry.prompt_version,
            retrieval_config=entry.retrieval_config,
            latency_ms=0,
            token_usage=TokenUsage(),
            estimated_cost_usd=None,
        )

    def _response_cache_config_metadata(
        self,
        response_cache_context: ResponseCacheContext,
    ) -> dict[str, object]:
        return {
            "response_cache_enabled": response_cache_context.enabled,
            "response_cache_provider": response_cache_context.provider,
            "response_cache_exact_enabled": response_cache_context.exact_enabled,
            "response_cache_semantic_enabled": response_cache_context.semantic_enabled,
        }

    def _response_cache_metadata(
        self,
        response_cache_context: ResponseCacheContext,
    ) -> dict[str, object]:
        return {
            **self._response_cache_config_metadata(response_cache_context),
            "response_cache_hit": response_cache_context.hit,
            "response_cache_type": response_cache_context.cache_type,
            "response_cache_reason": response_cache_context.reason,
            "response_cache_write_reason": response_cache_context.write_reason,
            "response_cache_entry_id": response_cache_context.entry_id,
            "response_cache_distance": response_cache_context.distance,
            "response_cache_threshold": response_cache_context.threshold,
            "response_cache_lookup_latency_ms": response_cache_context.lookup_latency_ms,
            "response_cache_exact_lookup_latency_ms": response_cache_context.exact_lookup_latency_ms,
            "response_cache_semantic_lookup_latency_ms": response_cache_context.semantic_lookup_latency_ms,
            "embedding_latency_ms": response_cache_context.embedding_latency_ms,
            "retrieval_latency_ms": response_cache_context.retrieval_latency_ms,
            "llm_latency_ms": response_cache_context.llm_latency_ms,
            "total_latency_ms": response_cache_context.total_latency_ms,
            "latency_saved_estimate_ms": response_cache_context.latency_saved_estimate_ms,
        }

    def _response_cache_message_metadata(
        self,
        response_cache_context: ResponseCacheContext,
    ) -> dict[str, object]:
        return {
            "hit": response_cache_context.hit,
            "type": response_cache_context.cache_type,
            "reason": response_cache_context.reason,
            "entry_id": response_cache_context.entry_id,
        }

    def _log_response_cache_context(
        self,
        response_cache_context: ResponseCacheContext,
    ) -> None:
        logger.info(
            "Response cache outcome",
            extra={
                "response_cache": self._response_cache_metadata(response_cache_context)
            },
        )

    def _is_time_sensitive_query(self, message: str) -> bool:
        normalized = message.casefold()
        time_sensitive_terms = (
            "latest",
            "current",
            "today",
            "now",
            "recent",
            "news",
            "as of",
            "currently",
            "up to date",
        )
        return any(term in normalized for term in time_sensitive_terms)

    def _looks_like_error_response(self, message: str) -> bool:
        normalized = message.casefold()
        error_markers = (
            "please try again",
            "unable to",
            "error",
            "failed",
        )
        return any(marker in normalized for marker in error_markers)

    def _estimated_response_total_latency_ms(
        self,
        response_cache_context: ResponseCacheContext,
    ) -> int | None:
        measured_latencies = [
            value
            for value in (
                response_cache_context.embedding_latency_ms,
                response_cache_context.retrieval_latency_ms,
                response_cache_context.llm_latency_ms,
            )
            if value is not None
        ]
        if not measured_latencies:
            return None
        return sum(measured_latencies)
