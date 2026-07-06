from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import logging
import random
from typing import Protocol, TYPE_CHECKING

from app.config import Settings
from app.infrastructure.observability.langfuse_client import (
    LangfuseClient,
    ObservationHandle,
    RootObservationHandle,
)

if TYPE_CHECKING:
    from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
CONTENT_PREVIEW_LIMIT = 240


@dataclass(slots=True)
class ObservabilityTrace:
    root_observation: RootObservationHandle | None = None

    @property
    def is_active(self) -> bool:
        return self.root_observation is not None


class ObservabilityTracer(Protocol):
    def start_chat_request(
        self,
        *,
        question: str,
        conversation_id: str,
        session_id: str | None,
        user_id: str | None,
        endpoint: str,
        endpoint_name: str,
        channel: str,
        llm_provider: str | None,
        llm_model: str | None,
    ) -> ObservabilityTrace:
        ...

    def start_retrieval(
        self,
        trace: ObservabilityTrace,
        *,
        original_query: str,
        rewritten_query: str | None,
        retriever_type: str,
        top_k: int,
        embedding_provider: str,
        embedding_model: str,
        vector_store: str | None,
    ) -> ObservationHandle | None:
        ...

    def complete_retrieval(
        self,
        trace: ObservabilityTrace,
        *,
        observation: ObservationHandle | None,
        retrieved_chunks: Sequence[RetrievedChunk],
        latency_ms: int,
    ) -> None:
        ...

    def start_llm_call(
        self,
        trace: ObservabilityTrace,
        *,
        provider: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> ObservationHandle | None:
        ...

    def complete_llm_call(
        self,
        trace: ObservabilityTrace,
        *,
        observation: ObservationHandle | None,
        provider: str | None,
        model: str | None,
        latency_ms: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
        error_message: str | None = None,
    ) -> None:
        ...

    def complete_chat_request(
        self,
        trace: ObservabilityTrace,
        *,
        final_answer: str,
        conversation_id: str,
        latency_ms: int,
        llm_provider: str | None,
        llm_model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        ...

    def capture_error(
        self,
        trace: ObservabilityTrace,
        *,
        error_message: str,
        latency_ms: int,
    ) -> None:
        ...

    def flush(self) -> None:
        ...


class NoOpTracer:
    def start_chat_request(self, **kwargs: object) -> ObservabilityTrace:
        return ObservabilityTrace()

    def start_retrieval(self, trace: ObservabilityTrace, **kwargs: object) -> ObservationHandle | None:
        return None

    def complete_retrieval(self, trace: ObservabilityTrace, **kwargs: object) -> None:
        return None

    def start_llm_call(self, trace: ObservabilityTrace, **kwargs: object) -> ObservationHandle | None:
        return None

    def complete_llm_call(self, trace: ObservabilityTrace, **kwargs: object) -> None:
        return None

    def complete_chat_request(self, trace: ObservabilityTrace, **kwargs: object) -> None:
        return None

    def capture_error(self, trace: ObservabilityTrace, **kwargs: object) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:
    def __init__(
        self,
        *,
        client: LangfuseClient,
        environment: str,
        release: str | None,
        sample_rate: float,
        random_value_factory: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._environment = environment
        self._release = release
        self._sample_rate = sample_rate
        self._random_value_factory = random_value_factory or random.random

    def start_chat_request(
        self,
        *,
        question: str,
        conversation_id: str,
        session_id: str | None,
        user_id: str | None,
        endpoint: str,
        endpoint_name: str,
        channel: str,
        llm_provider: str | None,
        llm_model: str | None,
    ) -> ObservabilityTrace:
        if not self._should_sample():
            return ObservabilityTrace()

        root_observation = self._client.start_root_observation(
            name="chat_request",
            input_payload={
                "question": question,
                "conversation_id": conversation_id,
            },
            metadata={
                "endpoint": endpoint,
                "endpoint_name": endpoint_name,
                "channel": channel,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            },
            user_id=user_id,
            session_id=session_id or conversation_id,
            environment=self._environment,
            release=self._release,
        )
        return ObservabilityTrace(root_observation=root_observation)

    def start_retrieval(
        self,
        trace: ObservabilityTrace,
        *,
        original_query: str,
        rewritten_query: str | None,
        retriever_type: str,
        top_k: int,
        embedding_provider: str,
        embedding_model: str,
        vector_store: str | None,
    ) -> ObservationHandle | None:
        if not trace.is_active:
            return None

        return self._client.start_observation(
            name="retrieval",
            as_type="retriever",
            input_payload={
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "retriever_type": retriever_type,
                "top_k": top_k,
            },
            metadata={
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "vector_store": vector_store,
            },
            version=self._release,
        )

    def complete_retrieval(
        self,
        trace: ObservabilityTrace,
        *,
        observation: ObservationHandle | None,
        retrieved_chunks: Sequence[RetrievedChunk],
        latency_ms: int,
    ) -> None:
        if not trace.is_active or observation is None:
            return

        try:
            observation.update(
                output={
                    "latency_ms": latency_ms,
                    "retrieved_sources": [chunk.source for chunk in retrieved_chunks],
                    "chunk_ids": [self._chunk_id(chunk) for chunk in retrieved_chunks],
                    "results": [
                        {
                            "rank": index,
                            "chunk_id": self._chunk_id(chunk),
                            "source_name": chunk.source,
                            "score": chunk.similarity,
                            "content_preview": self._truncate_text(chunk.content),
                        }
                        for index, chunk in enumerate(retrieved_chunks, start=1)
                    ],
                }
            )
        finally:
            observation.end()

    def start_llm_call(
        self,
        trace: ObservabilityTrace,
        *,
        provider: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> ObservationHandle | None:
        if not trace.is_active:
            return None

        model_parameters: dict[str, object] = {}
        if temperature is not None:
            model_parameters["temperature"] = temperature
        if max_tokens is not None:
            model_parameters["max_tokens"] = max_tokens

        return self._client.start_observation(
            name="llm_call",
            as_type="generation",
            input_payload=None,
            metadata={"provider": provider},
            version=self._release,
            model=model,
            model_parameters=model_parameters or None,
        )

    def complete_llm_call(
        self,
        trace: ObservabilityTrace,
        *,
        observation: ObservationHandle | None,
        provider: str | None,
        model: str | None,
        latency_ms: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
        error_message: str | None = None,
    ) -> None:
        if not trace.is_active or observation is None:
            return

        try:
            update_payload: dict[str, object] = {
                "output": {
                    "provider": provider,
                    "model": model,
                    "latency_ms": latency_ms,
                    "error": error_message,
                },
                "usage_details": (
                    {
                        "input": input_tokens or 0,
                        "output": output_tokens or 0,
                        "total": total_tokens or 0,
                    }
                    if any(value is not None for value in (input_tokens, output_tokens, total_tokens))
                    else None
                ),
            }
            if estimated_cost_usd is not None:
                update_payload["cost_details"] = {"total": estimated_cost_usd}

            observation.update(
                **update_payload,
            )
        finally:
            observation.end()

    def complete_chat_request(
        self,
        trace: ObservabilityTrace,
        *,
        final_answer: str,
        conversation_id: str,
        latency_ms: int,
        llm_provider: str | None,
        llm_model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        if not trace.is_active or trace.root_observation is None:
            return

        try:
            trace.root_observation.update(
                output={
                    "final_answer": final_answer,
                    "conversation_id": conversation_id,
                    "latency_ms": latency_ms,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )
        finally:
            trace.root_observation.close()

    def capture_error(
        self,
        trace: ObservabilityTrace,
        *,
        error_message: str,
        latency_ms: int,
    ) -> None:
        if not trace.is_active or trace.root_observation is None:
            return

        try:
            trace.root_observation.update(
                output={
                    "error": error_message,
                    "latency_ms": latency_ms,
                }
            )
        finally:
            trace.root_observation.close()

    def flush(self) -> None:
        self._client.flush()

    def _should_sample(self) -> bool:
        if self._sample_rate <= 0.0:
            return False
        if self._sample_rate >= 1.0:
            return True
        return self._random_value_factory() <= self._sample_rate

    def _chunk_id(self, chunk: RetrievedChunk) -> str:
        value = chunk.metadata.get("chunk_id")
        if isinstance(value, str) and value.strip():
            return value
        return chunk.id

    def _truncate_text(self, value: str) -> str:
        normalized = value.strip()
        if len(normalized) <= CONTENT_PREVIEW_LIMIT:
            return normalized
        return normalized[:CONTENT_PREVIEW_LIMIT]


def _build_tracer(settings: Settings) -> ObservabilityTracer:
    if not settings.enable_langfuse_observability:
        return NoOpTracer()

    try:
        client = LangfuseClient(
            public_key=settings.langfuse_public_key or "",
            secret_key=settings.langfuse_secret_key or "",
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
            release=settings.langfuse_release,
            sample_rate=settings.langfuse_sample_rate,
        )
    except Exception:
        logger.warning(
            "Langfuse initialization failed. Continuing with no-op observability.",
            exc_info=True,
        )
        return NoOpTracer()

    return LangfuseTracer(
        client=client,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        sample_rate=settings.langfuse_sample_rate,
    )


@lru_cache
def _get_cached_tracer(settings: Settings) -> ObservabilityTracer:
    return _build_tracer(settings)


def get_tracer(
    settings: Settings,
    *,
    client: LangfuseClient | None = None,
    random_value_factory: Callable[[], float] | None = None,
) -> ObservabilityTracer:
    if client is None and random_value_factory is None:
        return _get_cached_tracer(settings)

    if not settings.enable_langfuse_observability:
        return NoOpTracer()
    if client is None:
        client = LangfuseClient(
            public_key=settings.langfuse_public_key or "",
            secret_key=settings.langfuse_secret_key or "",
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
            release=settings.langfuse_release,
            sample_rate=settings.langfuse_sample_rate,
        )
    return LangfuseTracer(
        client=client,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        sample_rate=settings.langfuse_sample_rate,
        random_value_factory=random_value_factory,
    )
