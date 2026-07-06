from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.domain.tracing import (
    ChatTraceCreate,
    ChatTraceRead,
    ChatTraceStepCreate,
    ChatTraceStepRead,
    ChatTraceUpdate,
    TraceStatus,
    TraceStepType,
)
from app.repositories.models import ChatTrace, ChatTraceStep
from app.repositories.tracing_repository import TraceRepository, TraceRepositoryError


class TraceServiceError(Exception):
    """Raised when trace persistence fails."""


class TraceService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def start_trace(
        self,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        status: TraceStatus = TraceStatus.STARTED,
        error_message: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        prompt_version: str | None = None,
        retriever_type: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: Decimal | float | None = None,
        latency_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatTraceRead:
        payload = ChatTraceCreate(
            conversation_id=conversation_id,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            input_text=input_text,
            output_text=output_text,
            status=status,
            error_message=error_message,
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_version=prompt_version,
            retriever_type=retriever_type,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
            metadata=self._sanitize_mapping(metadata),
        )
        record = self._write(lambda repository: repository.create_trace(payload))
        return self._build_trace_read(record)

    def add_step(
        self,
        *,
        trace_id: str,
        step_type: TraceStepType,
        status: TraceStatus = TraceStatus.SUCCESS,
        step_index: int | None = None,
        name: str | None = None,
        input_payload: Mapping[str, Any] | None = None,
        output_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> ChatTraceStepRead:
        payload = ChatTraceStepCreate(
            trace_id=trace_id,
            step_index=step_index,
            step_type=step_type,
            status=status,
            name=name,
            input_payload=self._sanitize_optional_mapping(input_payload),
            output_payload=self._sanitize_optional_mapping(output_payload),
            metadata=self._sanitize_mapping(metadata),
            latency_ms=latency_ms,
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at,
        )
        record = self._write(lambda repository: repository.create_step(payload))
        return self._build_step_read(record)

    def complete_trace(
        self,
        trace_id: str,
        *,
        output_text: str | None = None,
        status: TraceStatus = TraceStatus.SUCCESS,
        error_message: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        prompt_version: str | None = None,
        retriever_type: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: Decimal | float | None = None,
        latency_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatTraceRead:
        payload = ChatTraceUpdate(
            output_text=output_text,
            status=status,
            error_message=error_message,
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_version=prompt_version,
            retriever_type=retriever_type,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
            metadata=self._sanitize_mapping(metadata) if metadata is not None else None,
        )
        record = self._write(lambda repository: repository.update_trace(trace_id, payload))
        return self._build_trace_read(record)

    def fail_trace(
        self,
        trace_id: str,
        *,
        error_message: str | None,
        latency_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatTraceRead:
        return self.complete_trace(
            trace_id,
            status=TraceStatus.ERROR,
            error_message=error_message,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def get_trace(self, trace_id: str) -> ChatTraceRead | None:
        with self._session_factory() as session:
            repository = TraceRepository(session)
            try:
                trace = repository.get_trace(trace_id)
                if trace is None:
                    return None
                steps = repository.list_steps(trace_id)
            except TraceRepositoryError as exc:
                raise TraceServiceError() from exc
        return self._build_trace_read(trace, steps=steps)

    def _write(self, operation):
        with self._session_factory() as session:
            repository = TraceRepository(session)
            try:
                return operation(repository)
            except TraceRepositoryError as exc:
                raise TraceServiceError() from exc

    def _build_trace_read(
        self,
        record: ChatTrace,
        *,
        steps: Sequence[ChatTraceStep] | None = None,
    ) -> ChatTraceRead:
        return ChatTraceRead(
            id=record.id,
            conversation_id=record.conversation_id,
            user_id=record.user_id,
            request_id=record.request_id,
            session_id=record.session_id,
            input_text=record.input_text,
            output_text=record.output_text,
            status=TraceStatus(record.status),
            error_message=record.error_message,
            llm_provider=record.llm_provider,
            llm_model=record.llm_model,
            prompt_version=record.prompt_version,
            retriever_type=record.retriever_type,
            embedding_provider=record.embedding_provider,
            embedding_model=record.embedding_model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            estimated_cost_usd=record.estimated_cost_usd,
            latency_ms=record.latency_ms,
            metadata=dict(record.trace_metadata or {}),
            created_at=record.created_at,
            updated_at=record.updated_at,
            steps=[self._build_step_read(step) for step in steps or ()],
        )

    def _build_step_read(self, record: ChatTraceStep) -> ChatTraceStepRead:
        return ChatTraceStepRead(
            id=record.id,
            trace_id=record.trace_id,
            step_index=record.step_index,
            step_type=TraceStepType(record.step_type),
            status=TraceStatus(record.status),
            name=record.name,
            input_payload=dict(record.input_payload or {}) if record.input_payload is not None else None,
            output_payload=dict(record.output_payload or {}) if record.output_payload is not None else None,
            metadata=dict(record.step_metadata or {}),
            latency_ms=record.latency_ms,
            error_message=record.error_message,
            started_at=record.started_at,
            completed_at=record.completed_at,
            created_at=record.created_at,
        )

    def _sanitize_mapping(self, value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        return {str(key): self._sanitize_value(item) for key, item in value.items()}

    def _sanitize_optional_mapping(
        self,
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return self._sanitize_mapping(value)

    def _sanitize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, datetime):
            normalized = value
            if normalized.tzinfo is None:
                normalized = normalized.replace(tzinfo=UTC)
            return normalized.isoformat()
        if isinstance(value, Mapping):
            return {str(key): self._sanitize_value(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_value(item) for item in value]
        return str(value)
