from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.tracing.schemas import ChatTraceCreate, ChatTraceStepCreate, ChatTraceUpdate
from app.repositories.models import ChatTrace, ChatTraceStep, utcnow


class TraceRepositoryError(Exception):
    """Raised when trace persistence fails."""


class TraceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_trace(self, payload: ChatTraceCreate) -> ChatTrace:
        record = ChatTrace(
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
            request_id=payload.request_id,
            session_id=payload.session_id,
            input_text=payload.input_text,
            output_text=payload.output_text,
            status=payload.status,
            error_message=payload.error_message,
            llm_provider=payload.llm_provider,
            llm_model=payload.llm_model,
            prompt_version=payload.prompt_version,
            retriever_type=payload.retriever_type,
            embedding_provider=payload.embedding_provider,
            embedding_model=payload.embedding_model,
            input_tokens=payload.input_tokens,
            output_tokens=payload.output_tokens,
            total_tokens=payload.total_tokens,
            estimated_cost_usd=payload.estimated_cost_usd,
            latency_ms=payload.latency_ms,
            trace_metadata=dict(payload.metadata),
        )
        self._session.add(record)
        return self._commit_and_refresh(record)

    def get_trace(self, trace_id: str) -> ChatTrace | None:
        statement = select(ChatTrace).where(ChatTrace.id == trace_id)
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise TraceRepositoryError() from exc

    def update_trace(self, trace_id: str, payload: ChatTraceUpdate) -> ChatTrace:
        trace = self.get_trace(trace_id)
        if trace is None:
            raise TraceRepositoryError(f"Trace not found: {trace_id}")

        if "conversation_id" in payload.model_fields_set:
            trace.conversation_id = payload.conversation_id
        if "user_id" in payload.model_fields_set:
            trace.user_id = payload.user_id
        if "request_id" in payload.model_fields_set:
            trace.request_id = payload.request_id
        if "session_id" in payload.model_fields_set:
            trace.session_id = payload.session_id
        if "input_text" in payload.model_fields_set:
            trace.input_text = payload.input_text
        if "output_text" in payload.model_fields_set:
            trace.output_text = payload.output_text
        if "status" in payload.model_fields_set and payload.status is not None:
            trace.status = payload.status
        if "error_message" in payload.model_fields_set:
            trace.error_message = payload.error_message
        if "llm_provider" in payload.model_fields_set:
            trace.llm_provider = payload.llm_provider
        if "llm_model" in payload.model_fields_set:
            trace.llm_model = payload.llm_model
        if "prompt_version" in payload.model_fields_set:
            trace.prompt_version = payload.prompt_version
        if "retriever_type" in payload.model_fields_set:
            trace.retriever_type = payload.retriever_type
        if "embedding_provider" in payload.model_fields_set:
            trace.embedding_provider = payload.embedding_provider
        if "embedding_model" in payload.model_fields_set:
            trace.embedding_model = payload.embedding_model
        if "input_tokens" in payload.model_fields_set:
            trace.input_tokens = payload.input_tokens
        if "output_tokens" in payload.model_fields_set:
            trace.output_tokens = payload.output_tokens
        if "total_tokens" in payload.model_fields_set:
            trace.total_tokens = payload.total_tokens
        if "estimated_cost_usd" in payload.model_fields_set:
            trace.estimated_cost_usd = payload.estimated_cost_usd
        if "latency_ms" in payload.model_fields_set:
            trace.latency_ms = payload.latency_ms
        if "metadata" in payload.model_fields_set and payload.metadata is not None:
            trace.trace_metadata = dict(payload.metadata)

        trace.updated_at = utcnow()
        return self._commit_and_refresh(trace)

    def create_step(self, payload: ChatTraceStepCreate) -> ChatTraceStep:
        next_step_index = payload.step_index or self._next_step_index(payload.trace_id)
        record = ChatTraceStep(
            trace_id=payload.trace_id,
            step_index=next_step_index,
            step_type=payload.step_type,
            status=payload.status,
            name=payload.name,
            input_payload=payload.input_payload,
            output_payload=payload.output_payload,
            step_metadata=dict(payload.metadata),
            latency_ms=payload.latency_ms,
            error_message=payload.error_message,
            started_at=payload.started_at,
            completed_at=payload.completed_at,
        )
        self._session.add(record)
        return self._commit_and_refresh(record)

    def list_steps(self, trace_id: str) -> Sequence[ChatTraceStep]:
        statement = (
            select(ChatTraceStep)
            .where(ChatTraceStep.trace_id == trace_id)
            .order_by(ChatTraceStep.step_index.asc(), ChatTraceStep.created_at.asc(), ChatTraceStep.id.asc())
        )
        try:
            return list(self._session.scalars(statement))
        except SQLAlchemyError as exc:
            raise TraceRepositoryError() from exc

    def _next_step_index(self, trace_id: str) -> int:
        statement = select(func.max(ChatTraceStep.step_index)).where(ChatTraceStep.trace_id == trace_id)
        try:
            current_index = self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise TraceRepositoryError() from exc
        return (current_index or 0) + 1

    def _commit_and_refresh(self, instance: ChatTrace | ChatTraceStep) -> ChatTrace | ChatTraceStep:
        try:
            self._session.commit()
            self._session.refresh(instance)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise TraceRepositoryError() from exc
        return instance
