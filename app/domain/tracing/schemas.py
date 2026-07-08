from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.tracing.enums import TraceStatus, TraceStepType


class ChatTraceStepCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    trace_id: str = Field(..., min_length=1, max_length=36)
    step_index: int | None = Field(default=None, ge=1)
    step_type: TraceStepType
    status: TraceStatus = TraceStatus.SUCCESS
    name: str | None = Field(default=None, max_length=255)
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = Field(default=None, ge=0)
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ChatTraceStepRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    trace_id: str
    step_index: int
    step_type: TraceStepType
    status: TraceStatus
    name: str | None = None
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


class ChatTraceCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = Field(default=None, max_length=255)
    request_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)
    input_text: str | None = None
    output_text: str | None = None
    status: TraceStatus = TraceStatus.STARTED
    error_message: str | None = None
    llm_provider: str | None = Field(default=None, max_length=50)
    llm_model: str | None = Field(default=None, max_length=255)
    observability_provider: str | None = Field(default=None, max_length=50)
    external_trace_id: str | None = Field(default=None, max_length=255)
    prompt_version: str | None = Field(default=None, max_length=50)
    retriever_type: str | None = Field(default=None, max_length=100)
    embedding_provider: str | None = Field(default=None, max_length=50)
    embedding_model: str | None = Field(default=None, max_length=255)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | float | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTraceUpdate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = Field(default=None, max_length=255)
    request_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)
    input_text: str | None = None
    output_text: str | None = None
    status: TraceStatus | None = None
    error_message: str | None = None
    llm_provider: str | None = Field(default=None, max_length=50)
    llm_model: str | None = Field(default=None, max_length=255)
    observability_provider: str | None = Field(default=None, max_length=50)
    external_trace_id: str | None = Field(default=None, max_length=255)
    prompt_version: str | None = Field(default=None, max_length=50)
    retriever_type: str | None = Field(default=None, max_length=100)
    embedding_provider: str | None = Field(default=None, max_length=50)
    embedding_model: str | None = Field(default=None, max_length=255)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | float | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None


class ChatTraceRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    conversation_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    status: TraceStatus
    error_message: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    observability_provider: str | None = None
    external_trace_id: str | None = None
    prompt_version: str | None = None
    retriever_type: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    steps: list[ChatTraceStepRead] = Field(default_factory=list)
