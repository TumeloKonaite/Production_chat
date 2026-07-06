from __future__ import annotations

from enum import StrEnum


class TraceStatus(StrEnum):
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class TraceStepType(StrEnum):
    REQUEST_RECEIVED = "request_received"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    PROMPT_BUILT = "prompt_built"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    RESPONSE_GENERATED = "response_generated"
    ERROR = "error"


TRACE_STATUS_VALUES = tuple(status.value for status in TraceStatus)
TRACE_STEP_TYPE_VALUES = tuple(step_type.value for step_type in TraceStepType)
