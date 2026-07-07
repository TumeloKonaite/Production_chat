from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Input schema for the user message submitted to the chat endpoint.
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=36)
    prompt_version: str | None = Field(
        default=None,
        min_length=1,
        description="Prompt template version to use for this response.",
    )
    model_config_id: str | None = Field(
        default=None,
        min_length=1,
        description="Configured model ID to use for this response, for example openai:gpt-4.1-mini.",
    )


class TokenUsageResponse(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Output schema for the assistant reply returned by the backend.
    conversation_id: str
    message_id: str
    message: str
    model: str
    model_provider: str
    model_name: str
    model_config_id: str
    prompt_version: str
    retrieval_config: str
    latency_ms: int | None = None
    token_usage: TokenUsageResponse
    estimated_cost_usd: float | None = None
    response_cache_hit: bool
    response_cache_type: Literal["exact", "semantic"] | None = None
    response_cache_reason: str
    response_cache_distance: float | None = None


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("comment", mode="before")
    @classmethod
    def _normalize_comment(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = None
    created_at: datetime
    updated_at: datetime
