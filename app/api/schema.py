from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Input schema for the user message submitted to the chat endpoint.
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=36)
    prompt_version: str | None = Field(
        default=None,
        min_length=1,
        description="Prompt template version to use for this response.",
    )


class TokenUsageResponse(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    # Output schema for the assistant reply returned by the backend.
    conversation_id: str
    message: str
    model: str
    prompt_version: str
    latency_ms: int | None = None
    token_usage: TokenUsageResponse
