from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Input schema for the user message submitted to the chat endpoint.
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    # Output schema for the assistant reply returned by the backend.
    response: str
