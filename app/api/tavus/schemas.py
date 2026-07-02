from pydantic import BaseModel, Field


class TavusConversationCreateRequest(BaseModel):
    visitor_name: str = Field(default="Website visitor", min_length=1, max_length=255)
    conversation_id: str | None = Field(
        default=None,
        max_length=36,
        description="Optional existing backend conversation UUID to link to the Tavus session.",
    )


class TavusConversationResponse(BaseModel):
    conversation_id: str
    conversation_url: str


class TavusToolRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    tavus_conversation_id: str | None = Field(default=None, min_length=1, max_length=255)
    visitor_name: str | None = Field(default=None, min_length=1, max_length=255)


class TavusToolResponse(BaseModel):
    response: str


class TavusConversationEndRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=255)
