class ChatServiceError(Exception):
    """Raised when the chat workflow cannot complete safely."""


class InvalidChatMessageError(ValueError):
    """Raised when a user message is empty after normalization."""


class ChatService:
    def __init__(self, llm_service) -> None:
        self.llm_service = llm_service

    async def generate_reply(self, message: str) -> str:
        # Trim whitespace here so the API accepts normal user input but still blocks blank prompts.
        normalized_message = message.strip()
        if not normalized_message:
            raise InvalidChatMessageError("Chat message cannot be empty.")

        return await self.llm_service.generate_response(normalized_message)
