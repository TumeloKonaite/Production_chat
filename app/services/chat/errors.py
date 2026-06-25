class ChatServiceError(Exception):
    """Raised when the chat workflow cannot complete safely."""


class InvalidChatMessageError(ValueError):
    """Raised when a user message is empty after normalization."""


class InvalidConversationIdError(ValueError):
    """Raised when a provided conversation id is not a valid UUID."""


class ConversationNotFoundError(ChatServiceError):
    """Raised when a requested conversation does not exist."""


class ChatPersistenceError(ChatServiceError):
    """Raised when chat persistence fails."""
