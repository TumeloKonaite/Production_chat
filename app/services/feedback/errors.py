class MessageFeedbackError(Exception):
    """Raised when the feedback workflow cannot complete safely."""


class MessageFeedbackTargetNotFoundError(MessageFeedbackError):
    """Raised when the target message for feedback does not exist."""


class InvalidFeedbackTargetError(MessageFeedbackError):
    """Raised when feedback is submitted for an unsupported message type."""


class MessageFeedbackPersistenceError(MessageFeedbackError):
    """Raised when feedback persistence fails."""
