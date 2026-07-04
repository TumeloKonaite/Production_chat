class LLMConfigurationError(Exception):
    """Raised when LLM configuration is missing or invalid."""


class LLMServiceError(Exception):
    """Raised when the LLM provider request fails safely."""

    def __init__(self, message: str = "LLM provider request failed.") -> None:
        super().__init__(message)
