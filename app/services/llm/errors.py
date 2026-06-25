class LLMConfigurationError(Exception):
    """Raised when LLM configuration is missing or invalid."""


class LLMServiceError(Exception):
    """Raised when the LLM provider request fails safely."""
