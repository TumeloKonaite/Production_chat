class TavusServiceError(Exception):
    """Raised when the Tavus integration fails to complete an upstream request."""


class TavusConfigurationError(TavusServiceError):
    """Raised when required Tavus settings are missing."""
