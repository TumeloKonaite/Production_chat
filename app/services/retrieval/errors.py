class EmbeddingConfigurationError(Exception):
    """Raised when the configured embedding provider cannot be used safely."""


class VectorIndexConfigurationError(Exception):
    """Raised when the stored vector index is incompatible with the active embedding config."""
