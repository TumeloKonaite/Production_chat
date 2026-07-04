class EmbeddingConfigurationError(Exception):
    """Raised when the configured embedding provider cannot be used safely."""


class VectorIndexConfigurationError(Exception):
    """Raised when the stored vector index is incompatible with the active embedding config."""


class UnsupportedRetrieverError(Exception):
    """Raised when the requested retriever type cannot be constructed."""


class UnsupportedRerankerError(Exception):
    """Raised when the requested reranker type cannot be constructed."""


class InvalidRerankerResultError(Exception):
    """Raised when a reranker returns an invalid chunk ordering."""
