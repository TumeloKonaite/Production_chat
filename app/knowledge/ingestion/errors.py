class KnowledgeIngestionServiceError(Exception):
    """Raised when knowledge ingestion cannot complete."""


class KnowledgeIngestionValidationError(KnowledgeIngestionServiceError):
    """Raised when a knowledge ingestion request is invalid."""


class KnowledgeIngestionNotFoundError(KnowledgeIngestionServiceError):
    """Raised when the requested knowledge source does not exist."""


class KnowledgeIngestionConflictError(KnowledgeIngestionServiceError):
    """Raised when the requested knowledge source cannot be ingested right now."""


class KnowledgeIngestionGoneError(KnowledgeIngestionServiceError):
    """Raised when the requested knowledge source has been deleted."""
