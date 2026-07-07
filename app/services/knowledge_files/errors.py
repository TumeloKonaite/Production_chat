class KnowledgeFileUploadError(Exception):
    """Raised when a knowledge file upload cannot complete."""


class KnowledgeFileValidationError(ValueError):
    """Raised when a knowledge file upload request is invalid."""
