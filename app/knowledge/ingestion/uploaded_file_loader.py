from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePath

from app.infrastructure.storage import KnowledgeFileStorage, StorageError
from app.knowledge.ingestion.errors import (
    KnowledgeIngestionServiceError,
    KnowledgeIngestionValidationError,
)
from app.knowledge.ingestion.loader import SourceDocument
from app.repositories.models import KnowledgeFile

_SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})


class UploadedKnowledgeFileLoader:
    def __init__(self, *, storage: KnowledgeFileStorage) -> None:
        self._storage = storage

    def load_file(self, knowledge_file: KnowledgeFile) -> SourceDocument:
        extension = PurePath(knowledge_file.original_filename).suffix.casefold()
        if extension not in _SUPPORTED_EXTENSIONS:
            raise KnowledgeIngestionValidationError("Only .md and .txt files are supported.")

        try:
            file_bytes = self._storage.download_file(storage_path=knowledge_file.storage_path)
        except StorageError as exc:
            if _is_missing_storage_object(exc):
                raise KnowledgeIngestionServiceError("Storage object not found.") from exc
            raise KnowledgeIngestionServiceError("Unable to download uploaded knowledge file.") from exc

        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeIngestionValidationError(
                "Uploaded knowledge files must be valid UTF-8 text."
            ) from exc

        updated_at = knowledge_file.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        return SourceDocument(
            source=knowledge_file.storage_path,
            text=text,
            updated_at=updated_at,
            chunk_source_type="uploaded_file",
            metadata={
                "source_type": "uploaded_file",
                "file_id": knowledge_file.id,
                "original_filename": knowledge_file.original_filename,
                "storage_provider": knowledge_file.storage_provider,
                "storage_bucket": knowledge_file.storage_bucket,
                "storage_path": knowledge_file.storage_path,
                "content_type": "markdown" if extension == ".md" else "text",
            },
        )


def _is_missing_storage_object(error: StorageError) -> bool:
    cause = error.__cause__
    if isinstance(cause, FileNotFoundError):
        return True

    response = getattr(cause, "response", None)
    if not isinstance(response, dict):
        return False

    code = str(response.get("Error", {}).get("Code", "")).casefold()
    return code in {"nosuchkey", "404", "notfound"}
