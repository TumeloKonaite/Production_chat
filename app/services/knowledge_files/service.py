from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import logging
from pathlib import PurePath
import re
import uuid

from sqlalchemy.orm import Session

from app.config import Settings
from app.infrastructure.storage import KnowledgeFileStorage, StorageError
from app.repositories.knowledge_file_repository import (
    KnowledgeFileRepository,
    KnowledgeFileRepositoryError,
)
from app.repositories.models import KnowledgeFile
from app.services.knowledge_files.errors import (
    KnowledgeFileUploadError,
    KnowledgeFileValidationError,
)

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = frozenset({".md", ".txt"})
_CONTENT_TYPES_BY_EXTENSION = {
    ".md": frozenset({"text/markdown", "text/plain", "application/octet-stream"}),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
}
_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class KnowledgeFileUploadService:
    def __init__(
        self,
        *,
        settings: Settings,
        storage: KnowledgeFileStorage,
        repository_factory: Callable[[Session], KnowledgeFileRepository] = KnowledgeFileRepository,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._repository_factory = repository_factory

    def upload_file(
        self,
        session: Session,
        *,
        filename: str | None,
        content_type: str | None,
        file_bytes: bytes,
    ) -> KnowledgeFile:
        safe_filename = _sanitize_filename(filename)
        extension = PurePath(safe_filename).suffix.casefold()
        self._validate_upload(
            filename=safe_filename,
            extension=extension,
            content_type=content_type,
            file_size_bytes=len(file_bytes),
        )

        knowledge_file_id = str(uuid.uuid4())
        storage_path = f"{knowledge_file_id}/{safe_filename}"

        try:
            stored_file = self._storage.upload_file(
                file_bytes=file_bytes,
                storage_path=storage_path,
                content_type=content_type,
            )
        except StorageError as exc:
            raise KnowledgeFileUploadError("Unable to upload knowledge file.") from exc

        record = KnowledgeFile(
            id=knowledge_file_id,
            original_filename=safe_filename,
            content_type=content_type,
            file_size_bytes=len(file_bytes),
            storage_provider=stored_file.provider,
            storage_bucket=stored_file.bucket,
            storage_path=stored_file.path,
            checksum=sha256(file_bytes).hexdigest(),
            status="uploaded",
        )

        repository = self._repository_factory(session)
        try:
            return repository.create(record)
        except KnowledgeFileRepositoryError as exc:
            self._cleanup_orphaned_object(storage_path=stored_file.path)
            raise KnowledgeFileUploadError(
                "Unable to persist uploaded knowledge file metadata."
            ) from exc

    def download_file_bytes(self, *, storage_path: str) -> bytes:
        try:
            return self._storage.download_file(storage_path=storage_path)
        except StorageError as exc:
            raise KnowledgeFileUploadError("Unable to download knowledge file.") from exc

    def _validate_upload(
        self,
        *,
        filename: str,
        extension: str,
        content_type: str | None,
        file_size_bytes: int,
    ) -> None:
        if not filename:
            raise KnowledgeFileValidationError("A filename is required.")
        if extension not in _ALLOWED_EXTENSIONS:
            raise KnowledgeFileValidationError("Only .md and .txt files are supported.")
        if file_size_bytes == 0:
            raise KnowledgeFileValidationError("Knowledge files cannot be empty.")
        if file_size_bytes > self._settings.knowledge_upload_max_bytes:
            raise KnowledgeFileValidationError("Knowledge file exceeds the maximum upload size.")
        if content_type:
            allowed_content_types = _CONTENT_TYPES_BY_EXTENSION.get(extension, frozenset())
            if content_type not in allowed_content_types:
                raise KnowledgeFileValidationError("Knowledge file content type is not supported.")

    def _cleanup_orphaned_object(self, *, storage_path: str) -> None:
        try:
            self._storage.delete_file(storage_path=storage_path)
        except StorageError:
            logger.exception(
                "Failed to clean up orphaned knowledge file object.",
                extra={"storage_path": storage_path},
            )


def _sanitize_filename(filename: str | None) -> str:
    if filename is None:
        raise KnowledgeFileValidationError("A file is required.")

    candidate = PurePath(filename).name.strip()
    if not candidate or candidate in {".", ".."}:
        raise KnowledgeFileValidationError("A valid filename is required.")

    sanitized = _FILENAME_SANITIZE_PATTERN.sub("-", candidate).strip(" .-_")
    if not sanitized:
        raise KnowledgeFileValidationError("A valid filename is required.")

    return sanitized
