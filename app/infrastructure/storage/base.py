from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class StorageError(Exception):
    """Raised when object storage operations fail."""


@dataclass(frozen=True, slots=True)
class StoredFile:
    provider: str
    bucket: str
    path: str
    size_bytes: int
    content_type: str | None = None


class KnowledgeFileStorage(Protocol):
    def upload_file(
        self,
        *,
        file_bytes: bytes,
        storage_path: str,
        content_type: str | None = None,
    ) -> StoredFile:
        ...

    def download_file(
        self,
        *,
        storage_path: str,
    ) -> bytes:
        ...

    def delete_file(
        self,
        *,
        storage_path: str,
    ) -> None:
        ...
