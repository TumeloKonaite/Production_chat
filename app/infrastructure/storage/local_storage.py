from __future__ import annotations

from pathlib import Path

from app.infrastructure.storage.base import KnowledgeFileStorage, StorageError, StoredFile


class LocalKnowledgeFileStorage(KnowledgeFileStorage):
    def __init__(self, *, root_dir: Path, bucket: str) -> None:
        self._root_dir = root_dir
        self._bucket = bucket

    def upload_file(
        self,
        *,
        file_bytes: bytes,
        storage_path: str,
        content_type: str | None = None,
    ) -> StoredFile:
        target_path = self._resolve_storage_path(storage_path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(file_bytes)
        except OSError as exc:
            raise StorageError("Unable to write knowledge file to local storage.") from exc

        return StoredFile(
            provider="local",
            bucket=self._bucket,
            path=storage_path,
            size_bytes=len(file_bytes),
            content_type=content_type,
        )

    def download_file(
        self,
        *,
        storage_path: str,
    ) -> bytes:
        target_path = self._resolve_storage_path(storage_path)
        try:
            return target_path.read_bytes()
        except OSError as exc:
            raise StorageError("Unable to read knowledge file from local storage.") from exc

    def delete_file(
        self,
        *,
        storage_path: str,
    ) -> None:
        target_path = self._resolve_storage_path(storage_path)
        try:
            target_path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Unable to delete knowledge file from local storage.") from exc

    def _resolve_storage_path(self, storage_path: str) -> Path:
        return self._root_dir / self._bucket / Path(storage_path)
