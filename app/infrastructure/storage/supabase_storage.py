from __future__ import annotations

from pathlib import PurePosixPath

import httpx

from app.infrastructure.storage.base import KnowledgeFileStorage, StorageError, StoredFile

_DEFAULT_CONTENT_TYPE = "text/plain;charset=UTF-8"


class SupabaseKnowledgeFileStorage(KnowledgeFileStorage):
    def __init__(
        self,
        *,
        url: str,
        service_role_key: str,
        bucket: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._bucket = bucket
        self._client = http_client or httpx.Client(
            base_url=f"{url.rstrip('/')}/storage/v1",
            headers={
                "apiKey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
            },
            timeout=20.0,
        )

    def upload_file(
        self,
        *,
        file_bytes: bytes,
        storage_path: str,
        content_type: str | None = None,
    ) -> StoredFile:
        filename = PurePosixPath(storage_path).name or "upload"
        try:
            response = self._client.post(
                f"/object/{self._bucket}/{storage_path}",
                headers={
                    "cache-control": "max-age=3600",
                    "x-upsert": "false",
                },
                data={"cacheControl": "3600"},
                files={
                    "file": (
                        filename,
                        file_bytes,
                        content_type or _DEFAULT_CONTENT_TYPE,
                    )
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise StorageError("Unable to upload knowledge file to Supabase Storage.") from exc
        except httpx.HTTPError as exc:
            raise StorageError("Unable to upload knowledge file to Supabase Storage.") from exc

        return StoredFile(
            provider="supabase",
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
        try:
            response = self._client.get(f"/object/{self._bucket}/{storage_path}")
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise StorageError("Storage object not found.") from FileNotFoundError(storage_path)
            raise StorageError("Unable to download knowledge file from Supabase Storage.") from exc
        except httpx.HTTPError as exc:
            raise StorageError("Unable to download knowledge file from Supabase Storage.") from exc

    def delete_file(
        self,
        *,
        storage_path: str,
    ) -> None:
        try:
            response = self._client.request(
                "DELETE",
                f"/object/{self._bucket}",
                json={"prefixes": [storage_path]},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise StorageError("Unable to delete knowledge file from Supabase Storage.") from exc
        except httpx.HTTPError as exc:
            raise StorageError("Unable to delete knowledge file from Supabase Storage.") from exc
