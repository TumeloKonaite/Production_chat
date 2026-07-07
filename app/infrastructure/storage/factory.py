from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.infrastructure.storage.base import KnowledgeFileStorage
from app.infrastructure.storage.local_storage import LocalKnowledgeFileStorage
from app.infrastructure.storage.minio_storage import MinioKnowledgeFileStorage


def create_knowledge_file_storage(settings: Settings) -> KnowledgeFileStorage:
    if settings.storage_provider == "minio":
        return MinioKnowledgeFileStorage(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )

    return LocalKnowledgeFileStorage(
        root_dir=Path(settings.local_storage_path),
        bucket=settings.minio_bucket,
    )
