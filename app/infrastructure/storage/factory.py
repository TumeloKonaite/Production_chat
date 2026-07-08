from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.infrastructure.storage.base import KnowledgeFileStorage
from app.infrastructure.storage.local_storage import LocalKnowledgeFileStorage
from app.infrastructure.storage.minio_storage import MinioKnowledgeFileStorage
from app.infrastructure.storage.supabase_storage import SupabaseKnowledgeFileStorage


def create_knowledge_file_storage(settings: Settings) -> KnowledgeFileStorage:
    if settings.storage_provider == "minio":
        return MinioKnowledgeFileStorage(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
    if settings.storage_provider == "supabase":
        if settings.supabase_url is None:
            raise ValueError("SUPABASE_URL is required when STORAGE_PROVIDER=supabase.")
        if settings.supabase_service_role_key is None:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY is required when STORAGE_PROVIDER=supabase."
            )
        if settings.supabase_storage_bucket is None:
            raise ValueError(
                "SUPABASE_STORAGE_BUCKET is required when STORAGE_PROVIDER=supabase."
            )
        return SupabaseKnowledgeFileStorage(
            url=settings.supabase_url,
            service_role_key=settings.supabase_service_role_key,
            bucket=settings.supabase_storage_bucket,
        )
    if settings.storage_provider == "local":
        return LocalKnowledgeFileStorage(
            root_dir=Path(settings.local_storage_path),
            bucket=settings.minio_bucket,
        )

    raise ValueError(f"Unsupported storage provider: {settings.storage_provider}")
