from __future__ import annotations

from collections.abc import Generator
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import get_knowledge_file_upload_service
from app.config import Settings
from app.infrastructure.storage import (
    LocalKnowledgeFileStorage,
    MinioKnowledgeFileStorage,
    SupabaseKnowledgeFileStorage,
    create_knowledge_file_storage,
)
from app.repositories.db.base import Base
from app.repositories.knowledge_file_repository import KnowledgeFileRepositoryError
from app.repositories.models import KnowledgeFile
from app.main import app
from app.services.knowledge_files import KnowledgeFileUploadService, KnowledgeFileUploadError


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def build_test_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite:///unused-for-tests.db",
        "openai_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": "openrouter-test-key",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "ingestion_api_secret": "ingestion-secret",
        "eval_admin_token": "eval-secret",
        "default_model_config_id": "openai:gpt-4.1-mini",
        "model_configs_json": None,
        "embedding_provider": "hf",
        "knowledge_embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "knowledge_collection_name": "personal_knowledge_base",
        "default_prompt_version": "v1_professional",
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "personal-chatbot-model-comparison",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "storage_provider": "local",
        "minio_endpoint": "http://localhost:9000",
        "minio_access_key": "minioadmin",
        "minio_secret_key": "minioadmin",
        "minio_bucket": "knowledge-files",
        "minio_secure": False,
        "local_storage_path": str(tmp_path / "storage"),
        "knowledge_upload_max_bytes": 10485760,
        "supabase_url": "https://project.supabase.co",
        "supabase_service_role_key": "service-role",
        "supabase_storage_bucket": "knowledge-files",
    }
    values.update(overrides)
    return Settings(**values)


def build_session_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def make_db_session_override(
    session_factory: sessionmaker[Session],
):
    def _override() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    return _override


def build_upload_service(settings: Settings) -> KnowledgeFileUploadService:
    return KnowledgeFileUploadService(
        settings=settings,
        storage=create_knowledge_file_storage(settings),
    )


def build_supabase_http_client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="https://project.supabase.co/storage/v1",
        headers={
            "apiKey": "service-role",
            "Authorization": "Bearer service-role",
        },
        transport=httpx.MockTransport(handler),
    )


def test_knowledge_file_upload_accepts_markdown_file_and_persists_metadata(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_session_factory(tmp_path)
    upload_service = build_upload_service(settings)

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    file_bytes = b"# Company\n\nGrounded answers.\n"
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("company-profile.md", file_bytes, "text/markdown")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["original_filename"] == "company-profile.md"
    assert payload["content_type"] == "text/markdown"
    assert payload["file_size_bytes"] == len(file_bytes)
    assert payload["storage_provider"] == "local"
    assert payload["storage_bucket"] == "knowledge-files"
    assert payload["storage_path"].endswith("/company-profile.md")
    assert payload["status"] == "uploaded"

    with session_factory() as session:
        stored_files = list(session.scalars(select(KnowledgeFile)))

    assert len(stored_files) == 1
    assert stored_files[0].storage_path == payload["storage_path"]
    assert upload_service.download_file_bytes(storage_path=payload["storage_path"]) == file_bytes


def test_knowledge_file_upload_accepts_text_file(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_session_factory(tmp_path)
    upload_service = build_upload_service(settings)

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("notes.txt", b"Operational notes", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["original_filename"] == "notes.txt"
    assert response.json()["content_type"] == "text/plain"


def test_knowledge_file_upload_rejects_empty_file(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_session_factory(tmp_path)
    upload_service = build_upload_service(settings)

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Knowledge files cannot be empty."}


def test_knowledge_file_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    session_factory = build_session_factory(tmp_path)
    upload_service = build_upload_service(settings)

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("manual.pdf", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only .md and .txt files are supported."}


def test_knowledge_file_upload_rejects_oversized_file(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path, knowledge_upload_max_bytes=4)
    session_factory = build_session_factory(tmp_path)
    upload_service = build_upload_service(settings)

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("profile.md", b"12345", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Knowledge file exceeds the maximum upload size."}


def test_knowledge_file_upload_metadata_failure_triggers_cleanup(tmp_path: Path) -> None:
    class RecordingStorage:
        def __init__(self) -> None:
            self.uploaded_paths: list[str] = []
            self.deleted_paths: list[str] = []

        def upload_file(
            self,
            *,
            file_bytes: bytes,
            storage_path: str,
            content_type: str | None = None,
        ):
            self.uploaded_paths.append(storage_path)
            return type(
                "Stored",
                (),
                {
                    "provider": "local",
                    "bucket": "knowledge-files",
                    "path": storage_path,
                    "size_bytes": len(file_bytes),
                    "content_type": content_type,
                },
            )()

        def download_file(self, *, storage_path: str) -> bytes:
            raise AssertionError("download_file should not be called")

        def delete_file(self, *, storage_path: str) -> None:
            self.deleted_paths.append(storage_path)

    class FailingRepository:
        def __init__(self, session: Session) -> None:
            self._session = session

        def create(self, knowledge_file: KnowledgeFile) -> KnowledgeFile:
            raise KnowledgeFileRepositoryError()

    settings = build_test_settings(tmp_path)
    session_factory = build_session_factory(tmp_path)
    storage = RecordingStorage()
    upload_service = KnowledgeFileUploadService(
        settings=settings,
        storage=storage,
        repository_factory=FailingRepository,
    )

    with session_factory() as session:
        with pytest.raises(
            KnowledgeFileUploadError,
            match="Unable to persist uploaded knowledge file metadata.",
        ):
            upload_service.upload_file(
                session,
                filename="company-profile.md",
                content_type="text/markdown",
                file_bytes=b"# Company\n",
            )

    assert len(storage.uploaded_paths) == 1
    assert storage.deleted_paths == storage.uploaded_paths


def test_knowledge_file_upload_metadata_failure_triggers_supabase_cleanup(tmp_path: Path) -> None:
    requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        requests.append((request.method, request.url.path, body))
        if request.method == "POST":
            return httpx.Response(200, json={"Key": "ignored"})
        if request.method == "DELETE":
            return httpx.Response(200, json=[{"name": "company-profile.md"}])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    class FailingRepository:
        def __init__(self, session: Session) -> None:
            self._session = session

        def create(self, knowledge_file: KnowledgeFile) -> KnowledgeFile:
            raise KnowledgeFileRepositoryError()

    settings = build_test_settings(tmp_path, storage_provider="supabase")
    session_factory = build_session_factory(tmp_path)
    storage = SupabaseKnowledgeFileStorage(
        url=settings.supabase_url or "",
        service_role_key=settings.supabase_service_role_key or "",
        bucket=settings.supabase_storage_bucket or "",
        http_client=build_supabase_http_client(handler),
    )
    upload_service = KnowledgeFileUploadService(
        settings=settings,
        storage=storage,
        repository_factory=FailingRepository,
    )

    with session_factory() as session:
        with pytest.raises(
            KnowledgeFileUploadError,
            match="Unable to persist uploaded knowledge file metadata.",
        ):
            upload_service.upload_file(
                session,
                filename="company-profile.md",
                content_type="text/markdown",
                file_bytes=b"# Company\n",
            )

    assert [method for method, _path, _body in requests] == ["POST", "DELETE"]
    assert requests[1][1] == "/storage/v1/object/knowledge-files"
    assert json.loads(requests[1][2].decode("utf-8"))["prefixes"][0].endswith(
        "/company-profile.md"
    )


def test_storage_factory_returns_minio_storage_when_configured(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path, storage_provider="minio")

    storage = create_knowledge_file_storage(settings)

    assert isinstance(storage, MinioKnowledgeFileStorage)


def test_storage_factory_returns_local_storage_when_configured(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path, storage_provider="local")

    storage = create_knowledge_file_storage(settings)

    assert isinstance(storage, LocalKnowledgeFileStorage)


def test_storage_factory_returns_supabase_storage_when_configured(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path, storage_provider="supabase")

    storage = create_knowledge_file_storage(settings)

    assert isinstance(storage, SupabaseKnowledgeFileStorage)


def test_storage_factory_rejects_missing_supabase_config(tmp_path: Path) -> None:
    settings = build_test_settings(
        tmp_path,
        storage_provider="supabase",
        supabase_url=None,
    )

    with pytest.raises(ValueError, match="SUPABASE_URL is required when STORAGE_PROVIDER=supabase."):
        create_knowledge_file_storage(settings)


def test_supabase_storage_upload_uses_expected_bucket_and_path(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["Content-Type"]
        captured["authorization"] = request.headers["Authorization"]
        captured["api_key"] = request.headers["apiKey"]
        captured["body"] = request.read()
        return httpx.Response(200, json={"Key": "uploaded/company-profile.md"})

    storage = SupabaseKnowledgeFileStorage(
        url="https://project.supabase.co",
        service_role_key="service-role",
        bucket="knowledge-files",
        http_client=build_supabase_http_client(handler),
    )

    stored_file = storage.upload_file(
        file_bytes=b"# Company\n",
        storage_path="uploaded/company-profile.md",
        content_type="text/markdown",
    )

    assert stored_file.provider == "supabase"
    assert stored_file.bucket == "knowledge-files"
    assert stored_file.path == "uploaded/company-profile.md"
    assert captured["method"] == "POST"
    assert captured["path"] == "/storage/v1/object/knowledge-files/uploaded/company-profile.md"
    assert captured["authorization"] == "Bearer service-role"
    assert captured["api_key"] == "service-role"
    assert "multipart/form-data" in str(captured["content_type"])
    assert b'filename="company-profile.md"' in captured["body"]
    assert b"# Company\n" in captured["body"]


def test_supabase_storage_download_returns_file_bytes() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, content=b"# Company\n")

    storage = SupabaseKnowledgeFileStorage(
        url="https://project.supabase.co",
        service_role_key="service-role",
        bucket="knowledge-files",
        http_client=build_supabase_http_client(handler),
    )

    downloaded = storage.download_file(storage_path="uploaded/company-profile.md")

    assert downloaded == b"# Company\n"
    assert captured == {
        "method": "GET",
        "path": "/storage/v1/object/knowledge-files/uploaded/company-profile.md",
    }


def test_supabase_storage_delete_removes_expected_object() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(200, json=[{"name": "uploaded/company-profile.md"}])

    storage = SupabaseKnowledgeFileStorage(
        url="https://project.supabase.co",
        service_role_key="service-role",
        bucket="knowledge-files",
        http_client=build_supabase_http_client(handler),
    )

    storage.delete_file(storage_path="uploaded/company-profile.md")

    assert captured == {
        "method": "DELETE",
        "path": "/storage/v1/object/knowledge-files",
        "json": {"prefixes": ["uploaded/company-profile.md"]},
    }


def test_knowledge_file_upload_persists_supabase_metadata(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        return httpx.Response(200, json={"Key": "ignored"})

    settings = build_test_settings(tmp_path, storage_provider="supabase")
    session_factory = build_session_factory(tmp_path)
    upload_service = KnowledgeFileUploadService(
        settings=settings,
        storage=SupabaseKnowledgeFileStorage(
            url=settings.supabase_url or "",
            service_role_key=settings.supabase_service_role_key or "",
            bucket=settings.supabase_storage_bucket or "",
            http_client=build_supabase_http_client(handler),
        ),
    )

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_db_session] = make_db_session_override(session_factory)
    app.dependency_overrides[get_knowledge_file_upload_service] = lambda: upload_service

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/files",
        files={"file": ("company-profile.md", b"# Company\n", "text/markdown")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["storage_provider"] == "supabase"
    assert payload["storage_bucket"] == "knowledge-files"
    assert payload["storage_path"].endswith("/company-profile.md")


def test_minio_storage_round_trip_when_enabled() -> None:
    if os.getenv("RUN_MINIO_INTEGRATION_TESTS", "").casefold() != "true":
        pytest.skip("Set RUN_MINIO_INTEGRATION_TESTS=true to run MinIO integration tests.")

    storage = MinioKnowledgeFileStorage(
        endpoint=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.getenv("MINIO_BUCKET", "knowledge-files"),
        secure=os.getenv("MINIO_SECURE", "").casefold() in {"1", "true", "yes", "on"},
    )

    stored_file = storage.upload_file(
        file_bytes=b"# Integration\n",
        storage_path="integration-tests/integration.md",
        content_type="text/markdown",
    )
    downloaded = storage.download_file(storage_path=stored_file.path)
    storage.delete_file(storage_path=stored_file.path)

    assert downloaded == b"# Integration\n"
