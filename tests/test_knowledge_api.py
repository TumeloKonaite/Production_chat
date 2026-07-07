from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import get_knowledge_ingestion_service_factory
from app.config import Settings
from app.knowledge.ingestion import (
    KnowledgeIngestionDocumentResult,
    KnowledgeIngestionRunResult,
    KnowledgeIngestionServiceError,
)
from app.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def build_test_settings() -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        tavus_api_key="tavus-test-key",
        tavus_base_url="https://tavus.example",
        tavus_face_id="face_123",
        tavus_pal_id="pal_123",
        public_backend_url="https://backend.example",
        tavus_tool_secret="tool-secret",
        ingestion_api_secret="ingestion-secret",
        eval_admin_token="eval-secret",
        default_model_config_id="openai:gpt-4.1-mini",
        model_configs_json=None,
        embedding_provider="hf",
        knowledge_embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version="v1_professional",
        conversation_history_limit=10,
        retriever_type="vector",
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="personal-chatbot-model-comparison",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
        storage_provider="local",
        minio_endpoint="http://localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="knowledge-files",
        minio_secure=False,
        local_storage_path=".pytest_tmp/storage",
        knowledge_upload_max_bytes=10485760,
    )


class FakeKnowledgeIngestionService:
    def __init__(
        self,
        result: KnowledgeIngestionRunResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result or KnowledgeIngestionRunResult(
            status="ok",
            documents_loaded=2,
            results=[
                KnowledgeIngestionDocumentResult(source="profile.md", chunk_count=3),
                KnowledgeIngestionDocumentResult(source="projects.md", chunk_count=7),
            ],
        )
        self.error = error
        self.calls: list[Session] = []

    def run(self, session: Session) -> KnowledgeIngestionRunResult:
        self.calls.append(session)
        if self.error is not None:
            raise self.error
        return self.result


def override_db_session() -> Generator[Session, None, None]:
    yield Session()


class CapturingIngestionServiceFactory:
    def __init__(self) -> None:
        self.created_settings: list[Settings] = []
        self.service = FakeKnowledgeIngestionService()

    def __call__(self, settings: Settings | None = None) -> FakeKnowledgeIngestionService:
        if settings is not None:
            self.created_settings.append(settings)
        return self.service


class FakeTracker:
    def __init__(self) -> None:
        self.enabled = True
        self.run_names: list[str] = []
        self.logged_params: list[dict[str, object]] = []

    @contextmanager
    def run(self, run_name: str):
        self.run_names.append(run_name)
        yield None

    def log_params(self, params: dict[str, object]) -> None:
        self.logged_params.append(params)


def test_knowledge_ingest_rejects_missing_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: (lambda settings=None: FakeKnowledgeIngestionService())
    )

    client = TestClient(app)
    response = client.post("/api/knowledge/ingest")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_rejects_invalid_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: (lambda settings=None: FakeKnowledgeIngestionService())
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_accepts_valid_secret_and_returns_summary() -> None:
    factory = CapturingIngestionServiceFactory()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: factory

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "experiment_name": None,
        "embedding_provider": "hf",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "documents_loaded": 2,
        "chunks_created": 10,
        "results": [
            {"source": "profile.md", "chunk_count": 3},
            {"source": "projects.md", "chunk_count": 7},
        ],
    }
    assert len(factory.service.calls) == 1
    assert len(factory.created_settings) == 1
    assert factory.created_settings[0].embedding_provider == "hf"
    assert factory.created_settings[0].knowledge_embedding_model == "all-MiniLM-L6-v2"
    assert factory.created_settings[0].embedding_dimension == 384


def test_knowledge_ingest_accepts_request_level_hf_embedding_override() -> None:
    factory = CapturingIngestionServiceFactory()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: factory

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={
            "experiment_name": "hf-minilm-baseline",
            "embedding_provider": "hf",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "reset_existing_vectors": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["experiment_name"] == "hf-minilm-baseline"
    assert response.json()["embedding_provider"] == "hf"
    assert response.json()["embedding_model"] == "all-MiniLM-L6-v2"
    assert response.json()["embedding_dimension"] == 384
    assert factory.created_settings[0].embedding_provider == "hf"
    assert factory.created_settings[0].knowledge_embedding_model == "all-MiniLM-L6-v2"
    assert factory.created_settings[0].embedding_dimension == 384


def test_knowledge_ingest_accepts_request_level_openai_embedding_override() -> None:
    factory = CapturingIngestionServiceFactory()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: factory

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={
            "experiment_name": "openai-text-embedding-3-small",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "reset_existing_vectors": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["experiment_name"] == "openai-text-embedding-3-small"
    assert response.json()["embedding_provider"] == "openai"
    assert response.json()["embedding_model"] == "text-embedding-3-small"
    assert response.json()["embedding_dimension"] == 1536
    assert factory.created_settings[0].embedding_provider == "openai"
    assert factory.created_settings[0].knowledge_embedding_model == "text-embedding-3-small"
    assert factory.created_settings[0].embedding_dimension == 1536


def test_knowledge_ingest_rejects_partial_embedding_override() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: (lambda settings=None: FakeKnowledgeIngestionService())
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"embedding_model": "text-embedding-3-small"},
    )

    assert response.status_code == 422
    assert "embedding_provider, embedding_model, and embedding_dimension" in str(
        response.json()
    )


def test_knowledge_ingest_requires_reset_flag_for_embedding_override() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: (lambda settings=None: FakeKnowledgeIngestionService())
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
        },
    )

    assert response.status_code == 422
    assert "reset_existing_vectors=true is required" in str(response.json())


def test_knowledge_ingest_logs_effective_embedding_config_when_tracking_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = CapturingIngestionServiceFactory()
    tracker = FakeTracker()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: factory
    monkeypatch.setattr("app.api.knowledge.routes.create_experiment_tracker", lambda *_: tracker)

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={
            "experiment_name": "openai-text-embedding-3-small",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "reset_existing_vectors": True,
        },
    )

    assert response.status_code == 200
    assert tracker.run_names == ["openai-text-embedding-3-small"]
    assert tracker.logged_params == [
        {
            "experiment_name": "openai-text-embedding-3-small",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "reset_existing_vectors": True,
        }
    ]


def test_knowledge_ingest_returns_clear_server_error() -> None:
    fake_service = FakeKnowledgeIngestionService(error=KnowledgeIngestionServiceError())
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: (
        lambda settings=None: fake_service
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to ingest knowledge. Please try again."}


def test_knowledge_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/knowledge/ingest" in paths
    assert "/api/knowledge/files" in paths
