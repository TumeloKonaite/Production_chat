from __future__ import annotations

from collections.abc import Generator

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


def test_knowledge_ingest_rejects_missing_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: FakeKnowledgeIngestionService
    )

    client = TestClient(app)
    response = client.post("/api/knowledge/ingest")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_rejects_invalid_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = (
        lambda: FakeKnowledgeIngestionService
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_accepts_valid_secret_and_returns_summary() -> None:
    fake_service = FakeKnowledgeIngestionService()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: (
        lambda: fake_service
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "documents_loaded": 2,
        "results": [
            {"source": "profile.md", "chunk_count": 3},
            {"source": "projects.md", "chunk_count": 7},
        ],
    }
    assert len(fake_service.calls) == 1


def test_knowledge_ingest_returns_clear_server_error() -> None:
    fake_service = FakeKnowledgeIngestionService(error=KnowledgeIngestionServiceError())
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_service_factory] = lambda: (
        lambda: fake_service
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
