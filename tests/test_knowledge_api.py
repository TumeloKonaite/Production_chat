from __future__ import annotations

from collections.abc import Generator
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import get_knowledge_ingestion_orchestrator
from app.config import Settings
from app.knowledge.ingestion import (
    KnowledgeIngestionConflictError,
    KnowledgeIngestionGoneError,
    KnowledgeIngestionNotFoundError,
    KnowledgeIngestionServiceError,
    KnowledgeIngestionTriggerResult,
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


class FakeKnowledgeIngestionOrchestrator:
    def __init__(
        self,
        result: KnowledgeIngestionTriggerResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result or KnowledgeIngestionTriggerResult(
            job_id=str(uuid4()),
            status="queued",
            source_type="local_directory",
            file_id=None,
        )
        self.error = error
        self.calls: list[tuple[Session, object, Settings]] = []

    def trigger(
        self,
        session: Session,
        *,
        request,
        effective_settings: Settings,
    ) -> KnowledgeIngestionTriggerResult:
        self.calls.append((session, request, effective_settings))
        if self.error is not None:
            raise self.error
        return self.result


def override_db_session() -> Generator[Session, None, None]:
    yield Session()


def test_knowledge_ingest_rejects_missing_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = (
        lambda: FakeKnowledgeIngestionOrchestrator()
    )

    client = TestClient(app)
    response = client.post("/api/knowledge/ingest")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_rejects_invalid_secret() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = (
        lambda: FakeKnowledgeIngestionOrchestrator()
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid ingestion secret."}


def test_knowledge_ingest_returns_queued_job_for_local_directory() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        result=KnowledgeIngestionTriggerResult(
            job_id=str(uuid4()),
            status="queued",
            source_type="local_directory",
            file_id=None,
        )
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": orchestrator.result.job_id,
        "status": "queued",
        "source_type": "local_directory",
        "file_id": None,
    }
    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0][1].source_type == "local_directory"


def test_knowledge_ingest_accepts_request_level_embedding_override() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "reset_existing_vectors": True,
        },
    )

    assert response.status_code == 202
    effective_settings = orchestrator.calls[0][2]
    assert effective_settings.embedding_provider == "openai"
    assert effective_settings.knowledge_embedding_model == "text-embedding-3-small"
    assert effective_settings.embedding_dimension == 1536


def test_knowledge_ingest_rejects_partial_embedding_override() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = (
        lambda: FakeKnowledgeIngestionOrchestrator()
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


def test_knowledge_ingest_accepts_uploaded_file_request() -> None:
    file_id = str(uuid4())
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        result=KnowledgeIngestionTriggerResult(
            job_id=str(uuid4()),
            status="queued",
            source_type="uploaded_file",
            file_id=file_id,
        )
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file", "file_id": file_id},
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": orchestrator.result.job_id,
        "status": "queued",
        "source_type": "uploaded_file",
        "file_id": file_id,
    }
    assert str(orchestrator.calls[0][1].file_id) == file_id


def test_knowledge_ingest_returns_skipped_job_for_duplicate_request() -> None:
    file_id = str(uuid4())
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        result=KnowledgeIngestionTriggerResult(
            job_id=str(uuid4()),
            status="skipped",
            source_type="uploaded_file",
            file_id=file_id,
        )
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file", "file_id": file_id},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


def test_knowledge_ingest_uploaded_file_requires_file_id() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = (
        lambda: FakeKnowledgeIngestionOrchestrator()
    )

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file"},
    )

    assert response.status_code == 422
    assert "file_id is required" in str(response.json())


def test_knowledge_ingest_uploaded_file_maps_not_found_error() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        error=KnowledgeIngestionNotFoundError("Knowledge file not found.")
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file", "file_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Knowledge file not found."}


def test_knowledge_ingest_uploaded_file_maps_conflict_error() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        error=KnowledgeIngestionConflictError("Knowledge file is already being ingested.")
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file", "file_id": str(uuid4())},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Knowledge file is already being ingested."}


def test_knowledge_ingest_uploaded_file_maps_gone_error() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator(
        error=KnowledgeIngestionGoneError("Knowledge file has been deleted.")
    )
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

    client = TestClient(app)
    response = client.post(
        "/api/knowledge/ingest",
        headers={"x-ingestion-secret": "ingestion-secret"},
        json={"source_type": "uploaded_file", "file_id": str(uuid4())},
    )

    assert response.status_code == 410
    assert response.json() == {"detail": "Knowledge file has been deleted."}


def test_knowledge_ingest_returns_clear_server_error() -> None:
    orchestrator = FakeKnowledgeIngestionOrchestrator(error=KnowledgeIngestionServiceError())
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_knowledge_ingestion_orchestrator] = lambda: orchestrator

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
