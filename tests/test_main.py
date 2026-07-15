from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+psycopg://postgres:secret@db.example.com:5432/app",
        "openai_api_key": "openai-secret",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": None,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "ingestion_api_secret": None,
        "eval_admin_token": None,
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
        "mlflow_experiment_name": "production-chatbot",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "app_env": "production",
        "frontend_origin": "https://frontend.example.com",
        "vector_store_provider": "supabase_pgvector",
        "supabase_url": "https://project.supabase.co",
        "supabase_service_role_key": "supabase-secret",
        "enable_redis": True,
        "upstash_redis_rest_url": "https://cache.example.com",
        "upstash_redis_rest_token": "redis-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_create_app_uses_configured_frontend_origin() -> None:
    app = create_app(build_settings(frontend_origin="https://frontend.example.com, https://admin.example.com"))
    cors_middleware = next(
        middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware"
    )

    assert cors_middleware.kwargs["allow_origins"] == [
        "https://frontend.example.com",
        "https://admin.example.com",
    ]


def test_startup_logs_report_configuration_without_secrets(caplog) -> None:
    app = create_app(build_settings())

    with caplog.at_level(logging.INFO):
        with TestClient(app):
            pass

    assert "App environment: production" in caplog.text
    assert "Vector store provider: supabase_pgvector" in caplog.text
    assert "Langfuse enabled: False" in caplog.text
    assert "MLflow tracking enabled: False" in caplog.text
    assert "Redis configured: True" in caplog.text
    assert "Supabase configured: True" in caplog.text
    assert "OpenAI base URL configured: True" in caplog.text
    assert "openai-secret" not in caplog.text
    assert "supabase-secret" not in caplog.text
    assert "redis-secret" not in caplog.text
    assert "postgresql+psycopg://postgres:secret@db.example.com:5432/app" not in caplog.text


def test_startup_does_not_initialize_experiment_tracking(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.tracking.setup.create_experiment_tracker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tracking should not initialize")),
    )

    app = create_app(build_settings())

    with TestClient(app):
        pass
