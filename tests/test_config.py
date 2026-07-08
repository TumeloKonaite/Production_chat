from __future__ import annotations

from collections.abc import Generator

import pytest

from app.config import (
    DEFAULT_LOCAL_FRONTEND_ORIGIN,
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
    DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    SUPPORTED_APP_ENVS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_RERANKER_TYPES,
    SUPPORTED_RESPONSE_CACHE_PROVIDERS,
    SUPPORTED_RETRIEVER_TYPES,
    SUPPORTED_STORAGE_PROVIDERS,
    SUPPORTED_VECTOR_STORE_PROVIDERS,
    get_settings,
)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_uses_default_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    settings = get_settings()

    assert settings.openai_base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.openrouter_base_url == DEFAULT_OPENROUTER_BASE_URL


def test_get_settings_uses_custom_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/custom/")

    settings = get_settings()

    assert settings.openai_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1/custom"


def test_get_settings_uses_generic_llm_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_PROMPT_COST_PER_1M_TOKENS", "3.0")
    monkeypatch.setenv("LLM_COMPLETION_COST_PER_1M_TOKENS", "15.0")

    settings = get_settings()

    assert settings.default_model_config_id == "openrouter:anthropic/claude-3.5-sonnet"
    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "anthropic/claude-3.5-sonnet"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_api_key == "generic-key"
    assert settings.openrouter_api_key == "generic-key"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_prompt_cost_per_1m_tokens == 3.0
    assert settings.llm_completion_cost_per_1m_tokens == 15.0


def test_get_settings_uses_default_chunking_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    monkeypatch.delenv("DATABASE_DIRECT_URL", raising=False)
    monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)
    monkeypatch.delenv("STORAGE_PROVIDER", raising=False)
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    monkeypatch.delenv("MINIO_BUCKET", raising=False)
    monkeypatch.delenv("MINIO_SECURE", raising=False)
    monkeypatch.delenv("LOCAL_STORAGE_PATH", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("RETRIEVER_TYPE", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    monkeypatch.delenv("EVAL_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("ENABLE_LANGFUSE", raising=False)
    monkeypatch.delenv("ENABLE_LANGFUSE_OBSERVABILITY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_ENVIRONMENT", raising=False)
    monkeypatch.delenv("LANGFUSE_RELEASE", raising=False)
    monkeypatch.delenv("LANGFUSE_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("LANGFUSE_EXPORT_DEFAULT_LIMIT", raising=False)
    monkeypatch.delenv("ENABLE_REDIS", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("RATE_LIMIT_MAX_REQUESTS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("EXACT_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("EXACT_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("REQUEST_LOCK_ENABLED", raising=False)
    monkeypatch.delenv("REQUEST_LOCK_TTL_SECONDS", raising=False)
    monkeypatch.delenv("ENABLE_RESPONSE_CACHE", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_PROVIDER", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_TOKEN", raising=False)
    monkeypatch.delenv("ENABLE_EXACT_RESPONSE_CACHE", raising=False)
    monkeypatch.delenv("ENABLE_SEMANTIC_RESPONSE_CACHE", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_EXACT_PREFIX", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_SEMANTIC_INDEX", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_DISTANCE_THRESHOLD", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_MAX_RESULTS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_STORE_PRIVATE_SESSIONS", raising=False)
    monkeypatch.delenv("RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION", raising=False)
    monkeypatch.delenv("ENABLE_RATE_LIMITING", raising=False)
    monkeypatch.delenv("RATE_LIMITING_FAIL_OPEN", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_REQUESTS_PER_10_MINUTES", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_REQUESTS_PER_DAY", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_DAILY_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_DAILY_COST_BUDGET_USD", raising=False)

    settings = get_settings()

    assert settings.app_env == "local"
    assert settings.frontend_origin is None
    assert settings.frontend_origins == [DEFAULT_LOCAL_FRONTEND_ORIGIN]
    assert settings.database_direct_url is None
    assert settings.vector_store_provider == "pgvector"
    assert settings.storage_provider == "minio"
    assert settings.minio_endpoint == "http://localhost:9000"
    assert settings.minio_access_key == "minioadmin"
    assert settings.minio_secret_key == "minioadmin"
    assert settings.minio_bucket == "knowledge-files"
    assert settings.minio_secure is False
    assert settings.local_storage_path == ".storage"
    assert settings.knowledge_upload_max_bytes == 10485760
    assert settings.supabase_storage_bucket == "knowledge-files"
    assert settings.knowledge_chunk_size == DEFAULT_KNOWLEDGE_CHUNK_SIZE
    assert settings.knowledge_chunk_overlap == DEFAULT_KNOWLEDGE_CHUNK_OVERLAP
    assert settings.retriever_type == "vector"
    assert settings.retrieval_top_k == 5
    assert settings.eval_admin_token is None
    assert settings.enable_query_rewriting is False
    assert settings.query_rewrite_model == "openai:gpt-4.1-mini"
    assert settings.query_rewrite_temperature == 0.0
    assert settings.query_rewrite_prompt_version == "v1"
    assert settings.query_rewrite_timeout_seconds == 10
    assert settings.query_rewrite_max_tokens == 128
    assert settings.enable_reranking is False
    assert settings.reranker_type == "none"
    assert settings.reranker_model == "openai:gpt-4.1-mini"
    assert settings.reranker_initial_top_k == 20
    assert settings.reranker_final_top_k == 5
    assert settings.enable_langfuse_observability is False
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.langfuse_environment == "local"
    assert settings.langfuse_release is None
    assert settings.langfuse_sample_rate == 1.0
    assert settings.langfuse_export_default_limit == 100
    assert settings.enable_production_feedback_export is False
    assert settings.allow_raw_production_text_in_evals is False
    assert settings.enable_redis is False
    assert settings.upstash_redis_rest_url is None
    assert settings.upstash_redis_rest_token is None
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_max_requests == 20
    assert settings.rate_limit_window_seconds == 60
    assert settings.exact_cache_enabled is True
    assert settings.exact_cache_ttl_seconds == 300
    assert settings.request_lock_enabled is True
    assert settings.request_lock_ttl_seconds == 30
    assert settings.enable_response_cache is False
    assert settings.response_cache_provider == "redis"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.redis_token is None
    assert settings.resolved_redis_url == "redis://localhost:6379/0"
    assert settings.redis_healthcheck_enabled is False
    assert settings.enable_exact_response_cache is True
    assert settings.enable_semantic_response_cache is False
    assert settings.response_cache_ttl_seconds == 604800
    assert settings.response_cache_exact_prefix == "chat:exact"
    assert settings.response_cache_semantic_index == "chat_semantic_cache"
    assert settings.response_cache_distance_threshold == 0.10
    assert settings.response_cache_max_results == 3
    assert settings.response_cache_store_private_sessions is False
    assert settings.response_cache_knowledge_base_version == "personal_knowledge_base"
    assert settings.enable_rate_limiting is False
    assert settings.rate_limiting_fail_open is True
    assert settings.chat_rate_limit_requests_per_10_minutes == 20
    assert settings.chat_rate_limit_requests_per_day == 100
    assert settings.chat_rate_limit_concurrent_requests == 3
    assert settings.chat_rate_limit_daily_token_budget == 100000
    assert settings.chat_rate_limit_daily_cost_budget_usd == 0.50


def test_get_settings_uses_configured_chunking_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("EVAL_ADMIN_TOKEN", "eval-secret")

    settings = get_settings()

    assert settings.knowledge_chunk_size == 500
    assert settings.knowledge_chunk_overlap == 100
    assert settings.eval_admin_token == "eval-secret"


def test_get_settings_prefers_knowledge_embedding_dimension_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    settings = get_settings()

    assert settings.embedding_dimension == 1536


def test_get_settings_uses_configured_query_rewrite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_QUERY_REWRITING", "true")
    monkeypatch.setenv("QUERY_REWRITE_MODEL", "openai:gpt-4.1")
    monkeypatch.setenv("QUERY_REWRITE_TEMPERATURE", "0.2")
    monkeypatch.setenv("QUERY_REWRITE_PROMPT_VERSION", "v2")
    monkeypatch.setenv("QUERY_REWRITE_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("QUERY_REWRITE_MAX_TOKENS", "64")

    settings = get_settings()

    assert settings.enable_query_rewriting is True
    assert settings.query_rewrite_model == "openai:gpt-4.1"
    assert settings.query_rewrite_temperature == 0.2
    assert settings.query_rewrite_prompt_version == "v2"
    assert settings.query_rewrite_timeout_seconds == 7
    assert settings.query_rewrite_max_tokens == 64


def test_get_settings_uses_configured_reranker_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    monkeypatch.setenv("RERANKER_TYPE", "llm")
    monkeypatch.setenv("RERANKER_MODEL", "openai:gpt-4.1")
    monkeypatch.setenv("RERANKER_INITIAL_TOP_K", "25")
    monkeypatch.setenv("RERANKER_FINAL_TOP_K", "7")

    settings = get_settings()

    assert settings.enable_reranking is True
    assert settings.reranker_type == "llm"
    assert settings.reranker_model == "openai:gpt-4.1"
    assert settings.reranker_initial_top_k == 25
    assert settings.reranker_final_top_k == 7


def test_get_settings_uses_configured_langfuse_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_LANGFUSE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com/")
    monkeypatch.setenv("LANGFUSE_ENVIRONMENT", "production")
    monkeypatch.setenv("LANGFUSE_RELEASE", "modal-v1")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("LANGFUSE_EXPORT_DEFAULT_LIMIT", "25")

    settings = get_settings()

    assert settings.enable_langfuse_observability is True
    assert settings.langfuse_public_key == "pk-lf-test"
    assert settings.langfuse_secret_key == "sk-lf-test"
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.langfuse_environment == "production"
    assert settings.langfuse_release == "modal-v1"
    assert settings.langfuse_sample_rate == 0.25
    assert settings.langfuse_export_default_limit == 25


def test_get_settings_uses_configured_feedback_export_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_PRODUCTION_FEEDBACK_EXPORT", "true")
    monkeypatch.setenv("ALLOW_RAW_PRODUCTION_TEXT_IN_EVALS", "true")

    settings = get_settings()

    assert settings.enable_production_feedback_export is True
    assert settings.allow_raw_production_text_in_evals is True


def test_get_settings_uses_configured_upstash_redis_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_REDIS", "true")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://cache.internal")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "upstash-token")
    monkeypatch.setenv("EXACT_CACHE_ENABLED", "true")
    monkeypatch.setenv("EXACT_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("REQUEST_LOCK_ENABLED", "false")
    monkeypatch.setenv("REQUEST_LOCK_TTL_SECONDS", "45")
    monkeypatch.setenv("RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION", "kb-v2")

    settings = get_settings()

    assert settings.enable_redis is True
    assert settings.upstash_redis_rest_url == "https://cache.internal"
    assert settings.upstash_redis_rest_token == "upstash-token"
    assert settings.redis_healthcheck_enabled is True
    assert settings.exact_cache_enabled is True
    assert settings.exact_cache_ttl_seconds == 3600
    assert settings.request_lock_enabled is False
    assert settings.request_lock_ttl_seconds == 45
    assert settings.response_cache_knowledge_base_version == "kb-v2"


def test_get_settings_uses_configured_fixed_window_rate_limit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_REDIS", "true")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://cache.internal")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "upstash-token")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "12")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "45")

    settings = get_settings()

    assert settings.enable_redis is True
    assert settings.redis_healthcheck_enabled is True
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_max_requests == 12
    assert settings.rate_limit_window_seconds == 45


def test_get_settings_uses_configured_app_and_supabase_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://frontend.example.com, https://admin.example.com")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://postgres:secret@db.example.com:5432/app")
    monkeypatch.setenv("DATABASE_DIRECT_URL", "postgresql+psycopg://postgres:secret@db-direct.example.com:5432/app")
    monkeypatch.setenv("LLM_API_KEY", "prod-key")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "supabase_pgvector")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "knowledge-base")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "mlflow-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "mlflow-pass")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_TOKEN", raising=False)

    settings = get_settings()

    assert settings.app_env == "production"
    assert settings.frontend_origins == [
        "https://frontend.example.com",
        "https://admin.example.com",
    ]
    assert settings.database_direct_url == (
        "postgresql+psycopg://postgres:secret@db-direct.example.com:5432/app"
    )
    assert settings.vector_store_provider == "supabase_pgvector"
    assert settings.supabase_url == "https://project.supabase.co"
    assert settings.supabase_service_role_key == "service-role"
    assert settings.supabase_storage_bucket == "knowledge-base"
    assert settings.supabase_configured is True
    assert settings.redis_url is None
    assert settings.mlflow_tracking_username == "mlflow-user"
    assert settings.mlflow_tracking_password == "mlflow-pass"
    assert settings.migration_database_url == (
        "postgresql+psycopg://postgres:secret@db-direct.example.com:5432/app"
    )


def test_get_settings_uses_configured_storage_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORAGE_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", "custom-storage")
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio.internal:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "custom-user")
    monkeypatch.setenv("MINIO_SECRET_KEY", "custom-pass")
    monkeypatch.setenv("MINIO_BUCKET", "custom-bucket")
    monkeypatch.setenv("MINIO_SECURE", "true")
    monkeypatch.setenv("KNOWLEDGE_UPLOAD_MAX_BYTES", "2048")

    settings = get_settings()

    assert settings.storage_provider == "local"
    assert settings.local_storage_path == "custom-storage"
    assert settings.minio_endpoint == "http://minio.internal:9000"
    assert settings.minio_access_key == "custom-user"
    assert settings.minio_secret_key == "custom-pass"
    assert settings.minio_bucket == "custom-bucket"
    assert settings.minio_secure is True
    assert settings.knowledge_upload_max_bytes == 2048


def test_get_settings_uses_supabase_storage_provider_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORAGE_PROVIDER", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "knowledge-files-prod")

    settings = get_settings()

    assert settings.storage_provider == "supabase"
    assert settings.supabase_url == "https://project.supabase.co"
    assert settings.supabase_service_role_key == "service-role"
    assert settings.supabase_storage_bucket == "knowledge-files-prod"


def test_get_settings_requires_frontend_origin_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://postgres:secret@db.example.com:5432/app")
    monkeypatch.setenv("LLM_API_KEY", "prod-key")
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)

    with pytest.raises(
        ValueError,
        match="FRONTEND_ORIGIN is required when APP_ENV=production.",
    ):
        get_settings()


def test_get_settings_uses_runtime_database_url_for_migrations_when_direct_url_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:secret@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require",
    )
    monkeypatch.setenv("LLM_API_KEY", "prod-key")
    monkeypatch.delenv("DATABASE_DIRECT_URL", raising=False)

    settings = get_settings()

    assert settings.database_direct_url is None
    assert settings.migration_database_url == settings.database_url


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_message"),
    [
        ("CHUNK_SIZE", "0", "CHUNK_SIZE must be greater than 0."),
        ("CHUNK_OVERLAP", "-1", "CHUNK_OVERLAP must be greater than or equal to 0."),
        ("CHUNK_SIZE", "abc", "CHUNK_SIZE must be an integer."),
        ("RETRIEVAL_TOP_K", "0", "RETRIEVAL_TOP_K must be greater than 0."),
        ("RETRIEVAL_TOP_K", "abc", "RETRIEVAL_TOP_K must be an integer."),
        ("QUERY_REWRITE_TEMPERATURE", "-0.1", "QUERY_REWRITE_TEMPERATURE must be greater than or equal to 0."),
        ("QUERY_REWRITE_MAX_TOKENS", "0", "QUERY_REWRITE_MAX_TOKENS must be greater than 0."),
        ("RERANKER_INITIAL_TOP_K", "0", "RERANKER_INITIAL_TOP_K must be greater than 0."),
        ("RERANKER_FINAL_TOP_K", "0", "RERANKER_FINAL_TOP_K must be greater than 0."),
        ("KNOWLEDGE_UPLOAD_MAX_BYTES", "0", "KNOWLEDGE_UPLOAD_MAX_BYTES must be greater than 0."),
        ("RESPONSE_CACHE_TTL_SECONDS", "0", "RESPONSE_CACHE_TTL_SECONDS must be greater than 0."),
        ("RESPONSE_CACHE_MAX_RESULTS", "0", "RESPONSE_CACHE_MAX_RESULTS must be greater than 0."),
        ("CHAT_RATE_LIMIT_REQUESTS_PER_10_MINUTES", "0", "CHAT_RATE_LIMIT_REQUESTS_PER_10_MINUTES must be greater than 0."),
        ("CHAT_RATE_LIMIT_REQUESTS_PER_DAY", "0", "CHAT_RATE_LIMIT_REQUESTS_PER_DAY must be greater than 0."),
        ("CHAT_RATE_LIMIT_CONCURRENT_REQUESTS", "0", "CHAT_RATE_LIMIT_CONCURRENT_REQUESTS must be greater than 0."),
        ("CHAT_RATE_LIMIT_DAILY_TOKEN_BUDGET", "0", "CHAT_RATE_LIMIT_DAILY_TOKEN_BUDGET must be greater than 0."),
        ("CHAT_RATE_LIMIT_DAILY_COST_BUDGET_USD", "-0.1", "CHAT_RATE_LIMIT_DAILY_COST_BUDGET_USD must be greater than or equal to 0."),
        ("LANGFUSE_SAMPLE_RATE", "-0.1", "LANGFUSE_SAMPLE_RATE must be greater than or equal to 0."),
        ("LANGFUSE_EXPORT_DEFAULT_LIMIT", "0", "LANGFUSE_EXPORT_DEFAULT_LIMIT must be greater than 0."),
        ("RATE_LIMIT_MAX_REQUESTS", "0", "RATE_LIMIT_MAX_REQUESTS must be greater than 0."),
        ("RATE_LIMIT_WINDOW_SECONDS", "0", "RATE_LIMIT_WINDOW_SECONDS must be greater than 0."),
        ("EXACT_CACHE_TTL_SECONDS", "0", "EXACT_CACHE_TTL_SECONDS must be greater than 0."),
        ("REQUEST_LOCK_TTL_SECONDS", "0", "REQUEST_LOCK_TTL_SECONDS must be greater than 0."),
    ],
)
def test_get_settings_rejects_invalid_chunking_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    expected_message: str,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=expected_message):
        get_settings()


def test_get_settings_rejects_invalid_retriever_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVER_TYPE", "semantic")

    supported_values = ", ".join(sorted(SUPPORTED_RETRIEVER_TYPES))
    with pytest.raises(ValueError, match=f"RETRIEVER_TYPE must be one of: {supported_values}."):
        get_settings()


def test_get_settings_rejects_invalid_reranker_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_TYPE", "cross_encoder")

    supported_values = ", ".join(sorted(SUPPORTED_RERANKER_TYPES))
    with pytest.raises(ValueError, match=f"RERANKER_TYPE must be one of: {supported_values}."):
        get_settings()


def test_get_settings_rejects_invalid_response_cache_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESPONSE_CACHE_PROVIDER", "memory")

    supported_values = ", ".join(sorted(SUPPORTED_RESPONSE_CACHE_PROVIDERS))
    with pytest.raises(
        ValueError,
        match=f"RESPONSE_CACHE_PROVIDER must be one of: {supported_values}.",
    ):
        get_settings()


def test_get_settings_rejects_invalid_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    supported_values = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
    with pytest.raises(ValueError, match=f"LLM_PROVIDER must be one of: {supported_values}."):
        get_settings()


def test_get_settings_rejects_invalid_app_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "staging")

    supported_values = ", ".join(sorted(SUPPORTED_APP_ENVS))
    with pytest.raises(ValueError, match=f"APP_ENV must be one of: {supported_values}."):
        get_settings()


def test_get_settings_rejects_invalid_vector_store_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "pinecone")

    supported_values = ", ".join(sorted(SUPPORTED_VECTOR_STORE_PROVIDERS))
    with pytest.raises(
        ValueError,
        match=f"VECTOR_STORE_PROVIDER must be one of: {supported_values}.",
    ):
        get_settings()


def test_get_settings_rejects_invalid_storage_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORAGE_PROVIDER", "s3")

    supported_values = ", ".join(sorted(SUPPORTED_STORAGE_PROVIDERS))
    with pytest.raises(
        ValueError,
        match=f"STORAGE_PROVIDER must be one of: {supported_values}.",
    ):
        get_settings()


def test_get_settings_rejects_chunk_overlap_that_is_not_smaller_than_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP must be smaller than CHUNK_SIZE."):
        get_settings()


def test_get_settings_rejects_langfuse_sample_rate_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.1")

    with pytest.raises(
        ValueError,
        match="LANGFUSE_SAMPLE_RATE must be less than or equal to 1.0.",
    ):
        get_settings()


def test_get_settings_rejects_response_cache_distance_threshold_above_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESPONSE_CACHE_DISTANCE_THRESHOLD", "2.1")

    with pytest.raises(
        ValueError,
        match="RESPONSE_CACHE_DISTANCE_THRESHOLD must be less than or equal to 2.0.",
    ):
        get_settings()


def test_get_settings_requires_langfuse_public_key_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_LANGFUSE", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    with pytest.raises(
        ValueError,
        match="LANGFUSE_PUBLIC_KEY is required when ENABLE_LANGFUSE=true.",
    ):
        get_settings()


def test_get_settings_requires_langfuse_secret_key_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_LANGFUSE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="LANGFUSE_SECRET_KEY is required when ENABLE_LANGFUSE=true.",
    ):
        get_settings()


def test_get_settings_requires_database_url_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_API_KEY", "prod-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL is required when APP_ENV=production."):
        get_settings()


def test_get_settings_requires_llm_api_key_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://postgres:secret@db.example.com:5432/app")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="LLM_API_KEY or OPENAI_API_KEY must be set when APP_ENV=production.",
    ):
        get_settings()


def test_get_settings_requires_upstash_credentials_when_redis_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_REDIS", "true")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)

    with pytest.raises(
        ValueError,
        match="UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required when ENABLE_REDIS=true.",
    ):
        get_settings()


def test_get_settings_requires_supabase_credentials_for_supabase_pgvector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "supabase_pgvector")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="SUPABASE_URL is required when VECTOR_STORE_PROVIDER=supabase_pgvector.",
    ):
        get_settings()


def test_get_settings_requires_supabase_credentials_for_supabase_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "pgvector")
    monkeypatch.setenv("STORAGE_PROVIDER", "supabase")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="SUPABASE_URL is required when STORAGE_PROVIDER=supabase.",
    ):
        get_settings()
