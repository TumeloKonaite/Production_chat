from dataclasses import dataclass
from functools import lru_cache
import os
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv

from app.infrastructure.prompts import normalize_prompt_version

# Load local development settings from `.env` without requiring callers to do it first.
load_dotenv()

DEFAULT_APP_ENV = "local"
DEFAULT_LOCAL_FRONTEND_ORIGIN = "http://localhost:5173"
DEFAULT_LOCAL_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot"
)
DEFAULT_LOCAL_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KNOWLEDGE_CHUNK_SIZE = 1000
DEFAULT_KNOWLEDGE_CHUNK_OVERLAP = 200
SUPPORTED_APP_ENVS = frozenset({"local", "production", "test"})
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "openrouter"})
SUPPORTED_RETRIEVER_TYPES = frozenset({"vector", "keyword", "hybrid"})
SUPPORTED_RERANKER_TYPES = frozenset({"none", "llm"})
SUPPORTED_RESPONSE_CACHE_PROVIDERS = frozenset({"redis"})
SUPPORTED_STORAGE_PROVIDERS = frozenset({"local", "minio", "supabase"})
SUPPORTED_VECTOR_STORE_PROVIDERS = frozenset({"pgvector", "supabase_pgvector"})


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    openai_api_key: str | None
    openai_base_url: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    tavus_api_key: str | None
    tavus_base_url: str
    tavus_face_id: str | None
    tavus_pal_id: str | None
    public_backend_url: str | None
    tavus_tool_secret: str | None
    ingestion_api_secret: str | None
    eval_admin_token: str | None
    default_model_config_id: str
    model_configs_json: str | None
    embedding_provider: str
    knowledge_embedding_model: str
    embedding_dimension: int
    knowledge_collection_name: str
    default_prompt_version: str
    conversation_history_limit: int
    retriever_type: str
    retrieval_top_k: int
    retrieval_min_similarity: float
    default_retrieval_config: str
    enable_mlflow_tracking: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment_name: str
    enable_dagshub_tracking: bool
    dagshub_repo_owner: str | None
    dagshub_repo_name: str | None
    dagshub_token: str | None
    app_env: str = DEFAULT_APP_ENV
    frontend_origin: str | None = None
    database_direct_url: str | None = None
    vector_store_provider: str = "pgvector"
    storage_provider: str = "minio"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "knowledge-files"
    minio_secure: bool = False
    local_storage_path: str = ".storage"
    knowledge_upload_max_bytes: int = 10485760
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_storage_bucket: str | None = "knowledge-files"
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = DEFAULT_OPENAI_BASE_URL
    llm_api_key: str | None = None
    llm_prompt_cost_per_1m_tokens: float | None = None
    llm_completion_cost_per_1m_tokens: float | None = None
    knowledge_chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE
    knowledge_chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP
    enable_query_rewriting: bool = False
    query_rewrite_model: str = "openai:gpt-4.1-mini"
    query_rewrite_temperature: float = 0.0
    query_rewrite_prompt_version: str = "v1"
    query_rewrite_timeout_seconds: int = 10
    query_rewrite_max_tokens: int = 128
    enable_reranking: bool = False
    reranker_type: str = "none"
    reranker_model: str = "openai:gpt-4.1-mini"
    reranker_initial_top_k: int = 20
    reranker_final_top_k: int = 5
    enable_langfuse_observability: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "local"
    langfuse_release: str | None = None
    langfuse_sample_rate: float = 1.0
    langfuse_export_default_limit: int = 100
    enable_production_feedback_export: bool = False
    allow_raw_production_text_in_evals: bool = False
    enable_response_cache: bool = False
    response_cache_provider: str = "redis"
    redis_url: str | None = None
    redis_token: str | None = None
    enable_exact_response_cache: bool = True
    enable_semantic_response_cache: bool = False
    response_cache_ttl_seconds: int = 604800
    response_cache_exact_prefix: str = "chat:exact"
    response_cache_semantic_index: str = "chat_semantic_cache"
    response_cache_distance_threshold: float = 0.10
    response_cache_max_results: int = 3
    response_cache_store_private_sessions: bool = False
    response_cache_knowledge_base_version: str = "default"
    enable_rate_limiting: bool = False
    rate_limiting_fail_open: bool = True
    chat_rate_limit_requests_per_10_minutes: int = 20
    chat_rate_limit_requests_per_day: int = 100
    chat_rate_limit_concurrent_requests: int = 3
    chat_rate_limit_daily_token_budget: int = 100000
    chat_rate_limit_daily_cost_budget_usd: float = 0.50

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def frontend_origins(self) -> list[str]:
        configured = self.frontend_origin
        if configured is None and self.app_env != "production":
            configured = DEFAULT_LOCAL_FRONTEND_ORIGIN
        if configured is None:
            return []
        return [value.strip() for value in configured.split(",") if value.strip()]

    @property
    def resolved_redis_url(self) -> str | None:
        if self.redis_url is None:
            return None
        return _inject_redis_token(self.redis_url, self.redis_token)

    @property
    def redis_configured(self) -> bool:
        return self.resolved_redis_url is not None

    @property
    def redis_healthcheck_enabled(self) -> bool:
        return self.enable_response_cache or self.enable_rate_limiting

    @property
    def migration_database_url(self) -> str:
        return self.database_direct_url or self.database_url

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def openai_base_url_configured(self) -> bool:
        return bool(self.llm_base_url.strip())


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _get_first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


def _get_non_empty_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _get_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if minimum is not None and value < minimum:
        comparator = "greater than 0" if minimum == 1 else f"greater than or equal to {minimum}"
        raise ValueError(f"{name} must be {comparator}.")

    return value


def _get_int_env_from_names(
    names: tuple[str, ...],
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            continue
        return _get_int_env(name, default, minimum=minimum)
    return default


def _get_float_env(name: str, default: float, *, minimum: float | None = None) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if minimum is not None and value < minimum:
        comparator = "greater than or equal to 0" if minimum == 0 else f"greater than or equal to {minimum}"
        raise ValueError(f"{name} must be {comparator}.")

    return value


def _get_optional_float_env(name: str, *, minimum: float | None = None) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if minimum is not None and value < minimum:
        comparator = "greater than or equal to 0" if minimum == 0 else f"greater than or equal to {minimum}"
        raise ValueError(f"{name} must be {comparator}.")

    return value


def _get_choice_env(name: str, default: str, *, supported_values: frozenset[str]) -> str:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    value = raw_value.strip().casefold()
    if value not in supported_values:
        supported = ", ".join(sorted(supported_values))
        raise ValueError(f"{name} must be one of: {supported}.")
    return value


def _extract_provider(model_config_id: str | None) -> str | None:
    if model_config_id is None or ":" not in model_config_id:
        return None
    provider, _model = model_config_id.split(":", 1)
    return provider.strip().casefold() or None


def _extract_model_name(model_config_id: str | None) -> str | None:
    if model_config_id is None or ":" not in model_config_id:
        return None
    _provider, model = model_config_id.split(":", 1)
    return model.strip() or None


def _normalize_model_config_id(provider: str, model: str, configured_model_config_id: str | None) -> str:
    if configured_model_config_id and ":" not in configured_model_config_id:
        return f"openai:{configured_model_config_id}"
    if configured_model_config_id is not None:
        return configured_model_config_id
    return f"{provider}:{model}"


def _get_llm_provider(default_model_config_id: str | None) -> str:
    provider = (
        _get_non_empty_env("LLM_PROVIDER")
        or _extract_provider(default_model_config_id)
        or "openai"
    ).casefold()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        supported_values = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ValueError(f"LLM_PROVIDER must be one of: {supported_values}.")
    return provider


def _inject_redis_token(redis_url: str, redis_token: str | None) -> str:
    if redis_token is None or "@" in redis_url:
        return redis_url

    parsed = urlsplit(redis_url)
    if not parsed.scheme or not parsed.hostname:
        return redis_url

    username = parsed.username or "default"
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{username}:{quote(redis_token, safe='')}@{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _validate_production_requirements(
    *,
    app_env: str,
    frontend_origin: str | None,
    database_url: str | None,
    llm_api_key: str | None,
) -> None:
    if app_env != "production":
        return
    if frontend_origin is None:
        raise ValueError("FRONTEND_ORIGIN is required when APP_ENV=production.")
    if database_url is None:
        raise ValueError("DATABASE_URL is required when APP_ENV=production.")
    if llm_api_key is None:
        raise ValueError(
            "LLM_API_KEY or OPENAI_API_KEY must be set when APP_ENV=production."
        )


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
    app_env = _get_choice_env(
        "APP_ENV",
        DEFAULT_APP_ENV,
        supported_values=SUPPORTED_APP_ENVS,
    )
    configured_model_config_id = _get_non_empty_env("DEFAULT_MODEL_CONFIG_ID")
    configured_llm_provider = _get_non_empty_env("LLM_PROVIDER")
    configured_llm_model = _get_non_empty_env("LLM_MODEL")
    llm_provider = (
        configured_llm_provider.casefold()
        if configured_llm_provider is not None
        else _get_llm_provider(configured_model_config_id)
    )
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        supported_values = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ValueError(f"LLM_PROVIDER must be one of: {supported_values}.")
    llm_model = (
        configured_llm_model
        or _extract_model_name(configured_model_config_id)
        or _get_non_empty_env("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )
    default_model_config_id = (
        f"{llm_provider}:{llm_model}"
        if configured_llm_provider is not None or configured_llm_model is not None
        else _normalize_model_config_id(
            llm_provider,
            llm_model,
            configured_model_config_id,
        )
    )

    llm_base_url_override = _get_non_empty_env("LLM_BASE_URL")
    openai_base_url = (
        llm_base_url_override
        if llm_provider == "openai" and llm_base_url_override is not None
        else (
            _get_non_empty_env("OPENAI_BASE_URL", default=DEFAULT_OPENAI_BASE_URL)
            or DEFAULT_OPENAI_BASE_URL
        )
    ).rstrip("/")
    openrouter_base_url = (
        llm_base_url_override
        if llm_provider == "openrouter" and llm_base_url_override is not None
        else (
            _get_non_empty_env("OPENROUTER_BASE_URL", default=DEFAULT_OPENROUTER_BASE_URL)
            or DEFAULT_OPENROUTER_BASE_URL
        )
    ).rstrip("/")

    llm_api_key_override = _get_non_empty_env("LLM_API_KEY")
    openai_api_key = (
        llm_api_key_override
        if llm_provider == "openai" and llm_api_key_override is not None
        else _get_non_empty_env("OPENAI_API_KEY")
    )
    openrouter_api_key = (
        llm_api_key_override
        if llm_provider == "openrouter" and llm_api_key_override is not None
        else _get_non_empty_env("OPENROUTER_API_KEY")
    )

    llm_base_url = openai_base_url if llm_provider == "openai" else openrouter_base_url
    llm_api_key = openai_api_key if llm_provider == "openai" else openrouter_api_key

    database_url = _get_non_empty_env("DATABASE_URL")
    if database_url is None and app_env != "production":
        database_url = DEFAULT_LOCAL_DATABASE_URL
    frontend_origin = _get_non_empty_env("FRONTEND_ORIGIN")
    _validate_production_requirements(
        app_env=app_env,
        frontend_origin=frontend_origin,
        database_url=database_url,
        llm_api_key=llm_api_key,
    )

    knowledge_chunk_size = _get_int_env(
        "CHUNK_SIZE",
        DEFAULT_KNOWLEDGE_CHUNK_SIZE,
        minimum=1,
    )
    knowledge_chunk_overlap = _get_int_env(
        "CHUNK_OVERLAP",
        DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
        minimum=0,
    )
    if knowledge_chunk_overlap >= knowledge_chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    vector_store_provider = _get_choice_env(
        "VECTOR_STORE_PROVIDER",
        "pgvector",
        supported_values=SUPPORTED_VECTOR_STORE_PROVIDERS,
    )
    storage_provider = _get_choice_env(
        "STORAGE_PROVIDER",
        "minio",
        supported_values=SUPPORTED_STORAGE_PROVIDERS,
    )
    supabase_url = _get_non_empty_env("SUPABASE_URL")
    supabase_service_role_key = _get_non_empty_env("SUPABASE_SERVICE_ROLE_KEY")
    if vector_store_provider == "supabase_pgvector" and supabase_url is None:
        raise ValueError(
            "SUPABASE_URL is required when VECTOR_STORE_PROVIDER=supabase_pgvector."
        )
    if vector_store_provider == "supabase_pgvector" and supabase_service_role_key is None:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY is required when VECTOR_STORE_PROVIDER=supabase_pgvector."
        )
    if storage_provider == "supabase" and supabase_url is None:
        raise ValueError("SUPABASE_URL is required when STORAGE_PROVIDER=supabase.")
    if storage_provider == "supabase" and supabase_service_role_key is None:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY is required when STORAGE_PROVIDER=supabase."
        )

    enable_langfuse_observability = _parse_bool(
        _get_first_env("ENABLE_LANGFUSE", "ENABLE_LANGFUSE_OBSERVABILITY"),
        default=False,
    )
    langfuse_public_key = _get_non_empty_env("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = _get_non_empty_env("LANGFUSE_SECRET_KEY")
    if enable_langfuse_observability and langfuse_public_key is None:
        raise ValueError("LANGFUSE_PUBLIC_KEY is required when ENABLE_LANGFUSE=true.")
    if enable_langfuse_observability and langfuse_secret_key is None:
        raise ValueError("LANGFUSE_SECRET_KEY is required when ENABLE_LANGFUSE=true.")
    langfuse_sample_rate = _get_float_env(
        "LANGFUSE_SAMPLE_RATE",
        1.0,
        minimum=0.0,
    )
    if langfuse_sample_rate > 1.0:
        raise ValueError("LANGFUSE_SAMPLE_RATE must be less than or equal to 1.0.")
    langfuse_export_default_limit = _get_int_env(
        "LANGFUSE_EXPORT_DEFAULT_LIMIT",
        100,
        minimum=1,
    )
    enable_production_feedback_export = _parse_bool(
        os.getenv("ENABLE_PRODUCTION_FEEDBACK_EXPORT"),
        default=False,
    )
    allow_raw_production_text_in_evals = _parse_bool(
        os.getenv("ALLOW_RAW_PRODUCTION_TEXT_IN_EVALS"),
        default=False,
    )
    enable_response_cache = _parse_bool(
        os.getenv("ENABLE_RESPONSE_CACHE"),
        default=False,
    )
    response_cache_provider = _get_choice_env(
        "RESPONSE_CACHE_PROVIDER",
        "redis",
        supported_values=SUPPORTED_RESPONSE_CACHE_PROVIDERS,
    )
    response_cache_ttl_seconds = _get_int_env(
        "RESPONSE_CACHE_TTL_SECONDS",
        604800,
        minimum=1,
    )
    response_cache_distance_threshold = _get_float_env(
        "RESPONSE_CACHE_DISTANCE_THRESHOLD",
        0.10,
        minimum=0.0,
    )
    if response_cache_distance_threshold > 2.0:
        raise ValueError("RESPONSE_CACHE_DISTANCE_THRESHOLD must be less than or equal to 2.0.")
    response_cache_max_results = _get_int_env(
        "RESPONSE_CACHE_MAX_RESULTS",
        3,
        minimum=1,
    )
    response_cache_store_private_sessions = _parse_bool(
        os.getenv("RESPONSE_CACHE_STORE_PRIVATE_SESSIONS"),
        default=False,
    )
    enable_rate_limiting = _parse_bool(
        os.getenv("ENABLE_RATE_LIMITING"),
        default=False,
    )
    rate_limiting_fail_open = _parse_bool(
        os.getenv("RATE_LIMITING_FAIL_OPEN"),
        default=True,
    )
    chat_rate_limit_requests_per_10_minutes = _get_int_env(
        "CHAT_RATE_LIMIT_REQUESTS_PER_10_MINUTES",
        20,
        minimum=1,
    )
    chat_rate_limit_requests_per_day = _get_int_env(
        "CHAT_RATE_LIMIT_REQUESTS_PER_DAY",
        100,
        minimum=1,
    )
    chat_rate_limit_concurrent_requests = _get_int_env(
        "CHAT_RATE_LIMIT_CONCURRENT_REQUESTS",
        3,
        minimum=1,
    )
    chat_rate_limit_daily_token_budget = _get_int_env(
        "CHAT_RATE_LIMIT_DAILY_TOKEN_BUDGET",
        100000,
        minimum=1,
    )
    chat_rate_limit_daily_cost_budget_usd = _get_float_env(
        "CHAT_RATE_LIMIT_DAILY_COST_BUDGET_USD",
        0.50,
        minimum=0.0,
    )

    redis_url = _get_non_empty_env("REDIS_URL")
    if redis_url is None and app_env != "production":
        redis_url = DEFAULT_LOCAL_REDIS_URL

    return Settings(
        app_env=app_env,
        frontend_origin=frontend_origin,
        database_url=database_url or DEFAULT_LOCAL_DATABASE_URL,
        database_direct_url=_get_non_empty_env("DATABASE_DIRECT_URL"),
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        tavus_api_key=os.getenv("TAVUS_API_KEY"),
        tavus_base_url=os.getenv("TAVUS_BASE_URL", "https://tavusapi.com"),
        tavus_face_id=os.getenv("TAVUS_FACE_ID"),
        tavus_pal_id=os.getenv("TAVUS_PAL_ID"),
        public_backend_url=os.getenv("PUBLIC_BACKEND_URL"),
        tavus_tool_secret=os.getenv("TAVUS_TOOL_SECRET"),
        ingestion_api_secret=os.getenv("INGESTION_API_SECRET"),
        eval_admin_token=_get_non_empty_env("EVAL_ADMIN_TOKEN"),
        default_model_config_id=default_model_config_id,
        model_configs_json=_get_non_empty_env("MODEL_CONFIGS_JSON"),
        embedding_provider=(
            _get_non_empty_env("EMBEDDING_PROVIDER", default="hf") or "hf"
        ).strip().casefold(),
        knowledge_embedding_model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimension=_get_int_env_from_names(
            ("KNOWLEDGE_EMBEDDING_DIMENSION", "EMBEDDING_DIMENSION"),
            384,
            minimum=1,
        ),
        knowledge_collection_name=os.getenv("KNOWLEDGE_COLLECTION_NAME", "personal_knowledge_base"),
        knowledge_chunk_size=knowledge_chunk_size,
        knowledge_chunk_overlap=knowledge_chunk_overlap,
        default_prompt_version=normalize_prompt_version(
            os.getenv(
                "DEFAULT_PROMPT_VERSION",
                os.getenv("PROMPT_VERSION", "v1_professional"),
            )
        ),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10")),
        retriever_type=_get_choice_env(
            "RETRIEVER_TYPE",
            "vector",
            supported_values=SUPPORTED_RETRIEVER_TYPES,
        ),
        retrieval_top_k=_get_int_env("RETRIEVAL_TOP_K", 5, minimum=1),
        retrieval_min_similarity=float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.55")),
        default_retrieval_config=os.getenv("DEFAULT_RETRIEVAL_CONFIG", "default"),
        vector_store_provider=vector_store_provider,
        storage_provider=storage_provider,
        minio_endpoint=(
            _get_non_empty_env("MINIO_ENDPOINT", default="http://localhost:9000")
            or "http://localhost:9000"
        ),
        minio_access_key=(
            _get_non_empty_env("MINIO_ACCESS_KEY", default="minioadmin")
            or "minioadmin"
        ),
        minio_secret_key=(
            _get_non_empty_env("MINIO_SECRET_KEY", default="minioadmin")
            or "minioadmin"
        ),
        minio_bucket=(
            _get_non_empty_env("MINIO_BUCKET", default="knowledge-files")
            or "knowledge-files"
        ),
        minio_secure=_parse_bool(os.getenv("MINIO_SECURE"), default=False),
        local_storage_path=(
            _get_non_empty_env("LOCAL_STORAGE_PATH", default=".storage")
            or ".storage"
        ),
        knowledge_upload_max_bytes=_get_int_env(
            "KNOWLEDGE_UPLOAD_MAX_BYTES",
            10485760,
            minimum=1,
        ),
        supabase_url=supabase_url,
        supabase_service_role_key=supabase_service_role_key,
        supabase_storage_bucket=(
            _get_non_empty_env("SUPABASE_STORAGE_BUCKET", default="knowledge-files")
            or "knowledge-files"
        ),
        enable_mlflow_tracking=_parse_bool(
            os.getenv("ENABLE_MLFLOW_TRACKING"),
            default=False,
        ),
        mlflow_tracking_uri=_get_non_empty_env("MLFLOW_TRACKING_URI"),
        mlflow_tracking_username=_get_non_empty_env("MLFLOW_TRACKING_USERNAME"),
        mlflow_tracking_password=_get_non_empty_env("MLFLOW_TRACKING_PASSWORD"),
        mlflow_experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "personal-chatbot-model-comparison",
        ),
        enable_dagshub_tracking=_parse_bool(
            os.getenv("ENABLE_DAGSHUB_TRACKING"),
            default=False,
        ),
        dagshub_repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
        dagshub_repo_name=os.getenv("DAGSHUB_REPO_NAME"),
        dagshub_token=os.getenv("DAGSHUB_TOKEN"),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_prompt_cost_per_1m_tokens=_get_optional_float_env(
            "LLM_PROMPT_COST_PER_1M_TOKENS",
            minimum=0.0,
        ),
        llm_completion_cost_per_1m_tokens=_get_optional_float_env(
            "LLM_COMPLETION_COST_PER_1M_TOKENS",
            minimum=0.0,
        ),
        enable_query_rewriting=_parse_bool(
            os.getenv("ENABLE_QUERY_REWRITING"),
            default=False,
        ),
        query_rewrite_model=(
            _get_non_empty_env("QUERY_REWRITE_MODEL", default="openai:gpt-4.1-mini")
            or "openai:gpt-4.1-mini"
        ),
        query_rewrite_temperature=_get_float_env(
            "QUERY_REWRITE_TEMPERATURE",
            0.0,
            minimum=0.0,
        ),
        query_rewrite_prompt_version=_get_non_empty_env(
            "QUERY_REWRITE_PROMPT_VERSION",
            default="v1",
        )
        or "v1",
        query_rewrite_timeout_seconds=_get_int_env(
            "QUERY_REWRITE_TIMEOUT_SECONDS",
            10,
            minimum=1,
        ),
        query_rewrite_max_tokens=_get_int_env(
            "QUERY_REWRITE_MAX_TOKENS",
            128,
            minimum=1,
        ),
        enable_reranking=_parse_bool(
            os.getenv("ENABLE_RERANKING"),
            default=False,
        ),
        reranker_type=_get_choice_env(
            "RERANKER_TYPE",
            "none",
            supported_values=SUPPORTED_RERANKER_TYPES,
        ),
        reranker_model=(
            _get_non_empty_env("RERANKER_MODEL", default="openai:gpt-4.1-mini")
            or "openai:gpt-4.1-mini"
        ),
        reranker_initial_top_k=_get_int_env(
            "RERANKER_INITIAL_TOP_K",
            20,
            minimum=1,
        ),
        reranker_final_top_k=_get_int_env(
            "RERANKER_FINAL_TOP_K",
            5,
            minimum=1,
        ),
        enable_langfuse_observability=enable_langfuse_observability,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_base_url=(
            _get_non_empty_env("LANGFUSE_BASE_URL", default="https://cloud.langfuse.com")
            or "https://cloud.langfuse.com"
        ).rstrip("/"),
        langfuse_environment=(
            _get_non_empty_env("LANGFUSE_ENVIRONMENT", default=app_env) or app_env
        ),
        langfuse_release=_get_non_empty_env("LANGFUSE_RELEASE"),
        langfuse_sample_rate=langfuse_sample_rate,
        langfuse_export_default_limit=langfuse_export_default_limit,
        enable_production_feedback_export=enable_production_feedback_export,
        allow_raw_production_text_in_evals=allow_raw_production_text_in_evals,
        enable_response_cache=enable_response_cache,
        response_cache_provider=response_cache_provider,
        redis_url=redis_url,
        redis_token=_get_non_empty_env("REDIS_TOKEN"),
        enable_exact_response_cache=_parse_bool(
            os.getenv("ENABLE_EXACT_RESPONSE_CACHE"),
            default=True,
        ),
        enable_semantic_response_cache=_parse_bool(
            os.getenv("ENABLE_SEMANTIC_RESPONSE_CACHE"),
            default=False,
        ),
        response_cache_ttl_seconds=response_cache_ttl_seconds,
        response_cache_exact_prefix=(
            _get_non_empty_env("RESPONSE_CACHE_EXACT_PREFIX", default="chat:exact")
            or "chat:exact"
        ),
        response_cache_semantic_index=(
            _get_non_empty_env(
                "RESPONSE_CACHE_SEMANTIC_INDEX",
                default="chat_semantic_cache",
            )
            or "chat_semantic_cache"
        ),
        response_cache_distance_threshold=response_cache_distance_threshold,
        response_cache_max_results=response_cache_max_results,
        response_cache_store_private_sessions=response_cache_store_private_sessions,
        response_cache_knowledge_base_version=(
            _get_non_empty_env(
                "RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION",
                default=os.getenv("KNOWLEDGE_COLLECTION_NAME", "personal_knowledge_base"),
            )
            or os.getenv("KNOWLEDGE_COLLECTION_NAME", "personal_knowledge_base")
        ),
        enable_rate_limiting=enable_rate_limiting,
        rate_limiting_fail_open=rate_limiting_fail_open,
        chat_rate_limit_requests_per_10_minutes=chat_rate_limit_requests_per_10_minutes,
        chat_rate_limit_requests_per_day=chat_rate_limit_requests_per_day,
        chat_rate_limit_concurrent_requests=chat_rate_limit_concurrent_requests,
        chat_rate_limit_daily_token_budget=chat_rate_limit_daily_token_budget,
        chat_rate_limit_daily_cost_budget_usd=chat_rate_limit_daily_cost_budget_usd,
    )
