from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv
from app.infrastructure.prompts import normalize_prompt_version

# Load local development settings from `.env` without requiring callers to do it first.
load_dotenv()

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KNOWLEDGE_CHUNK_SIZE = 1000
DEFAULT_KNOWLEDGE_CHUNK_OVERLAP = 200
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "openrouter"})
SUPPORTED_RETRIEVER_TYPES = frozenset({"vector", "keyword", "hybrid"})
SUPPORTED_RERANKER_TYPES = frozenset({"none", "llm"})


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


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


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


def _get_retriever_type_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    value = raw_value.strip().casefold()
    if value not in SUPPORTED_RETRIEVER_TYPES:
        supported_values = ", ".join(sorted(SUPPORTED_RETRIEVER_TYPES))
        raise ValueError(f"{name} must be one of: {supported_values}.")

    return value


def _get_reranker_type_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    value = raw_value.strip().casefold()
    if value not in SUPPORTED_RERANKER_TYPES:
        supported_values = ", ".join(sorted(SUPPORTED_RERANKER_TYPES))
        raise ValueError(f"{name} must be one of: {supported_values}.")

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


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
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

    enable_langfuse_observability = _parse_bool(
        os.getenv("ENABLE_LANGFUSE_OBSERVABILITY"),
        default=False,
    )
    langfuse_public_key = _get_non_empty_env("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = _get_non_empty_env("LANGFUSE_SECRET_KEY")
    if enable_langfuse_observability and langfuse_public_key is None:
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY is required when ENABLE_LANGFUSE_OBSERVABILITY=true."
        )
    if enable_langfuse_observability and langfuse_secret_key is None:
        raise ValueError(
            "LANGFUSE_SECRET_KEY is required when ENABLE_LANGFUSE_OBSERVABILITY=true."
        )
    langfuse_sample_rate = _get_float_env(
        "LANGFUSE_SAMPLE_RATE",
        1.0,
        minimum=0.0,
    )
    if langfuse_sample_rate > 1.0:
        raise ValueError("LANGFUSE_SAMPLE_RATE must be less than or equal to 1.0.")

    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot",
        ),
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
        embedding_dimension=_get_int_env("EMBEDDING_DIMENSION", 384, minimum=1),
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
        retriever_type=_get_retriever_type_env("RETRIEVER_TYPE", "vector"),
        retrieval_top_k=_get_int_env("RETRIEVAL_TOP_K", 5, minimum=1),
        retrieval_min_similarity=float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.55")),
        default_retrieval_config=os.getenv("DEFAULT_RETRIEVAL_CONFIG", "default"),
        enable_mlflow_tracking=_parse_bool(
            os.getenv("ENABLE_MLFLOW_TRACKING"),
            default=True,
        ),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
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
        reranker_type=_get_reranker_type_env("RERANKER_TYPE", "none"),
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
            _get_non_empty_env("LANGFUSE_ENVIRONMENT", default="local") or "local"
        ),
        langfuse_release=_get_non_empty_env("LANGFUSE_RELEASE"),
        langfuse_sample_rate=langfuse_sample_rate,
    )
