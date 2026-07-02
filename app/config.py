from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv
from app.infrastructure.prompts import normalize_prompt_version

# Load local development settings from `.env` without requiring callers to do it first.
load_dotenv()

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
    default_model_config_id: str
    model_configs_json: str | None
    knowledge_embedding_model: str
    knowledge_collection_name: str
    default_prompt_version: str
    conversation_history_limit: int
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


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
    configured_model = os.getenv("DEFAULT_MODEL_CONFIG_ID")
    if not configured_model:
        openai_model = os.getenv("OPENAI_MODEL")
        configured_model = (
            f"openai:{openai_model}" if openai_model and ":" not in openai_model else openai_model
        )

    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot",
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=(
            _get_non_empty_env("OPENAI_BASE_URL", "LLM_BASE_URL", default=DEFAULT_OPENAI_BASE_URL)
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=(
            _get_non_empty_env("OPENROUTER_BASE_URL", default=DEFAULT_OPENROUTER_BASE_URL)
            or DEFAULT_OPENROUTER_BASE_URL
        ).rstrip("/"),
        tavus_api_key=os.getenv("TAVUS_API_KEY"),
        tavus_base_url=os.getenv("TAVUS_BASE_URL", "https://tavusapi.com"),
        tavus_face_id=os.getenv("TAVUS_FACE_ID"),
        tavus_pal_id=os.getenv("TAVUS_PAL_ID"),
        public_backend_url=os.getenv("PUBLIC_BACKEND_URL"),
        tavus_tool_secret=os.getenv("TAVUS_TOOL_SECRET"),
        ingestion_api_secret=os.getenv("INGESTION_API_SECRET"),
        default_model_config_id=configured_model or "openai:gpt-4.1-mini",
        model_configs_json=_get_non_empty_env("MODEL_CONFIGS_JSON"),
        knowledge_embedding_model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        knowledge_collection_name=os.getenv("KNOWLEDGE_COLLECTION_NAME", "personal_knowledge_base"),
        default_prompt_version=normalize_prompt_version(
            os.getenv(
                "DEFAULT_PROMPT_VERSION",
                os.getenv("PROMPT_VERSION", "v1_professional"),
            )
        ),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
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
    )
