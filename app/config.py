from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

# Load local development settings from `.env` without requiring callers to do it first.
load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    openai_api_key: str | None
    openai_model: str
    knowledge_embedding_model: str
    knowledge_collection_name: str
    prompt_version: str
    conversation_history_limit: int
    retrieval_top_k: int
    retrieval_min_similarity: float


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot",
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        knowledge_embedding_model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        knowledge_collection_name=os.getenv("KNOWLEDGE_COLLECTION_NAME", "personal_knowledge_base"),
        prompt_version=os.getenv("PROMPT_VERSION", "v1"),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        retrieval_min_similarity=float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.55")),
    )
