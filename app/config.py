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
    prompt_version: str
    conversation_history_limit: int


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/production_chatbot",
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        prompt_version=os.getenv("PROMPT_VERSION", "v1"),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10")),
    )
