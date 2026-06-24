from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

# Load local development settings from `.env` without requiring callers to do it first.
load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    openai_api_key: str | None
    openai_model: str


@lru_cache
def get_settings() -> Settings:
    # Cache config so dependency injection reuses the same resolved settings object.
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
