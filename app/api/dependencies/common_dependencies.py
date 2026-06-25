from collections.abc import Generator

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.repositories.db.session import get_db_session as get_session_dependency


def get_app_settings() -> Settings:
    return get_settings()


def get_db_session() -> Generator[Session, None, None]:
    yield from get_session_dependency()
