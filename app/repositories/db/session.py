from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine(*, use_direct: bool = False) -> Engine:
    settings = get_settings()
    database_url = settings.migration_database_url if use_direct else settings.database_url
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        database_url,
        connect_args=connect_args,
        future=True,
    )


@lru_cache
def get_session_factory(*, use_direct: bool = False) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(use_direct=use_direct),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
