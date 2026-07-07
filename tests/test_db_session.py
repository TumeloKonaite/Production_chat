from __future__ import annotations

from collections.abc import Generator

import pytest

from app.config import get_settings
from app.repositories.db.session import get_engine, get_session_factory


@pytest.fixture(autouse=True)
def clear_cached_database_state() -> Generator[None, None, None]:
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    yield
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    get_settings.cache_clear()


def test_get_engine_preserves_sslmode_for_supabase_pooler_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres.project-ref:secret@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require",
    )
    monkeypatch.setenv("LLM_API_KEY", "prod-key")

    engine = get_engine()

    assert engine.url.host == "aws-0-eu-west-1.pooler.supabase.com"
    assert engine.url.port == 6543
    assert engine.url.query["sslmode"] == "require"


def test_get_engine_uses_direct_url_for_admin_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres.project-ref:secret@aws-0-eu-west-1.pooler.supabase.com:6543/postgres?sslmode=require",
    )
    monkeypatch.setenv(
        "DATABASE_DIRECT_URL",
        "postgresql+psycopg://postgres:secret@db.project-ref.supabase.co:5432/postgres?sslmode=require",
    )
    monkeypatch.setenv("LLM_API_KEY", "prod-key")

    engine = get_engine(use_direct=True)

    assert engine.url.host == "db.project-ref.supabase.co"
    assert engine.url.port == 5432
    assert engine.url.query["sslmode"] == "require"
