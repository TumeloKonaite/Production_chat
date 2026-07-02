from __future__ import annotations

from collections.abc import Generator

import pytest

from app.config import DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENROUTER_BASE_URL, get_settings


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
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/custom/")

    settings = get_settings()

    assert settings.openai_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1/custom"
