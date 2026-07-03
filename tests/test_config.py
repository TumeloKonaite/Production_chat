from __future__ import annotations

from collections.abc import Generator

import pytest

from app.config import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
    DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    SUPPORTED_RETRIEVER_TYPES,
    get_settings,
)


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


def test_get_settings_uses_default_chunking_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("RETRIEVER_TYPE", raising=False)
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)

    settings = get_settings()

    assert settings.knowledge_chunk_size == DEFAULT_KNOWLEDGE_CHUNK_SIZE
    assert settings.knowledge_chunk_overlap == DEFAULT_KNOWLEDGE_CHUNK_OVERLAP
    assert settings.retriever_type == "vector"
    assert settings.retrieval_top_k == 5


def test_get_settings_uses_configured_chunking_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    settings = get_settings()

    assert settings.knowledge_chunk_size == 500
    assert settings.knowledge_chunk_overlap == 100


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_message"),
    [
        ("CHUNK_SIZE", "0", "CHUNK_SIZE must be greater than 0."),
        ("CHUNK_OVERLAP", "-1", "CHUNK_OVERLAP must be greater than or equal to 0."),
        ("CHUNK_SIZE", "abc", "CHUNK_SIZE must be an integer."),
        ("RETRIEVAL_TOP_K", "0", "RETRIEVAL_TOP_K must be greater than 0."),
        ("RETRIEVAL_TOP_K", "abc", "RETRIEVAL_TOP_K must be an integer."),
    ],
)
def test_get_settings_rejects_invalid_chunking_values(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    expected_message: str,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=expected_message):
        get_settings()


def test_get_settings_rejects_invalid_retriever_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVER_TYPE", "semantic")

    supported_values = ", ".join(sorted(SUPPORTED_RETRIEVER_TYPES))
    with pytest.raises(ValueError, match=f"RETRIEVER_TYPE must be one of: {supported_values}."):
        get_settings()


def test_get_settings_rejects_chunk_overlap_that_is_not_smaller_than_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP must be smaller than CHUNK_SIZE."):
        get_settings()
