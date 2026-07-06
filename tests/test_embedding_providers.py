from __future__ import annotations

import json

import httpx

from app.infrastructure.embeddings import OpenAIEmbeddingProvider, OpenRouterEmbeddingProvider
from app.services.retrieval import EmbeddingConfigurationError


def test_openai_embedding_provider_uses_openai_embeddings_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )

    provider = OpenAIEmbeddingProvider(
        api_key="openai-test-key",
        base_url="https://api.openai.com/v1",
        model_name="text-embedding-3-small",
        dimension=3,
        transport=httpx.MockTransport(handler),
    )

    vectors = provider.embed_documents(["first", "second"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["authorization"] == "Bearer openai-test-key"
    assert captured["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }


def test_openai_embedding_provider_requires_api_key() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        base_url="https://api.openai.com/v1",
        model_name="text-embedding-3-small",
        dimension=3,
    )

    try:
        provider.embed_query("hello")
    except EmbeddingConfigurationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected EmbeddingConfigurationError")

    assert "OPENAI_API_KEY must be set" in message


def test_openrouter_embedding_provider_uses_openai_compatible_embeddings_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )

    provider = OpenRouterEmbeddingProvider(
        api_key="openrouter-test-key",
        base_url="https://openrouter.ai/api/v1",
        model_name="openai/text-embedding-3-small",
        dimension=3,
        transport=httpx.MockTransport(handler),
    )

    vectors = provider.embed_documents(["first", "second"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert captured["authorization"] == "Bearer openrouter-test-key"
    assert captured["payload"] == {
        "model": "openai/text-embedding-3-small",
        "input": ["first", "second"],
    }


def test_openrouter_embedding_provider_requires_api_key() -> None:
    provider = OpenRouterEmbeddingProvider(
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        model_name="openai/text-embedding-3-small",
        dimension=3,
    )

    try:
        provider.embed_query("hello")
    except EmbeddingConfigurationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected EmbeddingConfigurationError")

    assert "OPENROUTER_API_KEY must be set" in message
