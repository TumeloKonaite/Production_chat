from __future__ import annotations

from app.config import Settings
from app.infrastructure.embeddings.base import EmbeddingProvider
from app.infrastructure.embeddings.hf_provider import HuggingFaceEmbeddingProvider
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.embeddings.openrouter_provider import OpenRouterEmbeddingProvider
from app.infrastructure.embeddings.errors import EmbeddingConfigurationError


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hf":
        return HuggingFaceEmbeddingProvider(
            model_name=settings.knowledge_embedding_model,
            dimension=settings.embedding_dimension,
        )

    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.knowledge_embedding_model,
            dimension=settings.embedding_dimension,
        )

    if settings.embedding_provider == "openrouter":
        return OpenRouterEmbeddingProvider(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model_name=settings.knowledge_embedding_model,
            dimension=settings.embedding_dimension,
        )

    raise EmbeddingConfigurationError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )
