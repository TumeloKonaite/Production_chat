from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from app.services.retrieval.errors import EmbeddingConfigurationError


@dataclass(frozen=True, slots=True)
class EmbeddingDescriptor:
    provider: str
    model: str
    dimension: int

    def as_metadata(self) -> dict[str, object]:
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimension": self.dimension,
        }

    def as_config_string(self) -> str:
        return f"{self.provider}/{self.model}/{self.dimension}"


class EmbeddingProvider(Embeddings, ABC):
    provider_name: str

    def __init__(self, *, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return EmbeddingDescriptor(
            provider=self.provider_name,
            model=self.model_name,
            dimension=self.dimension,
        )

    def collection_metadata(self) -> dict[str, object]:
        return self.descriptor.as_metadata()

    def validate_embedding_dimensions(
        self,
        vectors: list[list[float]],
        *,
        operation: str,
    ) -> list[list[float]]:
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimension:
                raise EmbeddingConfigurationError(
                    "Configured embedding dimension does not match the provider output. "
                    f"Configured {self.dimension}, received {len(vector)} during {operation} "
                    f"for {self.provider_name}/{self.model_name} at item {index}."
                )
        return vectors

    def validate_query_dimension(self, vector: list[float]) -> list[float]:
        validated = self.validate_embedding_dimensions([vector], operation="query embedding")
        return validated[0]

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...
