from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.infrastructure.embeddings.base import EmbeddingProvider


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    provider_name = "hf"

    def __init__(self, *, model_name: str, dimension: int) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self._embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embeddings.embed_documents(texts)
        return self.validate_embedding_dimensions(vectors, operation="document embedding")

    def embed_query(self, text: str) -> list[float]:
        vector = self._embeddings.embed_query(text)
        return self.validate_query_dimension(vector)
