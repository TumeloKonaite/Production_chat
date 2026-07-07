from __future__ import annotations

from typing import Any

import httpx

from app.infrastructure.embeddings.base import EmbeddingProvider
from app.infrastructure.embeddings.errors import EmbeddingConfigurationError

OPENROUTER_EMBEDDING_TIMEOUT_SECONDS = 60.0


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model_name: str,
        dimension: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = self._request_embeddings(inputs=texts)
        vectors = self._extract_embeddings(payload, expected_count=len(texts))
        return self.validate_embedding_dimensions(vectors, operation="document embedding")

    def embed_query(self, text: str) -> list[float]:
        payload = self._request_embeddings(inputs=[text])
        vectors = self._extract_embeddings(payload, expected_count=1)
        return self.validate_query_dimension(vectors[0])

    def _request_embeddings(self, *, inputs: list[str]) -> dict[str, Any]:
        api_key = self._get_api_key()
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=OPENROUTER_EMBEDDING_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = client.post(
                    "/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "input": inputs,
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingConfigurationError(
                f"Unable to fetch embeddings from OpenRouter for model {self.model_name}."
            ) from exc
        except ValueError as exc:
            raise EmbeddingConfigurationError(
                f"Received an invalid embedding response from OpenRouter for model {self.model_name}."
            ) from exc

    def _extract_embeddings(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        raw_data = payload.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != expected_count:
            raise EmbeddingConfigurationError(
                "OpenRouter returned an unexpected number of embeddings for "
                f"{self.model_name}: expected {expected_count}."
            )

        ordered_rows = sorted(
            (row for row in raw_data if isinstance(row, dict)),
            key=lambda row: int(row.get("index", 0)),
        )
        vectors: list[list[float]] = []
        for row in ordered_rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not all(
                isinstance(value, (int, float)) for value in embedding
            ):
                raise EmbeddingConfigurationError(
                    f"OpenRouter returned an invalid embedding payload for {self.model_name}."
                )
            vectors.append([float(value) for value in embedding])

        if len(vectors) != expected_count:
            raise EmbeddingConfigurationError(
                "OpenRouter embedding response could not be normalized for "
                f"{self.model_name}: expected {expected_count} vectors."
            )

        return vectors

    def _get_api_key(self) -> str:
        if not self._api_key:
            raise EmbeddingConfigurationError(
                "OPENROUTER_API_KEY must be set when EMBEDDING_PROVIDER=openrouter."
            )
        return self._api_key
