from __future__ import annotations

from collections.abc import Callable

from app.services.retrieval.strategies.base import normalize_query, resolve_top_k
from app.services.retrieval.types import RetrievedChunk


class VectorRetriever:
    def __init__(
        self,
        *,
        default_top_k: int,
        search: Callable[[str, int], list[RetrievedChunk]],
    ) -> None:
        self._default_top_k = default_top_k
        self._search = search

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        normalized_query = normalize_query(query)
        if not normalized_query:
            return []

        limit = resolve_top_k(top_k, default_top_k=self._default_top_k)
        return self._search(normalized_query, limit)
