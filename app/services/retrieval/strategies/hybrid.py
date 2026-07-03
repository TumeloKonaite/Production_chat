from __future__ import annotations

from app.services.retrieval.strategies.base import normalize_query, resolve_top_k
from app.services.retrieval.types import RetrievedChunk, Retriever

_HYBRID_SCORE_NORMALIZER = 2.0


class HybridRetriever:
    def __init__(
        self,
        *,
        default_top_k: int,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
    ) -> None:
        self._default_top_k = default_top_k
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        normalized_query = normalize_query(query)
        if not normalized_query:
            return []

        limit = resolve_top_k(top_k, default_top_k=self._default_top_k)
        vector_results = self._vector_retriever.retrieve(normalized_query, top_k=limit)
        keyword_results = self._keyword_retriever.retrieve(normalized_query, top_k=limit)

        combined_scores: dict[str, float] = {}
        combined_chunks: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            combined_scores[chunk.id] = combined_scores.get(chunk.id, 0.0) + (1.0 / rank)
            combined_chunks.setdefault(chunk.id, chunk)

        for rank, chunk in enumerate(keyword_results, start=1):
            combined_scores[chunk.id] = combined_scores.get(chunk.id, 0.0) + (1.0 / rank)
            combined_chunks.setdefault(chunk.id, chunk)

        ranked_ids = sorted(
            combined_scores,
            key=lambda chunk_id: (
                combined_scores[chunk_id],
                combined_chunks[chunk_id].similarity,
                combined_chunks[chunk_id].source,
                combined_chunks[chunk_id].id,
            ),
            reverse=True,
        )

        merged_results: list[RetrievedChunk] = []
        for chunk_id in ranked_ids[:limit]:
            chunk = combined_chunks[chunk_id]
            merged_results.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    similarity=min(combined_scores[chunk_id] / _HYBRID_SCORE_NORMALIZER, 0.99),
                    metadata=dict(chunk.metadata),
                )
            )
        return merged_results
