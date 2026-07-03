from __future__ import annotations

from collections.abc import Callable, Sequence
import re

from app.repositories.models import KnowledgeChunk
from app.services.retrieval.strategies.base import normalize_query, resolve_top_k
from app.services.retrieval.types import RetrievedChunk

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_KEYWORD_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "background",
    "build",
    "built",
    "can",
    "could",
    "details",
    "did",
    "do",
    "does",
    "experience",
    "explain",
    "for",
    "from",
    "give",
    "has",
    "have",
    "help",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "more",
    "of",
    "on",
    "or",
    "please",
    "project",
    "projects",
    "tell",
    "that",
    "the",
    "this",
    "tumelo",
    "us",
    "was",
    "what",
    "which",
    "who",
    "with",
    "work",
    "worked",
    "you",
}


class KeywordRetriever:
    def __init__(
        self,
        *,
        default_top_k: int,
        chunk_loader: Callable[[], Sequence[KnowledgeChunk]],
    ) -> None:
        self._default_top_k = default_top_k
        self._chunk_loader = chunk_loader

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        normalized_query = normalize_query(query)
        if not normalized_query:
            return []

        limit = resolve_top_k(top_k, default_top_k=self._default_top_k)
        query_terms = self._extract_query_terms(normalized_query)
        if not query_terms:
            return []

        scored_chunks: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunk_loader():
            score = self._score_chunk(
                section=chunk.section,
                content=chunk.content,
                query_terms=query_terms,
            )
            if score <= 0:
                continue
            scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda item: (
                item[0],
                item[1].source == "projects.md",
                len(item[1].content),
            ),
            reverse=True,
        )
        return [
            self._to_retrieved_chunk(chunk=chunk, score=score)
            for score, chunk in scored_chunks[:limit]
        ]

    def _extract_query_terms(self, query: str) -> list[str]:
        tokens = _QUERY_TOKEN_RE.findall(query.casefold())
        return [
            token
            for token in tokens
            if len(token) >= 3 and token not in _KEYWORD_STOPWORDS
        ]

    def _score_chunk(
        self,
        *,
        section: str,
        content: str,
        query_terms: list[str],
    ) -> float:
        normalized_section_tokens = set(_QUERY_TOKEN_RE.findall(section.casefold()))
        normalized_content_tokens = set(_QUERY_TOKEN_RE.findall(content.casefold()))

        section_matches = [term for term in query_terms if term in normalized_section_tokens]
        content_matches = [term for term in query_terms if term in normalized_content_tokens]
        total_matches = len(set(section_matches + content_matches))
        if total_matches == 0:
            return 0.0

        score = 0.0
        score += len(section_matches) * 0.45
        score += len(content_matches) * 0.25
        if section_matches and len(section_matches) == len(query_terms):
            score += 0.35
        if content_matches and len(content_matches) == len(query_terms):
            score += 0.2

        return min(score, 0.99)

    def _to_retrieved_chunk(self, *, chunk: KnowledgeChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            id=chunk.id,
            source=chunk.source,
            section=chunk.section,
            content=chunk.content,
            similarity=score,
            metadata={
                **chunk.chunk_metadata,
                "chunk_id": chunk.id,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "section": chunk.section,
            },
        )
