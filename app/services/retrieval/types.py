from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    source: str
    section: str
    content: str
    similarity: float
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    initial_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    reranker_enabled: bool
    reranker_type: str
    reranker_model: str | None
    initial_top_k: int
    final_top_k: int


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        ...
