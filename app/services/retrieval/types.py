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


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        ...
