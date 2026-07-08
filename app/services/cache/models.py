from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

CacheType = Literal["exact", "semantic"]


@dataclass(frozen=True, slots=True)
class CacheScope:
    knowledge_base_version: str
    prompt_version: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    retriever_type: str
    top_k: int
    query_rewrite_enabled: bool
    reranker_enabled: bool
    retriever_config_hash: str

    def as_metadata(self) -> dict[str, object]:
        return {
            "knowledge_base_version": self.knowledge_base_version,
            "prompt_version": self.prompt_version,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "retriever_type": self.retriever_type,
            "top_k": self.top_k,
            "query_rewrite_enabled": self.query_rewrite_enabled,
            "reranker_enabled": self.reranker_enabled,
            "retriever_config_hash": self.retriever_config_hash,
        }


@dataclass(frozen=True, slots=True)
class CacheLookupRequest:
    request_hash: str
    normalized_question: str
    question_hash: str
    metadata_scope_hash: str
    metadata_scope: CacheScope


@dataclass(frozen=True, slots=True)
class SemanticCacheLookupRequest(CacheLookupRequest):
    question_embedding: list[float]
    distance_threshold: float
    max_results: int


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    entry_id: str
    cache_type: CacheType
    normalized_question: str
    question_hash: str
    answer_text: str
    source_documents: list[dict[str, object]]
    llm_provider: str
    llm_model: str
    prompt_version: str
    embedding_provider: str
    embedding_model: str
    knowledge_base_version: str
    retriever_type: str
    top_k: int
    query_rewrite_enabled: bool
    reranker_enabled: bool
    retriever_config_hash: str
    metadata_scope_hash: str
    retrieval_config: str
    created_at: datetime
    expires_at: datetime | None
    last_hit_at: datetime | None
    hit_count: int
    total_latency_ms: int | None = None
    embedding_latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    llm_latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ResponseCacheLookupOutcome:
    cache_type: CacheType
    hit: bool
    reason: str
    latency_ms: int
    entry: CacheLookupResult | None = None
    distance: float | None = None
    entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class CacheStoreEntry:
    entry_id: str
    normalized_question: str
    question_hash: str
    question_embedding: list[float] | None
    answer_text: str
    source_documents: list[dict[str, object]]
    llm_provider: str
    llm_model: str
    prompt_version: str
    embedding_provider: str
    embedding_model: str
    knowledge_base_version: str
    retriever_type: str
    top_k: int
    query_rewrite_enabled: bool
    reranker_enabled: bool
    retriever_config_hash: str
    metadata_scope_hash: str
    retrieval_config: str
    created_at: datetime
    expires_at: datetime | None
    total_latency_ms: int | None
    embedding_latency_ms: int | None
    retrieval_latency_ms: int | None
    llm_latency_ms: int | None


@dataclass(frozen=True, slots=True)
class ResponseCacheStoreOutcome:
    success: bool
    reason: str
    latency_ms: int
    entry_id: str | None = None
