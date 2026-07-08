from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

from app.services.cache import (
    CacheLookupRequest,
    CacheScope,
    CacheStoreEntry,
    RedisResponseCache,
    hash_scope,
    normalize_question,
    stable_hash,
)
from tests.test_chat_api import build_test_settings


class FakeCacheClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.deleted: list[str] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        self.deleted.append(key)
        existed = key in self.values
        self.values.pop(key, None)
        return int(existed)

    async def incr(self, key: str) -> int:
        raise NotImplementedError

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        raise NotImplementedError

    async def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        raise NotImplementedError


def _build_scope() -> CacheScope:
    return CacheScope(
        knowledge_base_version="personal_knowledge_base",
        prompt_version="v1_professional",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash="retriever-hash",
    )


def _build_request() -> CacheLookupRequest:
    scope = _build_scope()
    return CacheLookupRequest(
        request_hash="request-hash",
        normalized_question=normalize_question("Tell me about Tumelo's work."),
        question_hash=stable_hash(normalize_question("Tell me about Tumelo's work.")),
        metadata_scope_hash=hash_scope(scope),
        metadata_scope=scope,
    )


def _build_cache_entry_payload(*, expires_at: datetime | None = None) -> str:
    created_at = datetime.now(UTC)
    payload = {
        "entry_id": "request-hash",
        "normalized_question": normalize_question("Tell me about Tumelo's work."),
        "question_hash": stable_hash(normalize_question("Tell me about Tumelo's work.")),
        "question_embedding": None,
        "answer_text": "Cached answer.",
        "source_documents": [{"chunk_id": "chunk-1"}],
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "prompt_version": "v1_professional",
        "embedding_provider": "hf",
        "embedding_model": "all-MiniLM-L6-v2",
        "knowledge_base_version": "personal_knowledge_base",
        "retriever_type": "vector",
        "top_k": 5,
        "query_rewrite_enabled": False,
        "reranker_enabled": False,
        "retriever_config_hash": "retriever-hash",
        "metadata_scope_hash": hash_scope(_build_scope()),
        "retrieval_config": "default",
        "created_at": created_at.isoformat(),
        "expires_at": (expires_at or (created_at + timedelta(minutes=5))).isoformat(),
        "total_latency_ms": 1000,
        "embedding_latency_ms": None,
        "retrieval_latency_ms": 120,
        "llm_latency_ms": 842,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def test_redis_response_cache_returns_exact_hit() -> None:
    import asyncio

    cache_client = FakeCacheClient()
    cache_client.values["chat:exact:v1:request-hash"] = _build_cache_entry_payload()
    cache = RedisResponseCache(
        settings=build_test_settings(enable_redis=True, exact_cache_enabled=True),
        redis_client=cache_client,
    )

    result = asyncio.run(cache.get_exact(_build_request()))

    assert result.hit is True
    assert result.reason == "exact_hit"
    assert result.entry is not None
    assert result.entry.answer_text == "Cached answer."


def test_redis_response_cache_treats_expired_entry_as_miss_and_deletes_it() -> None:
    import asyncio

    cache_client = FakeCacheClient()
    cache_client.values["chat:exact:v1:request-hash"] = _build_cache_entry_payload(
        expires_at=datetime.now(UTC) - timedelta(minutes=1)
    )
    cache = RedisResponseCache(
        settings=build_test_settings(enable_redis=True, exact_cache_enabled=True),
        redis_client=cache_client,
    )

    result = asyncio.run(cache.get_exact(_build_request()))

    assert result.hit is False
    assert result.reason == "miss_expired"
    assert cache_client.deleted == ["chat:exact:v1:request-hash"]


def test_redis_response_cache_stores_exact_entries_under_request_hash() -> None:
    import asyncio

    cache_client = FakeCacheClient()
    cache = RedisResponseCache(
        settings=build_test_settings(enable_redis=True, exact_cache_enabled=True),
        redis_client=cache_client,
    )
    created_at = datetime.now(UTC)

    result = asyncio.run(
        cache.store(
            CacheStoreEntry(
                entry_id="request-hash",
                normalized_question=normalize_question("Tell me about Tumelo's work."),
                question_hash=stable_hash(normalize_question("Tell me about Tumelo's work.")),
                question_embedding=None,
                answer_text="Cached answer.",
                source_documents=[],
                llm_provider="openai",
                llm_model="gpt-4.1-mini",
                prompt_version="v1_professional",
                embedding_provider="hf",
                embedding_model="all-MiniLM-L6-v2",
                knowledge_base_version="personal_knowledge_base",
                retriever_type="vector",
                top_k=5,
                query_rewrite_enabled=False,
                reranker_enabled=False,
                retriever_config_hash="retriever-hash",
                metadata_scope_hash=hash_scope(_build_scope()),
                retrieval_config="default",
                created_at=created_at,
                expires_at=created_at + timedelta(minutes=5),
                total_latency_ms=1000,
                embedding_latency_ms=None,
                retrieval_latency_ms=120,
                llm_latency_ms=842,
            )
        )
    )

    assert result.success is True
    assert "chat:exact:v1:request-hash" in cache_client.values
