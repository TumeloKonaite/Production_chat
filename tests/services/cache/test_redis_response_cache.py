from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import app.services.cache.redis_response_cache as redis_response_cache_module
from app.config import Settings
from app.services.cache import (
    CacheLookupRequest,
    CacheScope,
    CacheStoreEntry,
    RedisResponseCache,
    SemanticCacheLookupRequest,
)


class FakeRedisClient:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.deleted: list[str] = []

    async def get(self, key: str) -> str | None:
        return self.storage.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.storage[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.storage.pop(key, None)


class FakeSemanticIndex:
    def key(self, entry_id: str) -> str:
        return f"semantic:{entry_id}"


class FakeSemanticCache:
    def __init__(self, results: list[dict[str, object]] | None = None) -> None:
        self.results = list(results or [])
        self.check_calls: list[dict[str, object]] = []
        self.store_calls: list[dict[str, object]] = []
        self.update_calls: list[tuple[str, dict[str, object]]] = []
        self.index = FakeSemanticIndex()

    async def acheck(self, **kwargs) -> list[dict[str, object]]:
        self.check_calls.append(dict(kwargs))
        return list(self.results)

    async def astore(self, **kwargs) -> str:
        self.store_calls.append(dict(kwargs))
        return "semantic:key"

    async def aupdate(self, key: str, **kwargs) -> None:
        self.update_calls.append((key, dict(kwargs)))


class FakeEmbeddingProvider:
    model_name = "all-MiniLM-L6-v2"
    dimension = 384

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite:///cache-tests.db",
        "openai_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": None,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "tavus_api_key": None,
        "tavus_base_url": "https://tavusapi.com",
        "tavus_face_id": None,
        "tavus_pal_id": None,
        "public_backend_url": None,
        "tavus_tool_secret": None,
        "ingestion_api_secret": None,
        "eval_admin_token": None,
        "default_model_config_id": "openai:gpt-4.1-mini",
        "model_configs_json": None,
        "embedding_provider": "hf",
        "knowledge_embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "knowledge_collection_name": "personal_knowledge_base",
        "default_prompt_version": "v1_professional",
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "production-chatbot",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "enable_response_cache": True,
        "enable_exact_response_cache": True,
        "enable_semantic_response_cache": False,
        "response_cache_provider": "redis",
        "redis_url": "redis://localhost:6379/0",
        "response_cache_ttl_seconds": 3600,
        "response_cache_exact_prefix": "chat:exact",
        "response_cache_semantic_index": "chat_semantic_cache",
        "response_cache_distance_threshold": 0.10,
        "response_cache_max_results": 3,
        "response_cache_store_private_sessions": False,
        "response_cache_knowledge_base_version": "kb-v1",
    }
    values.update(overrides)
    return Settings(**values)


def build_scope() -> CacheScope:
    return CacheScope(
        knowledge_base_version="kb-v1",
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


def build_lookup_request() -> CacheLookupRequest:
    return CacheLookupRequest(
        normalized_question="what does tumelo do?",
        question_hash="question-hash",
        metadata_scope_hash="scope-hash",
        metadata_scope=build_scope(),
    )


def build_store_entry(*, expires_at: datetime | None = None) -> CacheStoreEntry:
    created_at = datetime.now(UTC)
    return CacheStoreEntry(
        entry_id="entry-1",
        normalized_question="what does tumelo do?",
        question_hash="question-hash",
        question_embedding=[0.1, 0.2, 0.3],
        answer_text="Tumelo builds AI systems.",
        source_documents=[{"chunk_id": "chunk-1", "source": "projects.md"}],
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        prompt_version="v1_professional",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        knowledge_base_version="kb-v1",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash="retriever-hash",
        metadata_scope_hash="scope-hash",
        retrieval_config="default",
        created_at=created_at,
        expires_at=expires_at or (created_at + timedelta(hours=1)),
        total_latency_ms=1000,
        embedding_latency_ms=20,
        retrieval_latency_ms=120,
        llm_latency_ms=842,
    )


def test_redis_response_cache_returns_exact_hit_and_refreshes_metadata() -> None:
    redis_client = FakeRedisClient()
    cache = RedisResponseCache(
        settings=build_settings(),
        redis_client=redis_client,
    )
    entry = build_store_entry()

    asyncio.run(cache.store(entry))
    outcome = asyncio.run(cache.get_exact(build_lookup_request()))

    assert outcome.hit is True
    assert outcome.reason == "exact_hit"
    assert outcome.entry is not None
    assert outcome.entry.answer_text == "Tumelo builds AI systems."
    refreshed_payload = json.loads(next(iter(redis_client.storage.values())))
    assert refreshed_payload["hit_count"] == 1
    assert refreshed_payload["last_hit_at"] is not None


def test_redis_response_cache_treats_expired_exact_entries_as_miss() -> None:
    redis_client = FakeRedisClient()
    cache = RedisResponseCache(
        settings=build_settings(),
        redis_client=redis_client,
    )
    expired_entry = build_store_entry(expires_at=datetime.now(UTC) - timedelta(minutes=5))
    request = build_lookup_request()
    key = cache._build_exact_key(request.metadata_scope_hash, request.question_hash)
    asyncio.run(redis_client.set(key, cache._serialize_entry(expired_entry), ex=3600))

    outcome = asyncio.run(cache.get_exact(request))

    assert outcome.hit is False
    assert outcome.reason == "miss_expired"
    assert redis_client.deleted == [key]


def test_redis_response_cache_returns_semantic_hit_and_updates_entry() -> None:
    semantic_cache = FakeSemanticCache(
        [
            {
                "entry_id": "entry-1",
                "response": "Tumelo builds AI systems.",
                "vector_distance": 0.04,
                "metadata": {
                    "entry_id": "entry-1",
                    "normalized_question": "what does tumelo do?",
                    "question_hash": "question-hash",
                    "answer_text": "Tumelo builds AI systems.",
                    "source_documents": [],
                    "llm_provider": "openai",
                    "llm_model": "gpt-4.1-mini",
                    "prompt_version": "v1_professional",
                    "embedding_provider": "hf",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "knowledge_base_version": "kb-v1",
                    "retriever_type": "vector",
                    "top_k": 5,
                    "query_rewrite_enabled": False,
                    "reranker_enabled": False,
                    "retriever_config_hash": "retriever-hash",
                    "metadata_scope_hash": "scope-hash",
                    "retrieval_config": "default",
                    "created_at": datetime.now(UTC).isoformat(),
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    "last_hit_at": None,
                    "hit_count": 0,
                    "total_latency_ms": 1000,
                    "embedding_latency_ms": 20,
                    "retrieval_latency_ms": 120,
                    "llm_latency_ms": 842,
                },
            }
        ]
    )
    cache = RedisResponseCache(
        settings=build_settings(enable_semantic_response_cache=True),
        semantic_cache=semantic_cache,
    )
    cache._build_semantic_filter = lambda *_: "*"  # type: ignore[method-assign]
    request = SemanticCacheLookupRequest(
        normalized_question="what does tumelo do?",
        question_hash="question-hash",
        metadata_scope_hash="scope-hash",
        metadata_scope=build_scope(),
        question_embedding=[0.1, 0.2, 0.3],
        distance_threshold=0.10,
        max_results=3,
    )

    outcome = asyncio.run(cache.get_semantic(request))

    assert outcome.hit is True
    assert outcome.reason == "semantic_hit"
    assert outcome.distance == 0.04
    assert outcome.entry is not None
    assert outcome.entry.answer_text == "Tumelo builds AI systems."
    assert semantic_cache.update_calls[0][0] == "semantic:entry-1"
    assert semantic_cache.update_calls[0][1]["metadata"]["entry_id"] == "entry-1"
    assert semantic_cache.update_calls[0][1]["metadata"]["hit_count"] == 1
    assert semantic_cache.update_calls[0][1]["metadata"]["last_hit_at"] is not None


def test_redis_response_cache_stores_semantic_entries_when_enabled() -> None:
    semantic_cache = FakeSemanticCache()
    cache = RedisResponseCache(
        settings=build_settings(enable_semantic_response_cache=True),
        redis_client=FakeRedisClient(),
        semantic_cache=semantic_cache,
    )

    outcome = asyncio.run(cache.store(build_store_entry()))

    assert outcome.success is True
    assert outcome.reason == "write_success"
    assert semantic_cache.store_calls[0]["filters"] == {"metadata_scope_hash": "scope-hash"}


def test_redis_response_cache_disables_semantic_cache_after_redisearch_init_failure(
    monkeypatch,
) -> None:
    init_calls = 0

    class FailingSemanticCache:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal init_calls
            init_calls += 1
            raise RuntimeError("unknown command 'FT._LIST'")

    class FakeBaseVectorizer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    def fake_import_module(name: str):
        if name == "redisvl.extensions.cache.llm.semantic":
            return SimpleNamespace(SemanticCache=FailingSemanticCache)
        if name == "redisvl.utils.vectorize.base":
            return SimpleNamespace(BaseVectorizer=FakeBaseVectorizer)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(redis_response_cache_module, "import_module", fake_import_module)

    cache = RedisResponseCache(
        settings=build_settings(enable_semantic_response_cache=True),
        embedding_provider=FakeEmbeddingProvider(),
    )
    request = SemanticCacheLookupRequest(
        normalized_question="what does tumelo do?",
        question_hash="question-hash",
        metadata_scope_hash="scope-hash",
        metadata_scope=build_scope(),
        question_embedding=[0.1, 0.2, 0.3],
        distance_threshold=0.10,
        max_results=3,
    )

    first_outcome = asyncio.run(cache.get_semantic(request))
    second_outcome = asyncio.run(cache.get_semantic(request))

    assert first_outcome.hit is False
    assert first_outcome.reason == "redis_unavailable"
    assert second_outcome.hit is False
    assert second_outcome.reason == "redis_unavailable"
    assert init_calls == 1
    assert cache._semantic_cache_unavailable_reason == "redisearch_unavailable"
