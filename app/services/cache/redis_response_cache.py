from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib import import_module
import json
import logging
from time import perf_counter
from typing import Any, TYPE_CHECKING

from app.config import Settings
from app.services.cache.models import (
    CacheLookupRequest,
    CacheLookupResult,
    CacheStoreEntry,
    ResponseCacheLookupOutcome,
    ResponseCacheStoreOutcome,
    SemanticCacheLookupRequest,
)
from app.services.cache.response_cache import ResponseCache

if TYPE_CHECKING:
    from app.infrastructure.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class RedisResponseCache(ResponseCache):
    def __init__(
        self,
        *,
        settings: Settings,
        embedding_provider: EmbeddingProvider | None = None,
        redis_client: Any | None = None,
        semantic_cache: Any | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._exact_enabled = settings.enable_response_cache and settings.enable_exact_response_cache
        self._semantic_enabled = (
            settings.enable_response_cache and settings.enable_semantic_response_cache
        )
        self._redis_client = redis_client
        self._semantic_cache = semantic_cache
        self._semantic_cache_unavailable_reason: str | None = None

    async def get_exact(
        self,
        request: CacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        if not self._exact_enabled:
            return self._disabled_lookup("exact")

        started_at = perf_counter()
        client = self._get_redis_client()
        if client is None:
            return self._unavailable_lookup("exact", started_at)

        key = self._build_exact_key(request.metadata_scope_hash, request.question_hash)
        try:
            payload = await client.get(key)
        except Exception:
            logger.warning("Exact response cache lookup failed.", exc_info=True)
            return self._unavailable_lookup("exact", started_at)

        latency_ms = self._elapsed_ms(started_at)
        if payload is None:
            return ResponseCacheLookupOutcome(
                cache_type="exact",
                hit=False,
                reason="miss_no_exact_entry",
                latency_ms=latency_ms,
            )

        entry = self._deserialize_entry(payload, cache_type="exact")
        if entry is None:
            return ResponseCacheLookupOutcome(
                cache_type="exact",
                hit=False,
                reason="miss_metadata_mismatch",
                latency_ms=latency_ms,
            )

        now = self._utcnow()
        if entry.expires_at is not None and entry.expires_at <= now:
            try:
                await client.delete(key)
            except Exception:
                logger.warning("Expired exact response cache entry cleanup failed.", exc_info=True)
            return ResponseCacheLookupOutcome(
                cache_type="exact",
                hit=False,
                reason="miss_expired",
                latency_ms=latency_ms,
                entry_id=entry.entry_id,
            )

        if entry.metadata_scope_hash != request.metadata_scope_hash:
            return ResponseCacheLookupOutcome(
                cache_type="exact",
                hit=False,
                reason="miss_metadata_mismatch",
                latency_ms=latency_ms,
                entry_id=entry.entry_id,
            )

        await self._refresh_exact_entry(key, entry)
        return ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=latency_ms,
            entry=entry,
            entry_id=entry.entry_id,
        )

    async def get_semantic(
        self,
        request: SemanticCacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        if not self._semantic_enabled:
            return self._disabled_lookup("semantic")

        started_at = perf_counter()
        semantic_cache = self._get_semantic_cache()
        if semantic_cache is None:
            return self._unavailable_lookup("semantic", started_at)

        try:
            results = await semantic_cache.acheck(
                vector=request.question_embedding,
                num_results=request.max_results,
                distance_threshold=request.distance_threshold,
                filter_expression=self._build_semantic_filter(request.metadata_scope_hash),
            )
        except Exception:
            logger.warning("Semantic response cache lookup failed.", exc_info=True)
            return self._unavailable_lookup("semantic", started_at)

        latency_ms = self._elapsed_ms(started_at)
        if not results:
            return ResponseCacheLookupOutcome(
                cache_type="semantic",
                hit=False,
                reason="miss_no_semantic_candidates",
                latency_ms=latency_ms,
            )

        raw_result = results[0]
        distance = self._extract_distance(raw_result)
        if distance is not None and distance > request.distance_threshold:
            return ResponseCacheLookupOutcome(
                cache_type="semantic",
                hit=False,
                reason="miss_below_threshold",
                latency_ms=latency_ms,
                distance=distance,
            )

        entry = self._deserialize_semantic_entry(raw_result)
        if entry is None:
            return ResponseCacheLookupOutcome(
                cache_type="semantic",
                hit=False,
                reason="miss_metadata_mismatch",
                latency_ms=latency_ms,
                distance=distance,
            )

        now = self._utcnow()
        if entry.expires_at is not None and entry.expires_at <= now:
            return ResponseCacheLookupOutcome(
                cache_type="semantic",
                hit=False,
                reason="miss_expired",
                latency_ms=latency_ms,
                distance=distance,
                entry_id=entry.entry_id,
            )

        if entry.metadata_scope_hash != request.metadata_scope_hash:
            return ResponseCacheLookupOutcome(
                cache_type="semantic",
                hit=False,
                reason="miss_metadata_mismatch",
                latency_ms=latency_ms,
                distance=distance,
                entry_id=entry.entry_id,
            )

        await self._refresh_semantic_entry(entry)
        return ResponseCacheLookupOutcome(
            cache_type="semantic",
            hit=True,
            reason="semantic_hit",
            latency_ms=latency_ms,
            entry=entry,
            distance=distance,
            entry_id=entry.entry_id,
        )

    async def store(
        self,
        entry: CacheStoreEntry,
    ) -> ResponseCacheStoreOutcome:
        if not self._exact_enabled and not self._semantic_enabled:
            return ResponseCacheStoreOutcome(
                success=False,
                reason="write_skipped",
                latency_ms=0,
                entry_id=entry.entry_id,
            )

        started_at = perf_counter()
        exact_success = False
        semantic_success = False

        if self._exact_enabled:
            client = self._get_redis_client()
            if client is not None:
                try:
                    key = self._build_exact_key(entry.metadata_scope_hash, entry.question_hash)
                    await client.set(
                        key,
                        self._serialize_entry(entry),
                        ex=self._settings.response_cache_ttl_seconds,
                    )
                    exact_success = True
                except Exception:
                    logger.warning("Exact response cache write failed.", exc_info=True)

        if self._semantic_enabled and entry.question_embedding is not None:
            semantic_cache = self._get_semantic_cache()
            if semantic_cache is not None:
                try:
                    await semantic_cache.astore(
                        prompt=entry.normalized_question,
                        response=entry.answer_text,
                        vector=entry.question_embedding,
                        metadata=self._entry_payload(entry),
                        filters={
                            "metadata_scope_hash": entry.metadata_scope_hash,
                        },
                        ttl=self._settings.response_cache_ttl_seconds,
                    )
                    semantic_success = True
                except Exception:
                    logger.warning("Semantic response cache write failed.", exc_info=True)

        latency_ms = self._elapsed_ms(started_at)
        if exact_success or semantic_success:
            return ResponseCacheStoreOutcome(
                success=True,
                reason="write_success",
                latency_ms=latency_ms,
                entry_id=entry.entry_id,
            )
        return ResponseCacheStoreOutcome(
            success=False,
            reason="write_failed",
            latency_ms=latency_ms,
            entry_id=entry.entry_id,
        )

    def _get_redis_client(self) -> Any | None:
        if self._redis_client is not None:
            return self._redis_client
        try:
            redis_asyncio = import_module("redis.asyncio")
            self._redis_client = redis_asyncio.from_url(
                self._settings.redis_url,
                decode_responses=True,
            )
        except Exception:
            logger.warning("Redis client initialization failed.", exc_info=True)
            return None
        return self._redis_client

    def _get_semantic_cache(self) -> Any | None:
        if self._semantic_cache is not None:
            return self._semantic_cache
        if self._semantic_cache_unavailable_reason is not None:
            return None
        if self._embedding_provider is None:
            return None

        try:
            semantic_module = import_module("redisvl.extensions.cache.llm.semantic")
            vectorizer_module = import_module("redisvl.utils.vectorize.base")
            semantic_cache_cls = semantic_module.SemanticCache
            base_vectorizer_cls = vectorizer_module.BaseVectorizer
            self._semantic_cache = semantic_cache_cls(
                name=self._settings.response_cache_semantic_index,
                distance_threshold=self._settings.response_cache_distance_threshold,
                ttl=self._settings.response_cache_ttl_seconds,
                vectorizer=self._build_vectorizer(base_vectorizer_cls),
                filterable_fields=[
                    {"name": "metadata_scope_hash", "type": "tag"},
                ],
                redis_url=self._settings.redis_url,
            )
        except Exception as exc:
            if self._is_redisearch_unavailable_error(exc):
                self._semantic_cache_unavailable_reason = "redisearch_unavailable"
                logger.warning(
                    "Semantic response cache disabled because the Redis server does not "
                    "support RediSearch commands. Use Redis Stack or set "
                    "ENABLE_SEMANTIC_RESPONSE_CACHE=false."
                )
            else:
                self._semantic_cache_unavailable_reason = "initialization_failed"
                logger.warning(
                    "RedisVL semantic cache initialization failed. Semantic cache "
                    "disabled until restart.",
                    exc_info=True,
                )
            return None
        return self._semantic_cache

    def _is_redisearch_unavailable_error(self, exc: BaseException) -> bool:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = str(current).casefold()
            if "ft._list" in message:
                return True
            if "unknown command" in message and "ft." in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _build_vectorizer(self, base_vectorizer_cls: type) -> Any:
        embedding_provider = self._embedding_provider
        if embedding_provider is None:
            raise ValueError("An embedding provider is required for semantic response caching.")

        class EmbeddingProviderVectorizer(base_vectorizer_cls):
            @property
            def type(self) -> str:
                return "custom"

            def _embed(self, text: str, **kwargs) -> list[float]:
                return embedding_provider.embed_query(text)

            def _embed_many(
                self,
                texts: list[str],
                batch_size: int = 10,
                **kwargs,
            ) -> list[list[float]]:
                return [embedding_provider.embed_query(text) for text in texts]

            async def _aembed(self, text: str, **kwargs) -> list[float]:
                return embedding_provider.embed_query(text)

            async def _aembed_many(
                self,
                texts: list[str],
                batch_size: int = 10,
                **kwargs,
            ) -> list[list[float]]:
                return [embedding_provider.embed_query(text) for text in texts]

        return EmbeddingProviderVectorizer(
            model=embedding_provider.model_name,
            dims=embedding_provider.dimension,
            dtype="float32",
        )

    def _build_exact_key(self, metadata_scope_hash: str, question_hash: str) -> str:
        return (
            f"{self._settings.response_cache_exact_prefix}:"
            f"{metadata_scope_hash}:{question_hash}"
        )

    def _build_semantic_filter(self, metadata_scope_hash: str) -> Any:
        filter_module = import_module("redisvl.query.filter")
        return filter_module.Tag("metadata_scope_hash") == metadata_scope_hash

    async def _refresh_exact_entry(self, key: str, entry: CacheLookupResult) -> None:
        client = self._get_redis_client()
        if client is None:
            return

        try:
            ttl_seconds = await client.ttl(key)
            refreshed = replace(
                entry,
                last_hit_at=self._utcnow(),
                hit_count=entry.hit_count + 1,
            )
            if ttl_seconds is None or ttl_seconds <= 0:
                return
            await client.set(
                key,
                self._serialize_lookup_result(refreshed),
                ex=ttl_seconds,
            )
        except Exception:
            logger.warning("Exact response cache hit refresh failed.", exc_info=True)

    async def _refresh_semantic_entry(self, entry: CacheLookupResult) -> None:
        semantic_cache = self._get_semantic_cache()
        if semantic_cache is None:
            return

        try:
            refreshed = replace(
                entry,
                last_hit_at=self._utcnow(),
                hit_count=entry.hit_count + 1,
            )
            index = getattr(semantic_cache, "index", None)
            key = index.key(entry.entry_id) if index is not None else entry.entry_id
            await semantic_cache.aupdate(
                key,
                metadata=self._lookup_result_payload(refreshed),
            )
        except Exception:
            logger.warning("Semantic response cache hit refresh failed.", exc_info=True)

    def _serialize_entry(self, entry: CacheStoreEntry) -> str:
        return self._serialize_payload(self._entry_payload(entry))

    def _serialize_lookup_result(self, entry: CacheLookupResult) -> str:
        return self._serialize_payload(self._lookup_result_payload(entry))

    def _serialize_payload(self, payload: dict[str, object]) -> str:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _entry_payload(self, entry: CacheStoreEntry) -> dict[str, object]:
        payload = asdict(entry)
        payload["cache_type"] = "semantic"
        return self._normalize_payload(payload)

    def _lookup_result_payload(self, entry: CacheLookupResult) -> dict[str, object]:
        payload = asdict(entry)
        return self._normalize_payload(payload)

    def _normalize_payload(self, payload: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, datetime):
                normalized[key] = value.isoformat()
            elif isinstance(value, list):
                normalized[key] = value
            elif isinstance(value, dict):
                normalized[key] = value
            else:
                normalized[key] = value
        return normalized

    def _deserialize_entry(
        self,
        payload: str,
        *,
        cache_type: str,
    ) -> CacheLookupResult | None:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Response cache payload deserialization failed.", exc_info=True)
            return None
        raw["cache_type"] = cache_type
        return self._build_lookup_result(raw)

    def _deserialize_semantic_entry(self, payload: dict[str, object]) -> CacheLookupResult | None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        normalized = dict(metadata)
        normalized.setdefault("entry_id", self._extract_string(payload, "entry_id"))
        normalized.setdefault(
            "answer_text",
            self._extract_string(payload, "response") or self._extract_string(payload, "answer_text"),
        )
        normalized["cache_type"] = "semantic"
        return self._build_lookup_result(normalized)

    def _build_lookup_result(self, payload: dict[str, object]) -> CacheLookupResult | None:
        entry_id = self._extract_string(payload, "entry_id")
        normalized_question = self._extract_string(payload, "normalized_question")
        question_hash = self._extract_string(payload, "question_hash")
        answer_text = self._extract_string(payload, "answer_text")
        llm_provider = self._extract_string(payload, "llm_provider")
        llm_model = self._extract_string(payload, "llm_model")
        prompt_version = self._extract_string(payload, "prompt_version")
        embedding_provider = self._extract_string(payload, "embedding_provider")
        embedding_model = self._extract_string(payload, "embedding_model")
        knowledge_base_version = self._extract_string(payload, "knowledge_base_version")
        retriever_type = self._extract_string(payload, "retriever_type")
        retriever_config_hash = self._extract_string(payload, "retriever_config_hash")
        metadata_scope_hash = self._extract_string(payload, "metadata_scope_hash")
        retrieval_config = self._extract_string(payload, "retrieval_config")
        cache_type = self._extract_string(payload, "cache_type") or "exact"
        created_at = self._parse_datetime(payload.get("created_at"))
        if not all(
            [
                entry_id,
                normalized_question,
                question_hash,
                answer_text,
                llm_provider,
                llm_model,
                prompt_version,
                embedding_provider,
                embedding_model,
                knowledge_base_version,
                retriever_type,
                retriever_config_hash,
                metadata_scope_hash,
                retrieval_config,
                created_at,
            ]
        ):
            return None

        source_documents = payload.get("source_documents")
        if not isinstance(source_documents, list):
            source_documents = []

        return CacheLookupResult(
            entry_id=entry_id,
            cache_type="semantic" if cache_type == "semantic" else "exact",
            normalized_question=normalized_question,
            question_hash=question_hash,
            answer_text=answer_text,
            source_documents=[
                item for item in source_documents if isinstance(item, dict)
            ],
            llm_provider=llm_provider,
            llm_model=llm_model,
            prompt_version=prompt_version,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            knowledge_base_version=knowledge_base_version,
            retriever_type=retriever_type,
            top_k=int(payload.get("top_k", 0)),
            query_rewrite_enabled=bool(payload.get("query_rewrite_enabled", False)),
            reranker_enabled=bool(payload.get("reranker_enabled", False)),
            retriever_config_hash=retriever_config_hash,
            metadata_scope_hash=metadata_scope_hash,
            retrieval_config=retrieval_config,
            created_at=created_at,
            expires_at=self._parse_datetime(payload.get("expires_at")),
            last_hit_at=self._parse_datetime(payload.get("last_hit_at")),
            hit_count=int(payload.get("hit_count", 0)),
            total_latency_ms=self._optional_int(payload.get("total_latency_ms")),
            embedding_latency_ms=self._optional_int(payload.get("embedding_latency_ms")),
            retrieval_latency_ms=self._optional_int(payload.get("retrieval_latency_ms")),
            llm_latency_ms=self._optional_int(payload.get("llm_latency_ms")),
        )

    def _disabled_lookup(self, cache_type: str) -> ResponseCacheLookupOutcome:
        return ResponseCacheLookupOutcome(
            cache_type="semantic" if cache_type == "semantic" else "exact",
            hit=False,
            reason="disabled",
            latency_ms=0,
        )

    def _unavailable_lookup(
        self,
        cache_type: str,
        started_at: float,
    ) -> ResponseCacheLookupOutcome:
        return ResponseCacheLookupOutcome(
            cache_type="semantic" if cache_type == "semantic" else "exact",
            hit=False,
            reason="redis_unavailable",
            latency_ms=self._elapsed_ms(started_at),
        )

    def _extract_distance(self, payload: dict[str, object]) -> float | None:
        for key in ("vector_distance", "score", "distance"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _extract_string(self, payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _parse_datetime(self, value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, int((perf_counter() - started_at) * 1000))

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)
