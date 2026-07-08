from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
import logging
from time import perf_counter
from typing import Any

from app.config import Settings
from app.infrastructure.cache import CacheClient, build_exact_cache_key
from app.services.cache.models import (
    CacheLookupRequest,
    CacheLookupResult,
    CacheStoreEntry,
    ResponseCacheLookupOutcome,
    ResponseCacheStoreOutcome,
    SemanticCacheLookupRequest,
)
from app.services.cache.response_cache import ResponseCache

logger = logging.getLogger(__name__)


class RedisResponseCache(ResponseCache):
    def __init__(
        self,
        *,
        settings: Settings,
        embedding_provider: Any | None = None,
        redis_client: CacheClient | None = None,
        semantic_cache: Any | None = None,
    ) -> None:
        self._settings = settings
        self._redis_client = redis_client
        self._exact_enabled = settings.enable_redis and settings.exact_cache_enabled
        self._semantic_enabled = False

    async def get_exact(
        self,
        request: CacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        if not self._exact_enabled:
            return self._disabled_lookup("exact")

        started_at = perf_counter()
        client = self._redis_client
        if client is None:
            return self._unavailable_lookup("exact", started_at)

        key = build_exact_cache_key(request.request_hash)
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

        entry = self._deserialize_entry(payload)
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
        return self._disabled_lookup("semantic")

    async def store(
        self,
        entry: CacheStoreEntry,
    ) -> ResponseCacheStoreOutcome:
        if not self._exact_enabled:
            return ResponseCacheStoreOutcome(
                success=False,
                reason="write_skipped",
                latency_ms=0,
                entry_id=entry.entry_id,
            )

        started_at = perf_counter()
        client = self._redis_client
        if client is None:
            return ResponseCacheStoreOutcome(
                success=False,
                reason="write_failed",
                latency_ms=self._elapsed_ms(started_at),
                entry_id=entry.entry_id,
            )

        try:
            key = build_exact_cache_key(entry.entry_id)
            success = await client.set(
                key,
                self._serialize_entry(entry),
                self._settings.exact_cache_ttl_seconds,
            )
        except Exception:
            logger.warning("Exact response cache write failed.", exc_info=True)
            success = False

        return ResponseCacheStoreOutcome(
            success=success,
            reason="write_success" if success else "write_failed",
            latency_ms=self._elapsed_ms(started_at),
            entry_id=entry.entry_id,
        )

    def _serialize_entry(self, entry: CacheStoreEntry) -> str:
        payload = asdict(entry)
        return json.dumps(self._normalize_payload(payload), separators=(",", ":"), sort_keys=True)

    def _normalize_payload(self, payload: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, datetime):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        return normalized

    def _deserialize_entry(self, payload: str) -> CacheLookupResult | None:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Response cache payload deserialization failed.", exc_info=True)
            return None
        return self._build_lookup_result(raw)

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
            cache_type="exact",
            normalized_question=normalized_question,
            question_hash=question_hash,
            answer_text=answer_text,
            source_documents=[item for item in source_documents if isinstance(item, dict)],
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
