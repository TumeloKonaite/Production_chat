from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.cache.models import (
    CacheLookupRequest,
    CacheStoreEntry,
    ResponseCacheLookupOutcome,
    ResponseCacheStoreOutcome,
    SemanticCacheLookupRequest,
)


class ResponseCache(ABC):
    @abstractmethod
    async def get_exact(
        self,
        request: CacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        ...

    @abstractmethod
    async def get_semantic(
        self,
        request: SemanticCacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        ...

    @abstractmethod
    async def store(
        self,
        entry: CacheStoreEntry,
    ) -> ResponseCacheStoreOutcome:
        ...


class NoOpResponseCache(ResponseCache):
    async def get_exact(
        self,
        request: CacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        return ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="disabled",
            latency_ms=0,
        )

    async def get_semantic(
        self,
        request: SemanticCacheLookupRequest,
    ) -> ResponseCacheLookupOutcome:
        return ResponseCacheLookupOutcome(
            cache_type="semantic",
            hit=False,
            reason="disabled",
            latency_ms=0,
        )

    async def store(
        self,
        entry: CacheStoreEntry,
    ) -> ResponseCacheStoreOutcome:
        return ResponseCacheStoreOutcome(
            success=False,
            reason="write_skipped",
            latency_ms=0,
            entry_id=entry.entry_id,
        )
