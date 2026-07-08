from __future__ import annotations

import logging

from app.config import Settings
from app.infrastructure.cache import CacheClient, NullCacheClient, build_request_lock_key

logger = logging.getLogger(__name__)


class DuplicateRequestInProgressError(Exception):
    pass


class RequestLock:
    def __init__(
        self,
        *,
        settings: Settings,
        cache_client: CacheClient | None = None,
    ) -> None:
        self._settings = settings
        self._cache_client = cache_client or NullCacheClient()

    async def acquire(self, request_hash: str) -> bool:
        if not self._settings.enable_redis or not self._settings.request_lock_enabled:
            return True

        try:
            return await self._cache_client.set_if_not_exists(
                build_request_lock_key(request_hash),
                "1",
                self._settings.request_lock_ttl_seconds,
            )
        except Exception:
            logger.warning("Request lock acquisition failed.", exc_info=True)
            return True

    async def release(self, request_hash: str) -> None:
        if not self._settings.enable_redis or not self._settings.request_lock_enabled:
            return

        try:
            await self._cache_client.delete(build_request_lock_key(request_hash))
        except Exception:
            logger.warning("Request lock release failed.", exc_info=True)
