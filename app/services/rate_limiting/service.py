from __future__ import annotations

from datetime import UTC, datetime
import logging

from app.config import Settings
from app.infrastructure.cache import CacheClient, NullCacheClient, build_rate_limit_key
from app.services.rate_limiting.schemas import RateLimitActor, RateLimitLease

logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    def __init__(
        self,
        *,
        detail: str,
        retry_after_seconds: int,
        limit_type: str,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type


class RateLimitingBackendUnavailableError(Exception):
    def __init__(self, detail: str = "Rate limiting backend is unavailable.") -> None:
        super().__init__(detail)
        self.detail = detail


class RateLimitingService:
    def __init__(
        self,
        *,
        settings: Settings,
        cache_client: CacheClient | None = None,
        redis_client: CacheClient | None = None,
    ) -> None:
        self._settings = settings
        self._cache_client = cache_client or redis_client or NullCacheClient()

    async def enforce_request_limits(
        self,
        *,
        actor: RateLimitActor,
        endpoint: str,
    ) -> None:
        if not self._settings.enable_redis or not self._settings.rate_limit_enabled:
            return

        now = self._utcnow()
        window_seconds = self._settings.rate_limit_window_seconds
        window_bucket = int(now.timestamp()) // window_seconds
        key = build_rate_limit_key(
            endpoint=endpoint,
            actor_type=actor.actor_type,
            actor_id=actor.actor_id,
            window_bucket=window_bucket,
        )
        retry_after_seconds = self._seconds_until_window_end(now, window_seconds=window_seconds)

        try:
            current_value = int(await self._cache_client.incr(key))
            await self._cache_client.expire(key, retry_after_seconds)
        except Exception:
            logger.warning("Rate limit check failed. Continuing without blocking.", exc_info=True)
            return

        if current_value <= self._settings.rate_limit_max_requests:
            return

        raise RateLimitExceededError(
            detail="Rate limit exceeded. Please try again later.",
            retry_after_seconds=retry_after_seconds,
            limit_type="fixed_window",
        )

    async def acquire_concurrency_lease(
        self,
        *,
        actor: RateLimitActor,
        endpoint: str,
    ) -> RateLimitLease | None:
        return None

    async def release_concurrency_lease(self, lease: RateLimitLease | None) -> None:
        return None

    async def enforce_chat_budget(
        self,
        *,
        actor: RateLimitActor | None,
    ) -> None:
        return None

    async def record_llm_usage(
        self,
        *,
        actor: RateLimitActor | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
    ) -> None:
        return None

    def _seconds_until_window_end(self, now: datetime, *, window_seconds: int) -> int:
        current_second = int(now.timestamp())
        remainder = current_second % window_seconds
        return max(1, window_seconds - remainder)

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)
