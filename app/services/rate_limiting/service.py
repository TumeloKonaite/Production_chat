from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import import_module
import logging
from typing import Any

from app.config import Settings
from app.services.rate_limiting.schemas import RateLimitActor, RateLimitLease

logger = logging.getLogger(__name__)
_CONCURRENCY_TTL_SECONDS = 300
_WINDOW_10_MINUTES_SECONDS = 600


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
        redis_client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._redis_client = redis_client

    async def enforce_request_limits(
        self,
        *,
        actor: RateLimitActor,
        endpoint: str,
    ) -> None:
        if not self._settings.enable_rate_limiting:
            return

        now = self._utcnow()
        client = self._get_redis_client()
        if client is None:
            self._handle_backend_unavailable("request_limits")
            return

        await self._check_request_counter(
            client=client,
            actor=actor,
            endpoint=endpoint,
            key=self._request_window_key(
                endpoint=endpoint,
                actor=actor,
                window_started_at=now,
            ),
            limit=self._settings.chat_rate_limit_requests_per_10_minutes,
            expires_in_seconds=self._seconds_until_window_end(
                now,
                window_seconds=_WINDOW_10_MINUTES_SECONDS,
            ),
            limit_type="requests_per_10_minutes",
        )
        await self._check_request_counter(
            client=client,
            actor=actor,
            endpoint=endpoint,
            key=self._request_day_key(endpoint=endpoint, actor=actor, now=now),
            limit=self._settings.chat_rate_limit_requests_per_day,
            expires_in_seconds=self._seconds_until_day_end(now),
            limit_type="requests_per_day",
        )

    async def acquire_concurrency_lease(
        self,
        *,
        actor: RateLimitActor,
        endpoint: str,
    ) -> RateLimitLease | None:
        if not self._settings.enable_rate_limiting:
            return None

        client = self._get_redis_client()
        if client is None:
            self._handle_backend_unavailable("concurrency_limit")
            return None

        key = self._concurrency_key(endpoint=endpoint, actor=actor)
        try:
            current_value = int(await client.incr(key))
            await client.expire(key, _CONCURRENCY_TTL_SECONDS)
        except Exception:
            if self._handle_backend_failure("concurrency_limit"):
                return None
            raise

        if current_value > self._settings.chat_rate_limit_concurrent_requests:
            try:
                await client.decr(key)
            except Exception:
                logger.warning("Rate limit concurrency rollback failed.", exc_info=True)
            retry_after_seconds = self._safe_ttl_seconds(_CONCURRENCY_TTL_SECONDS)
            self._log_blocked(
                actor=actor,
                endpoint=endpoint,
                limit_type="concurrent_requests",
                limit_value=self._settings.chat_rate_limit_concurrent_requests,
                current_value=current_value,
                retry_after_seconds=retry_after_seconds,
            )
            raise RateLimitExceededError(
                detail="Too many concurrent chat requests. Please wait for earlier requests to finish.",
                retry_after_seconds=retry_after_seconds,
                limit_type="concurrent_requests",
            )

        return RateLimitLease(actor=actor, endpoint=endpoint, key=key)

    async def release_concurrency_lease(self, lease: RateLimitLease | None) -> None:
        if lease is None or not self._settings.enable_rate_limiting:
            return

        client = self._get_redis_client()
        if client is None:
            return

        try:
            remaining = int(await client.decr(lease.key))
            if remaining <= 0:
                await client.delete(lease.key)
        except Exception:
            logger.warning("Rate limit concurrency release failed.", exc_info=True)

    async def enforce_chat_budget(
        self,
        *,
        actor: RateLimitActor | None,
    ) -> None:
        if not self._settings.enable_rate_limiting or actor is None:
            return

        now = self._utcnow()
        client = self._get_redis_client()
        if client is None:
            self._handle_backend_unavailable("usage_budget")
            return

        token_key = self._usage_key(kind="tokens", actor=actor, now=now)
        cost_key = self._usage_key(kind="cost", actor=actor, now=now)
        try:
            current_tokens = self._safe_int(await client.get(token_key))
            current_cost = self._safe_float(await client.get(cost_key))
        except Exception:
            if self._handle_backend_failure("usage_budget"):
                return
            raise

        if current_tokens >= self._settings.chat_rate_limit_daily_token_budget:
            retry_after_seconds = self._seconds_until_day_end(now)
            self._log_blocked(
                actor=actor,
                endpoint="chat",
                limit_type="daily_token_budget",
                limit_value=self._settings.chat_rate_limit_daily_token_budget,
                current_value=current_tokens,
                retry_after_seconds=retry_after_seconds,
            )
            raise RateLimitExceededError(
                detail="Daily token budget exceeded. Please try again tomorrow.",
                retry_after_seconds=retry_after_seconds,
                limit_type="daily_token_budget",
            )

        if current_cost >= self._settings.chat_rate_limit_daily_cost_budget_usd:
            retry_after_seconds = self._seconds_until_day_end(now)
            self._log_blocked(
                actor=actor,
                endpoint="chat",
                limit_type="daily_cost_budget_usd",
                limit_value=self._settings.chat_rate_limit_daily_cost_budget_usd,
                current_value=current_cost,
                retry_after_seconds=retry_after_seconds,
            )
            raise RateLimitExceededError(
                detail="Daily LLM cost budget exceeded. Please try again tomorrow.",
                retry_after_seconds=retry_after_seconds,
                limit_type="daily_cost_budget_usd",
            )

    async def record_llm_usage(
        self,
        *,
        actor: RateLimitActor | None,
        total_tokens: int | None,
        estimated_cost_usd: float | None,
    ) -> None:
        if not self._settings.enable_rate_limiting or actor is None:
            return
        if total_tokens is None and estimated_cost_usd is None:
            return

        client = self._get_redis_client()
        if client is None:
            logger.warning("Skipping LLM usage recording because Redis is unavailable.")
            return

        now = self._utcnow()
        expires_in_seconds = self._seconds_until_day_end(now)
        try:
            if total_tokens is not None:
                token_key = self._usage_key(kind="tokens", actor=actor, now=now)
                await client.incrby(token_key, int(total_tokens))
                await client.expire(token_key, expires_in_seconds)
            if estimated_cost_usd is not None:
                cost_key = self._usage_key(kind="cost", actor=actor, now=now)
                await client.incrbyfloat(cost_key, float(estimated_cost_usd))
                await client.expire(cost_key, expires_in_seconds)
        except Exception:
            # Usage is recorded after spend has already occurred, so continue serving the response.
            logger.warning("LLM usage recording failed.", exc_info=True)

    async def _check_request_counter(
        self,
        *,
        client: Any,
        actor: RateLimitActor,
        endpoint: str,
        key: str,
        limit: int,
        expires_in_seconds: int,
        limit_type: str,
    ) -> None:
        try:
            current_value = int(await client.incr(key))
            await client.expire(key, expires_in_seconds)
        except Exception:
            if self._handle_backend_failure(limit_type):
                return
            raise

        if current_value <= limit:
            return

        self._log_blocked(
            actor=actor,
            endpoint=endpoint,
            limit_type=limit_type,
            limit_value=limit,
            current_value=current_value,
            retry_after_seconds=expires_in_seconds,
        )
        raise RateLimitExceededError(
            detail="Rate limit exceeded. Please try again later.",
            retry_after_seconds=expires_in_seconds,
            limit_type=limit_type,
        )

    def _request_window_key(
        self,
        *,
        endpoint: str,
        actor: RateLimitActor,
        window_started_at: datetime,
    ) -> str:
        bucket = int(window_started_at.timestamp()) // _WINDOW_10_MINUTES_SECONDS
        return f"rate_limit:{endpoint}:10m:{bucket}:{actor.actor_id}"

    def _request_day_key(
        self,
        *,
        endpoint: str,
        actor: RateLimitActor,
        now: datetime,
    ) -> str:
        return f"rate_limit:{endpoint}:day:{now.strftime('%Y%m%d')}:{actor.actor_id}"

    def _concurrency_key(
        self,
        *,
        endpoint: str,
        actor: RateLimitActor,
    ) -> str:
        return f"rate_limit:{endpoint}:concurrent:{actor.actor_id}"

    def _usage_key(
        self,
        *,
        kind: str,
        actor: RateLimitActor,
        now: datetime,
    ) -> str:
        return f"usage_budget:{kind}:day:{now.strftime('%Y%m%d')}:{actor.actor_id}"

    def _get_redis_client(self) -> Any | None:
        if self._redis_client is not None:
            return self._redis_client
        redis_url = self._settings.resolved_redis_url
        if redis_url is None:
            logger.info("Rate limiting disabled Redis usage because REDIS_URL is not configured.")
            return None
        try:
            redis_asyncio = import_module("redis.asyncio")
            self._redis_client = redis_asyncio.from_url(
                redis_url,
                decode_responses=True,
            )
        except Exception:
            logger.warning("Rate limiting Redis client initialization failed.", exc_info=True)
            return None
        return self._redis_client

    def _handle_backend_failure(self, operation: str) -> bool:
        logger.warning("Rate limiting Redis operation failed: %s", operation, exc_info=True)
        return self._handle_backend_unavailable(operation)

    def _handle_backend_unavailable(self, operation: str) -> bool:
        if self._settings.rate_limiting_fail_open:
            logger.warning(
                "Rate limiting backend unavailable during %s. Continuing because fail-open is enabled.",
                operation,
            )
            return True
        raise RateLimitingBackendUnavailableError()

    def _log_blocked(
        self,
        *,
        actor: RateLimitActor,
        endpoint: str,
        limit_type: str,
        limit_value: int | float,
        current_value: int | float,
        retry_after_seconds: int,
    ) -> None:
        logger.warning(
            "Rate limit blocked request",
            extra={
                "rate_limit": {
                    "actor_id": actor.actor_id,
                    "actor_type": actor.actor_type,
                    "endpoint": endpoint,
                    "limit_type": limit_type,
                    "limit_value": limit_value,
                    "current_value": current_value,
                    "retry_after_seconds": retry_after_seconds,
                    "timestamp": self._utcnow().isoformat(),
                }
            },
        )

    def _seconds_until_window_end(self, now: datetime, *, window_seconds: int) -> int:
        current_second = int(now.timestamp())
        remainder = current_second % window_seconds
        return self._safe_ttl_seconds(window_seconds - remainder)

    def _seconds_until_day_end(self, now: datetime) -> int:
        next_day = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            tzinfo=UTC,
        ) + timedelta(days=1)
        return self._safe_ttl_seconds(int((next_day - now).total_seconds()))

    def _safe_ttl_seconds(self, value: int) -> int:
        return max(1, value)

    def _safe_int(self, value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return 0
        return 0

    def _safe_float(self, value: object) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    def _utcnow(self) -> datetime:
        return datetime.now(UTC)
