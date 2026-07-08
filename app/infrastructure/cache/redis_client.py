from __future__ import annotations

import json
from typing import Protocol

import httpx

from app.config import Settings
from app.infrastructure.cache.null_cache import NullCacheClient


class CacheClient(Protocol):
    async def get(self, key: str) -> str | None:
        ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        ...

    async def delete(self, key: str) -> int:
        ...

    async def incr(self, key: str) -> int:
        ...

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        ...

    async def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        ...


class UpstashRedisError(RuntimeError):
    pass


class UpstashRedisCacheClient:
    def __init__(
        self,
        *,
        url: str,
        token: str,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    async def get(self, key: str) -> str | None:
        result = await self._execute(["GET", key])
        if result is None:
            return None
        if isinstance(result, str):
            return result
        return json.dumps(result, separators=(",", ":"), sort_keys=True)

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        result = await self._execute(["SET", key, value, "EX", str(_safe_ttl(ttl_seconds))])
        return result == "OK"

    async def delete(self, key: str) -> int:
        result = await self._execute(["DEL", key])
        return _as_int(result)

    async def incr(self, key: str) -> int:
        result = await self._execute(["INCR", key])
        return _as_int(result)

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        result = await self._execute(["EXPIRE", key, str(_safe_ttl(ttl_seconds))])
        return _as_int(result) == 1

    async def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        result = await self._execute(
            ["SET", key, value, "EX", str(_safe_ttl(ttl_seconds)), "NX"]
        )
        return result == "OK"

    async def _execute(self, command: list[str]) -> object:
        response = await self._http_client.post(self._url, json=command)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if error:
                raise UpstashRedisError(str(error))
            return payload.get("result")
        raise UpstashRedisError("Unexpected Upstash Redis response payload.")


def build_cache_client(settings: Settings) -> CacheClient:
    if not settings.enable_redis:
        return NullCacheClient()
    if not settings.upstash_redis_configured:
        raise ValueError(
            "UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN are required when ENABLE_REDIS=true."
        )
    return UpstashRedisCacheClient(
        url=settings.upstash_redis_rest_url or "",
        token=settings.upstash_redis_rest_token or "",
    )


def _safe_ttl(value: int) -> int:
    return max(1, int(value))


def _as_int(value: object) -> int:
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
