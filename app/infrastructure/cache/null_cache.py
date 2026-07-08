from __future__ import annotations


class NullCacheClient:
    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        return False

    async def delete(self, key: str) -> int:
        return 0

    async def incr(self, key: str) -> int:
        return 0

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return False

    async def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        return False
