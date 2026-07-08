from __future__ import annotations

from collections.abc import Generator

import pytest

from app.api.dependencies.chat_dependencies import get_rate_limiting_service
from app.main import app
from app.services.rate_limiting.schemas import RateLimitActor
from app.services.rate_limiting.service import RateLimitingService
from tests.test_chat_api import FakeLLMService, build_test_client, build_test_settings


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class FakeCacheClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.incr_calls: list[str] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> bool:
        self.values[key] = value
        self.expirations[key] = ttl_seconds
        return True

    async def delete(self, key: str) -> int:
        existed = key in self.values
        self.values.pop(key, None)
        self.expirations.pop(key, None)
        return int(existed)

    async def incr(self, key: str) -> int:
        self.incr_calls.append(key)
        current = int(self.values.get(key, "0")) + 1
        self.values[key] = str(current)
        return current

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        self.expirations[key] = ttl_seconds
        return True

    async def set_if_not_exists(self, key: str, value: str, ttl_seconds: int) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        self.expirations[key] = ttl_seconds
        return True


def _override_rate_limiter(service: RateLimitingService) -> None:
    app.dependency_overrides[get_rate_limiting_service] = lambda: service


def test_chat_rate_limit_blocks_when_fixed_window_is_exceeded(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Hello.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_redis": True,
            "rate_limit_enabled": True,
            "rate_limit_max_requests": 2,
            "request_lock_enabled": False,
        },
    )
    cache_client = FakeCacheClient()
    rate_limiter = RateLimitingService(
        settings=build_test_settings(
            enable_redis=True,
            rate_limit_enabled=True,
            rate_limit_max_requests=2,
            request_lock_enabled=False,
        ),
        cache_client=cache_client,
    )
    _override_rate_limiter(rate_limiter)

    headers = {"x-user-id": "user-123"}
    first_response = client.post("/chat", json={"message": "Hello 1"}, headers=headers)
    second_response = client.post("/chat", json={"message": "Hello 2"}, headers=headers)
    third_response = client.post("/chat", json={"message": "Hello 3"}, headers=headers)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert third_response.status_code == 429
    assert third_response.json()["detail"] == "Rate limit exceeded. Please try again later."
    assert third_response.json()["retry_after_seconds"] > 0


def test_rate_limiter_uses_user_id_when_available() -> None:
    cache_client = FakeCacheClient()
    service = RateLimitingService(
        settings=build_test_settings(
            enable_redis=True,
            rate_limit_enabled=True,
            rate_limit_max_requests=5,
        ),
        cache_client=cache_client,
    )

    import asyncio

    asyncio.run(
        service.enforce_request_limits(
            actor=RateLimitActor(actor_id="user:hashed", actor_type="user"),
            endpoint="chat",
        )
    )

    assert cache_client.incr_calls
    assert cache_client.incr_calls[0].startswith("rate:chat:user:user:hashed:")


def test_chat_rate_limiter_falls_back_to_ip_when_no_user_id_is_available(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Hello.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_redis": True,
            "rate_limit_enabled": True,
            "request_lock_enabled": False,
        },
    )
    cache_client = FakeCacheClient()
    rate_limiter = RateLimitingService(
        settings=build_test_settings(
            enable_redis=True,
            rate_limit_enabled=True,
            request_lock_enabled=False,
        ),
        cache_client=cache_client,
    )
    _override_rate_limiter(rate_limiter)

    response = client.post(
        "/chat",
        json={"message": "Hello"},
        headers={"x-forwarded-for": "203.0.113.42"},
    )

    assert response.status_code == 200
    assert any(key.startswith("rate:chat:ip:") for key in cache_client.incr_calls)


def test_chat_rate_limiting_is_skipped_when_redis_is_disabled(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Allowed.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_redis": False,
            "rate_limit_enabled": True,
            "request_lock_enabled": False,
        },
    )

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
