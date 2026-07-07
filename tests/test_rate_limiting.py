from __future__ import annotations

import asyncio
from collections.abc import Generator
import hashlib

import pytest

from app.api.dependencies.chat_dependencies import get_rate_limiting_service
from app.main import app
from app.services.rate_limiting.schemas import RateLimitActor
from app.services.rate_limiting.service import RateLimitingService
from tests.test_chat_api import (
    FakeLLMService,
    FakeResponseCache,
    build_cache_lookup_result,
    build_test_client,
    build_test_settings,
)
from app.services.cache import ResponseCacheLookupOutcome


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class FakeRedisRateLimitClient:
    def __init__(self, *, fail_methods: set[str] | None = None) -> None:
        self.storage: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.fail_methods = fail_methods or set()

    async def incr(self, key: str) -> int:
        self._maybe_fail("incr")
        value = self._as_int(self.storage.get(key)) + 1
        self.storage[key] = str(value)
        return value

    async def decr(self, key: str) -> int:
        self._maybe_fail("decr")
        value = self._as_int(self.storage.get(key)) - 1
        self.storage[key] = str(value)
        return value

    async def incrby(self, key: str, amount: int) -> int:
        self._maybe_fail("incrby")
        value = self._as_int(self.storage.get(key)) + amount
        self.storage[key] = str(value)
        return value

    async def incrbyfloat(self, key: str, amount: float) -> float:
        self._maybe_fail("incrbyfloat")
        value = self._as_float(self.storage.get(key)) + amount
        self.storage[key] = str(value)
        return value

    async def expire(self, key: str, seconds: int) -> bool:
        self._maybe_fail("expire")
        self.expirations[key] = seconds
        return True

    async def get(self, key: str) -> str | None:
        self._maybe_fail("get")
        return self.storage.get(key)

    async def delete(self, key: str) -> int:
        self._maybe_fail("delete")
        existed = key in self.storage
        self.storage.pop(key, None)
        self.expirations.pop(key, None)
        return int(existed)

    def _maybe_fail(self, method: str) -> None:
        if method in self.fail_methods:
            raise RuntimeError(f"forced failure for {method}")

    def _as_int(self, value: str | None) -> int:
        if value is None:
            return 0
        return int(float(value))

    def _as_float(self, value: str | None) -> float:
        if value is None:
            return 0.0
        return float(value)


def _actor_for_session(session_id: str) -> RateLimitActor:
    digest = hashlib.sha256(f"session:{session_id}".encode("utf-8")).hexdigest()
    return RateLimitActor(actor_id=f"session:{digest[:24]}", actor_type="session")


def _override_rate_limiter(service: RateLimitingService) -> None:
    app.dependency_overrides[get_rate_limiting_service] = lambda: service


def test_chat_request_limit_returns_429_after_limit_exceeded(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="First response.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_requests_per_10_minutes": 1,
            "chat_rate_limit_requests_per_day": 50,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    rate_limiter = RateLimitingService(
        settings=build_test_settings(
            enable_rate_limiting=True,
            chat_rate_limit_requests_per_10_minutes=1,
            chat_rate_limit_requests_per_day=50,
        ),
        redis_client=redis_client,
    )
    _override_rate_limiter(rate_limiter)

    headers = {"x-session-id": "session-1"}
    first_response = client.post("/chat", json={"message": "Hello"}, headers=headers)
    second_response = client.post("/chat", json={"message": "Hello again"}, headers=headers)

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json()["detail"] == "Rate limit exceeded. Please try again later."
    assert second_response.json()["retry_after_seconds"] > 0
    assert second_response.headers["Retry-After"] == str(
        second_response.json()["retry_after_seconds"]
    )


def test_chat_daily_request_limit_returns_429_after_limit_exceeded(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Daily response.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_requests_per_10_minutes": 50,
            "chat_rate_limit_requests_per_day": 1,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    rate_limiter = RateLimitingService(
        settings=build_test_settings(
            enable_rate_limiting=True,
            chat_rate_limit_requests_per_10_minutes=50,
            chat_rate_limit_requests_per_day=1,
        ),
        redis_client=redis_client,
    )
    _override_rate_limiter(rate_limiter)

    headers = {"x-session-id": "session-day"}
    first_response = client.post("/chat", json={"message": "Hello"}, headers=headers)
    second_response = client.post("/chat", json={"message": "Hello again"}, headers=headers)

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json()["detail"] == "Rate limit exceeded. Please try again later."


def test_chat_concurrency_limit_returns_429_when_existing_lease_is_active(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Concurrent response.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_concurrent_requests": 1,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(
        enable_rate_limiting=True,
        chat_rate_limit_concurrent_requests=1,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    actor = _actor_for_session("session-1")
    lease = asyncio.run(rate_limiter.acquire_concurrency_lease(actor=actor, endpoint="chat"))

    response = client.post("/chat", json={"message": "Hello"}, headers={"x-session-id": "session-1"})

    assert response.status_code == 429
    assert response.json()["detail"] == (
        "Too many concurrent chat requests. Please wait for earlier requests to finish."
    )
    assert fake_llm.calls == []

    asyncio.run(rate_limiter.release_concurrency_lease(lease))


def test_chat_concurrency_counter_is_released_after_successful_request(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Success response.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_concurrent_requests": 1,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(
        enable_rate_limiting=True,
        chat_rate_limit_concurrent_requests=1,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post("/chat", json={"message": "Hello"}, headers={"x-session-id": "session-2"})

    actor = _actor_for_session("session-2")
    concurrency_key = rate_limiter._concurrency_key(endpoint="chat", actor=actor)

    assert response.status_code == 200
    assert redis_client.storage.get(concurrency_key) is None


def test_chat_concurrency_counter_is_released_after_failed_request(tmp_path) -> None:
    fake_llm = FakeLLMService(fail=True)
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_concurrent_requests": 1,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(
        enable_rate_limiting=True,
        chat_rate_limit_concurrent_requests=1,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post("/chat", json={"message": "Hello"}, headers={"x-session-id": "session-3"})

    actor = _actor_for_session("session-3")
    concurrency_key = rate_limiter._concurrency_key(endpoint="chat", actor=actor)

    assert response.status_code == 502
    assert redis_client.storage.get(concurrency_key) is None


def test_chat_token_budget_blocks_before_llm_call(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="This should not be returned.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_daily_token_budget": 1000,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(
        enable_rate_limiting=True,
        chat_rate_limit_daily_token_budget=1000,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    actor = _actor_for_session("budget-session")
    token_key = rate_limiter._usage_key(kind="tokens", actor=actor, now=rate_limiter._utcnow())
    redis_client.storage[token_key] = "1000"

    response = client.post(
        "/chat",
        json={"message": "Tell me something expensive."},
        headers={"x-session-id": "budget-session"},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Daily token budget exceeded. Please try again tomorrow."
    assert fake_llm.calls == []


def test_chat_cost_budget_blocks_before_llm_call(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="This should not be returned.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "chat_rate_limit_daily_cost_budget_usd": 0.25,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(
        enable_rate_limiting=True,
        chat_rate_limit_daily_cost_budget_usd=0.25,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    actor = _actor_for_session("cost-session")
    cost_key = rate_limiter._usage_key(kind="cost", actor=actor, now=rate_limiter._utcnow())
    redis_client.storage[cost_key] = "0.25"

    response = client.post(
        "/chat",
        json={"message": "Tell me something expensive."},
        headers={"x-session-id": "cost-session"},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Daily LLM cost budget exceeded. Please try again tomorrow."
    assert fake_llm.calls == []


def test_chat_records_llm_usage_after_successful_llm_call(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Usage recorded.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={"enable_rate_limiting": True},
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(enable_rate_limiting=True)
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post(
        "/chat",
        json={"message": "Hello"},
        headers={"x-session-id": "usage-session"},
    )

    actor = _actor_for_session("usage-session")
    now = rate_limiter._utcnow()
    token_key = rate_limiter._usage_key(kind="tokens", actor=actor, now=now)
    cost_key = rate_limiter._usage_key(kind="cost", actor=actor, now=now)

    assert response.status_code == 200
    assert redis_client.storage[token_key] == "1380"
    assert float(redis_client.storage[cost_key]) == pytest.approx(0.000768)


def test_chat_cache_hit_does_not_increment_usage_budget(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="This should not be used.")
    fake_response_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=3,
            entry=build_cache_lookup_result(),
        )
    )
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        fake_response_cache=fake_response_cache,
        settings_overrides={
            "enable_response_cache": True,
            "enable_rate_limiting": True,
        },
    )
    redis_client = FakeRedisRateLimitClient()
    settings = build_test_settings(enable_rate_limiting=True)
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post(
        "/chat",
        json={"message": "Tell me about Tumelo's work."},
        headers={"x-session-id": "cache-session"},
    )

    actor = _actor_for_session("cache-session")
    now = rate_limiter._utcnow()
    token_key = rate_limiter._usage_key(kind="tokens", actor=actor, now=now)
    cost_key = rate_limiter._usage_key(kind="cost", actor=actor, now=now)

    assert response.status_code == 200
    assert token_key not in redis_client.storage
    assert cost_key not in redis_client.storage
    assert fake_llm.calls == []


def test_chat_rate_limiting_disabled_allows_request_without_redis(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="No limiter.")
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200


def test_chat_rate_limiting_fail_closed_returns_503_when_redis_is_unavailable(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Should not be returned.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "rate_limiting_fail_open": False,
        },
    )
    redis_client = FakeRedisRateLimitClient(fail_methods={"incr"})
    settings = build_test_settings(
        enable_rate_limiting=True,
        rate_limiting_fail_open=False,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post("/chat", json={"message": "Hello"}, headers={"x-session-id": "fail-closed"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Rate limiting backend is unavailable."}
    assert fake_llm.calls == []


def test_chat_rate_limiting_fail_open_allows_request_when_redis_is_unavailable(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Allowed through.")
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_rate_limiting": True,
            "rate_limiting_fail_open": True,
        },
    )
    redis_client = FakeRedisRateLimitClient(fail_methods={"incr"})
    settings = build_test_settings(
        enable_rate_limiting=True,
        rate_limiting_fail_open=True,
    )
    rate_limiter = RateLimitingService(settings=settings, redis_client=redis_client)
    _override_rate_limiter(rate_limiter)

    response = client.post("/chat", json={"message": "Hello"}, headers={"x-session-id": "fail-open"})

    assert response.status_code == 200


def test_api_chat_alias_matches_existing_chat_route(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Alias response.")
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/api/chat", json={"message": "Hello alias"})

    assert response.status_code == 200
    assert response.json()["message"] == "Alias response."
