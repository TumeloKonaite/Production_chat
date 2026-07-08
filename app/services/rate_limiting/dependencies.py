from __future__ import annotations

from collections.abc import AsyncGenerator
import hashlib

from fastapi import Depends, Request

from app.api.dependencies.chat_dependencies import get_rate_limiting_service
from app.services.rate_limiting.schemas import ChatRateLimitContext, RateLimitActor
from app.services.rate_limiting.service import RateLimitingService


async def require_chat_rate_limit(
    request: Request,
    rate_limiting_service: RateLimitingService = Depends(get_rate_limiting_service),
) -> AsyncGenerator[ChatRateLimitContext, None]:
    actor = _build_actor(request=request)
    await rate_limiting_service.enforce_request_limits(actor=actor, endpoint="chat")
    yield ChatRateLimitContext(actor=actor)


def _build_actor(*, request: Request) -> RateLimitActor:
    user_id = _extract_header_value(request, "x-user-id")
    if user_id is not None:
        return RateLimitActor(
            actor_id=_stable_actor_id("user", user_id),
            actor_type="user",
        )

    client_ip = _extract_client_ip(request)
    return RateLimitActor(
        actor_id=_stable_actor_id("ip", client_ip),
        actor_type="ip",
    )


def _extract_client_ip(request: Request) -> str:
    forwarded = _extract_header_value(request, "x-forwarded-for")
    if forwarded:
        first_hop = forwarded.split(",", 1)[0].strip()
        if first_hop:
            return first_hop

    real_ip = _extract_header_value(request, "x-real-ip")
    if real_ip:
        return real_ip

    if request.client is not None and request.client.host:
        return request.client.host

    return "unknown"


def _extract_header_value(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
def _stable_actor_id(actor_type: str, value: str) -> str:
    digest = hashlib.sha256(f"{actor_type}:{value}".encode("utf-8")).hexdigest()
    return f"{actor_type}:{digest[:24]}"
