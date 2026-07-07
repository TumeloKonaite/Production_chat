from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActorType = Literal["user", "session", "ip"]


@dataclass(frozen=True, slots=True)
class RateLimitActor:
    actor_id: str
    actor_type: ActorType


@dataclass(frozen=True, slots=True)
class RateLimitLease:
    actor: RateLimitActor
    endpoint: str
    key: str


@dataclass(frozen=True, slots=True)
class ChatRateLimitContext:
    actor: RateLimitActor
    endpoint: str = "chat"
