from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LLMChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int = 0


class LLMClient(Protocol):
    async def generate(
        self,
        messages: Sequence[LLMChatMessage],
        *,
        model: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        ...
