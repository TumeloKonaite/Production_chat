from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

import httpx

from app.config import Settings
from app.infrastructure.llm.base import LLMChatMessage, LLMClient, LLMResponse
from app.infrastructure.llm.text_normalization import normalize_llm_text
from app.services.llm.errors import LLMConfigurationError, LLMServiceError

OPENAI_TIMEOUT_SECONDS = 60.0


class OpenAIClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        provider: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> OpenAIClient:
        if provider == "openai":
            return cls(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                transport=transport,
            )
        if provider == "openrouter":
            return cls(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                transport=transport,
            )

        raise LLMConfigurationError(f"Unsupported OpenAI-compatible provider: {provider}")

    async def generate(
        self,
        messages: Sequence[LLMChatMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        api_key = self._get_api_key()
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        started_at = perf_counter()
        response_payload = await self._request_completion(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        assistant_response = self._extract_response_text(response_payload)
        if not assistant_response:
            raise LLMServiceError()

        response_model = response_payload.get("model")
        if not isinstance(response_model, str):
            response_model = model

        usage = response_payload.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

        return LLMResponse(
            content=normalize_llm_text(assistant_response),
            model=response_model,
            input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            latency_ms=latency_ms,
        )

    def _get_api_key(self) -> str:
        if not self._api_key:
            raise LLMConfigurationError()
        return self._api_key

    async def _request_completion(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout_seconds if timeout_seconds is not None else OPENAI_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = await client.post("/chat/completions", headers=headers, json=payload)
                if response.status_code >= 400:
                    raise LLMServiceError()
                return response.json()
        except httpx.HTTPError as exc:
            raise LLMServiceError() from exc
        except ValueError as exc:
            raise LLMServiceError() from exc

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        content = self._get_message_content(payload)
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()

    def _get_message_content(self, payload: dict[str, Any]) -> Any:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None

        message = first_choice.get("message")
        if not isinstance(message, dict):
            return None

        return message.get("content")
