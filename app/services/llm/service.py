from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from app.config import Settings
from app.services.llm.errors import LLMConfigurationError, LLMServiceError

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_TIMEOUT_SECONDS = 60.0


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
class LLMGeneratedResponse:
    message: str
    model: str
    prompt_version: str
    latency_ms: int | None
    token_usage: TokenUsage


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prompt_path = (
            Path(__file__).resolve().parent.parent.parent / "prompts" / "base_system_prompt.md"
        )

    @property
    def model(self) -> str:
        return self._settings.openai_model

    @property
    def prompt_version(self) -> str:
        return self._settings.prompt_version

    def load_system_prompt(self) -> str:
        return self._load_system_prompt()

    async def generate_response(
        self,
        messages: Sequence[LLMChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> LLMGeneratedResponse:
        api_key = self._get_api_key()
        prompt_text = system_prompt or self._load_system_prompt()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(messages=messages, system_prompt=prompt_text)
        started_at = perf_counter()
        response_payload = await self._request_completion(headers=headers, payload=payload)
        latency_ms = int((perf_counter() - started_at) * 1000)
        assistant_response = self._extract_response_text(response_payload)
        if not assistant_response:
            raise LLMServiceError()

        response_model = response_payload.get("model")
        # Fall back to the configured model name when the provider omits model metadata.
        if not isinstance(response_model, str):
            response_model = self.model

        return LLMGeneratedResponse(
            message=assistant_response,
            model=response_model,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            token_usage=self._extract_token_usage(response_payload),
        )

    def _get_api_key(self) -> str:
        if not self._settings.openai_api_key:
            raise LLMConfigurationError()
        return self._settings.openai_api_key

    def _build_payload(
        self,
        *,
        messages: Sequence[LLMChatMessage],
        system_prompt: str,
    ) -> dict[str, Any]:
        prompt_messages = [{"role": "developer", "content": system_prompt}]
        prompt_messages.extend(
            {"role": message.role, "content": message.content} for message in messages
        )
        return {
            "model": self.model,
            # Keep the prompt on the server so the frontend never handles model instructions or secrets.
            "messages": prompt_messages,
        }

    async def _request_completion(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=OPENAI_BASE_URL,
                timeout=OPENAI_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post("/chat/completions", headers=headers, json=payload)
                if response.status_code >= 400:
                    raise LLMServiceError()

                return response.json()
        except httpx.HTTPError as exc:
            raise LLMServiceError() from exc
        except ValueError as exc:
            raise LLMServiceError() from exc

    def _load_system_prompt(self) -> str:
        try:
            return self._prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LLMConfigurationError() from exc

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        # Support both plain string content and structured content blocks from the provider response.
        content = self._get_message_content(payload)
        return self._extract_content_text(content)

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

    def _extract_content_text(self, content: Any) -> str:
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

    def _extract_token_usage(self, payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return TokenUsage()

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        return TokenUsage(
            input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )
