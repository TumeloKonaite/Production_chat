from pathlib import Path
from typing import Any

import httpx

from app.config import Settings


class LLMConfigurationError(Exception):
    """Raised when LLM configuration is missing or invalid."""


class LLMServiceError(Exception):
    """Raised when the LLM provider request fails safely."""


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "base_system_prompt.md"

    async def generate_response(self, message: str) -> str:
        if not self._settings.openai_api_key:
            raise LLMConfigurationError()

        system_prompt = self._load_system_prompt()
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.openai_model,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        }

        try:
            async with httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                timeout=60.0,
            ) as client:
                response = await client.post("/chat/completions", headers=headers, json=payload)
                if response.status_code >= 400:
                    raise LLMServiceError()

                response_payload = response.json()
        except httpx.HTTPError as exc:
            raise LLMServiceError() from exc
        except ValueError as exc:
            raise LLMServiceError() from exc

        assistant_response = self._extract_response_text(response_payload)
        if not assistant_response:
            raise LLMServiceError()

        return assistant_response

    def _load_system_prompt(self) -> str:
        try:
            return self._prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LLMConfigurationError() from exc

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        message = first_choice.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts).strip()

        return ""
