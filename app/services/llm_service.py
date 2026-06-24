from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_TIMEOUT_SECONDS = 60.0


class LLMConfigurationError(Exception):
    """Raised when LLM configuration is missing or invalid."""


class LLMServiceError(Exception):
    """Raised when the LLM provider request fails safely."""


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prompt_path = (
            Path(__file__).resolve().parent.parent / "prompts" / "base_system_prompt.md"
        )

    async def generate_response(self, message: str) -> str:
        api_key = self._get_api_key()
        system_prompt = self._load_system_prompt()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(message=message, system_prompt=system_prompt)
        response_payload = await self._request_completion(headers=headers, payload=payload)
        assistant_response = self._extract_response_text(response_payload)
        if not assistant_response:
            raise LLMServiceError()

        return assistant_response

    def _get_api_key(self) -> str:
        if not self._settings.openai_api_key:
            raise LLMConfigurationError()
        return self._settings.openai_api_key

    def _build_payload(self, message: str, system_prompt: str) -> dict[str, Any]:
        return {
            "model": self._settings.openai_model,
            # Keep the prompt on the server so the frontend never handles model instructions or secrets.
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": message},
            ],
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
