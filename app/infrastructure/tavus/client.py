from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.services.tavus.errors import TavusConfigurationError, TavusServiceError

TAVUS_TIMEOUT_SECONDS = 30.0


class TavusClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def create_conversation(
        self,
        *,
        face_id: str,
        pal_id: str,
        conversational_context: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "face_id": face_id,
            "pal_id": pal_id,
        }
        if conversational_context:
            payload["conversational_context"] = conversational_context

        return await self._post("/v2/conversations", payload)

    async def end_conversation(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        return await self._post(f"/v2/conversations/{conversation_id}/end", {})

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        api_key = self._get_api_key()
        try:
            async with httpx.AsyncClient(
                base_url=self._settings.tavus_base_url.rstrip("/"),
                timeout=TAVUS_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    path,
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise TavusServiceError() from exc
        except ValueError as exc:
            raise TavusServiceError() from exc

    def _get_api_key(self) -> str:
        if not self._settings.tavus_api_key:
            raise TavusConfigurationError("Tavus API key is not configured.")
        return self._settings.tavus_api_key
