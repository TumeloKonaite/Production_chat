from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

OPENROUTER_PRICING_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class OpenRouterPricing:
    prompt_cost_per_1m_tokens: float | None
    completion_cost_per_1m_tokens: float | None


class OpenRouterPricingLookupError(RuntimeError):
    """Raised when OpenRouter model pricing cannot be fetched safely."""


async def fetch_openrouter_model_pricing(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenRouterPricing:
    if not api_key or not api_key.strip():
        raise OpenRouterPricingLookupError(
            "OpenRouter pricing lookup requires an API key for the active provider."
        )

    normalized_base_url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(
            base_url=normalized_base_url,
            timeout=OPENROUTER_PRICING_TIMEOUT_SECONDS,
            transport=transport,
        ) as client:
            response = await client.get(
                f"/model/{model}",
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
            )
    except httpx.HTTPError as exc:
        raise OpenRouterPricingLookupError(
            f"OpenRouter pricing lookup failed: {type(exc).__name__}."
        ) from exc

    if response.status_code >= 400:
        sanitized_excerpt = " ".join(response.text.split())
        if len(sanitized_excerpt) > 300:
            sanitized_excerpt = sanitized_excerpt[:297] + "..."
        raise OpenRouterPricingLookupError(
            f"OpenRouter pricing lookup returned HTTP {response.status_code}. "
            f"Response excerpt: {sanitized_excerpt}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenRouterPricingLookupError(
            "OpenRouter pricing lookup returned invalid JSON."
        ) from exc

    data = payload.get("data")
    if not isinstance(data, dict):
        raise OpenRouterPricingLookupError(
            "OpenRouter pricing lookup response did not include a data object."
        )

    pricing = data.get("pricing")
    if not isinstance(pricing, dict):
        raise OpenRouterPricingLookupError(
            "OpenRouter pricing lookup response did not include pricing metadata."
        )

    return OpenRouterPricing(
        prompt_cost_per_1m_tokens=_per_token_to_per_million(pricing.get("prompt")),
        completion_cost_per_1m_tokens=_per_token_to_per_million(pricing.get("completion")),
    )


def _per_token_to_per_million(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return round(float(Decimal(value.strip()) * Decimal(1_000_000)), 6)
    except (InvalidOperation, ValueError):
        return None
