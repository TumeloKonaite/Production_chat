from __future__ import annotations

import asyncio
import json

import httpx

from app.infrastructure.llm import fetch_openrouter_model_pricing
from evals.run_generation_eval import _merge_model_cost_override


def test_fetch_openrouter_model_pricing_converts_per_token_to_per_million() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/model/openai/gpt-oss-120b"
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "openai/gpt-oss-120b",
                    "pricing": {
                        "prompt": "0.00000015",
                        "completion": "0.0000006",
                    },
                }
            },
        )

    pricing = asyncio.run(
        fetch_openrouter_model_pricing(
            api_key="sk-or-v1-test",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-oss-120b",
            transport=httpx.MockTransport(handler),
        )
    )

    assert pricing.prompt_cost_per_1m_tokens == 0.15
    assert pricing.completion_cost_per_1m_tokens == 0.6


def test_merge_model_cost_override_updates_matching_entry() -> None:
    merged = _merge_model_cost_override(
        json.dumps(
            [
                {
                    "config_id": "openrouter:openai/gpt-oss-120b",
                    "provider": "openrouter",
                    "model": "openai/gpt-oss-120b",
                }
            ]
        ),
        config_id="openrouter:openai/gpt-oss-120b",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_cost_per_1m_tokens=0.15,
        completion_cost_per_1m_tokens=0.6,
    )

    payload = json.loads(merged)
    assert payload == [
        {
            "config_id": "openrouter:openai/gpt-oss-120b",
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "input_cost_per_1m_tokens": 0.15,
            "output_cost_per_1m_tokens": 0.6,
        }
    ]
