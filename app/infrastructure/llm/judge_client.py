from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx

from app.config import Settings
from app.domain.evals import JudgeEvaluation, JudgeMetricScore
from app.infrastructure.llm.base import TokenUsage
from app.infrastructure.llm.model_config import build_default_model_config
from app.infrastructure.llm.model_registry import ModelRegistry
from app.infrastructure.llm.openai_client import OpenAIClient
from app.services.llm.errors import LLMConfigurationError, LLMServiceError


class JudgeConfigurationError(Exception):
    """Raised when judge model configuration is missing or unsupported."""


class JudgeClientError(Exception):
    """Raised when the judge request or response cannot be used safely."""


class JudgeClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model_registry = ModelRegistry(
            default_model_config_id=settings.default_model_config_id,
            model_configs_json=settings.model_configs_json,
            default_model_config=build_default_model_config(settings),
        )
        self._clients = {
            "openai": OpenAIClient.from_settings(
                settings,
                provider="openai",
                transport=transport,
            ),
            "openrouter": OpenAIClient.from_settings(
                settings,
                provider="openrouter",
                transport=transport,
            ),
        }

    async def evaluate(
        self,
        *,
        prompt: str,
        model_config_id: str | None = None,
    ) -> tuple[JudgeEvaluation, TokenUsage, int, str]:
        model_config = self._model_registry.resolve(model_config_id)
        client = self._clients.get(model_config.provider)
        if client is None:
            raise JudgeConfigurationError(
                f"Unsupported judge model provider: {model_config.provider}"
            )
        payload = {
            "model": model_config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "developer",
                    "content": "You are a strict JSON evaluator. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        started_at = perf_counter()
        try:
            response_payload = await client._request_completion(
                headers={
                    "Authorization": f"Bearer {client._get_api_key()}",
                    "Content-Type": "application/json",
                },
                payload=payload,
            )
        except LLMConfigurationError as exc:
            raise JudgeConfigurationError() from exc
        except LLMServiceError as exc:
            raise JudgeClientError() from exc
        latency_ms = int((perf_counter() - started_at) * 1000)
        response_text = self._extract_response_text(response_payload)
        if not response_text:
            raise JudgeClientError()

        usage_payload = response_payload.get("usage")
        prompt_tokens = (
            usage_payload.get("prompt_tokens")
            if isinstance(usage_payload, dict)
            else None
        )
        completion_tokens = (
            usage_payload.get("completion_tokens")
            if isinstance(usage_payload, dict)
            else None
        )
        total_tokens = (
            usage_payload.get("total_tokens")
            if isinstance(usage_payload, dict)
            else None
        )

        try:
            parsed = json.loads(_extract_json_object_text(response_text))
        except json.JSONDecodeError as exc:
            raise JudgeClientError() from exc

        return (
            JudgeEvaluation(
                context_relevance=_parse_metric(parsed, "context_relevance"),
                faithfulness=_parse_metric(parsed, "faithfulness"),
                answer_relevance=_parse_metric(parsed, "answer_relevance"),
            ),
            TokenUsage(
                input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            ),
            latency_ms,
            model_config.model,
        )

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


def _parse_metric(payload: dict[str, Any], key: str) -> JudgeMetricScore:
    metric_payload = payload.get(key)
    if not isinstance(metric_payload, dict):
        raise JudgeClientError()

    score = _normalize_score(metric_payload.get("score"))
    reason = metric_payload.get("reason")
    if score is None or not isinstance(reason, str):
        raise JudgeClientError()
    if score < 0 or score > 2:
        raise JudgeClientError()

    return JudgeMetricScore(score=score, reason=reason.strip())


def _normalize_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_json_object_text(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
