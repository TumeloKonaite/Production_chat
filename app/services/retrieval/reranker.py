from __future__ import annotations

import asyncio
import json
from time import perf_counter

from app.config import Settings
from app.infrastructure.llm import ModelRegistry, OpenAIClient, build_default_model_config
from app.services.llm.errors import LLMConfigurationError, LLMServiceError
from app.services.retrieval.errors import InvalidRerankerResultError
from app.services.retrieval.types import RetrievedChunk

RERANKER_TYPE_NONE = "none"
RERANKER_TYPE_LLM = "llm"
LLM_RERANKER_MAX_TOKENS = 512
LLM_RERANKER_TIMEOUT_SECONDS = 30.0


class NoOpReranker:
    def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        final_top_k: int,
    ) -> list[RetrievedChunk]:
        del question
        return [_clone_chunk(chunk) for chunk in chunks[:final_top_k]]


class LLMReranker:
    def __init__(
        self,
        settings: Settings,
        *,
        clients: dict[str, OpenAIClient] | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._model_registry = model_registry or ModelRegistry(
            default_model_config_id=settings.default_model_config_id,
            model_configs_json=settings.model_configs_json,
            default_model_config=build_default_model_config(settings),
        )
        self._clients = clients or {
            "openai": OpenAIClient.from_settings(settings, provider="openai"),
            "openrouter": OpenAIClient.from_settings(settings, provider="openrouter"),
        }

    def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        final_top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        return asyncio.run(
            self._rerank(
                question=question,
                chunks=chunks,
                final_top_k=final_top_k,
            )
        )

    async def _rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        final_top_k: int,
    ) -> list[RetrievedChunk]:
        model_config = self._model_registry.resolve(self._settings.reranker_model)
        client = self._clients.get(model_config.provider)
        if client is None:
            raise LLMConfigurationError(
                f"No reranker client configured for provider: {model_config.provider}"
            )

        payload = {
            "model": model_config.model,
            "temperature": 0,
            "max_tokens": LLM_RERANKER_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "developer",
                    "content": (
                        "You are a document reranker. Rank the provided chunk IDs from most "
                        "relevant to least relevant for the user question. Return only valid "
                        'JSON in the form {"order": [1, 2, 3]}. Include every chunk ID exactly once.'
                    ),
                },
                {
                    "role": "user",
                    "content": _build_reranker_prompt(question=question, chunks=chunks),
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
                timeout_seconds=LLM_RERANKER_TIMEOUT_SECONDS,
            )
        except LLMConfigurationError:
            raise
        except LLMServiceError as exc:
            raise InvalidRerankerResultError("LLM reranker request failed.") from exc
        latency_ms = int((perf_counter() - started_at) * 1000)

        response_text = client._extract_response_text(response_payload)
        if not response_text:
            raise InvalidRerankerResultError("LLM reranker returned an empty response.")

        order = _parse_reranker_order(response_text, expected_count=len(chunks))
        response_model = response_payload.get("model")
        model_name = response_model if isinstance(response_model, str) else model_config.model

        reranked: list[RetrievedChunk] = []
        for reranker_rank, original_index in enumerate(order, start=1):
            source_chunk = chunks[original_index - 1]
            reranked.append(
                _clone_chunk(
                    source_chunk,
                    metadata_updates={
                        "reranker_rank": reranker_rank,
                        "reranker_type": RERANKER_TYPE_LLM,
                        "reranker_model": model_name,
                        "reranker_latency_ms": latency_ms,
                    },
                )
            )
        return reranked[:final_top_k]


def _build_reranker_prompt(*, question: str, chunks: list[RetrievedChunk]) -> str:
    rendered_chunks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        rendered_chunks.append(
            "\n".join(
                [
                    f"# CHUNK ID: {index}",
                    f"Source: {chunk.source}",
                    f"Section: {chunk.section}",
                    f"Original chunk_id: {chunk.id}",
                    chunk.content,
                ]
            )
        )

    return "\n\n".join(
        [
            f"Question:\n{question.strip()}",
            "Retrieved chunks:",
            "\n\n".join(rendered_chunks),
            "Return the chunk IDs ordered from most relevant to least relevant.",
        ]
    )


def _parse_reranker_order(response_text: str, *, expected_count: int) -> list[int]:
    parsed_response_text = _extract_json_object_text(response_text)
    try:
        payload = json.loads(parsed_response_text)
    except json.JSONDecodeError as exc:
        raise InvalidRerankerResultError(
            f"LLM reranker did not return valid JSON. Response excerpt: {_truncate_response(response_text)!r}"
        ) from exc

    order = payload.get("order")
    normalized_order = _normalize_order_list(order)
    if normalized_order is None:
        raise InvalidRerankerResultError(
            "LLM reranker response must include an integer order list. "
            f"Response excerpt: {_truncate_response(response_text)!r}"
        )

    expected_ids = set(range(1, expected_count + 1))
    actual_ids = set(normalized_order)
    if len(normalized_order) != expected_count or actual_ids != expected_ids:
        raise InvalidRerankerResultError(
            f"Invalid reranker order. Expected IDs {sorted(expected_ids)}, got {normalized_order}."
        )
    return normalized_order


def _normalize_order_list(order: object) -> list[int] | None:
    if not isinstance(order, list):
        return None

    normalized_order: list[int] = []
    for item in order:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            normalized_order.append(item)
            continue
        if isinstance(item, str) and item.strip():
            try:
                normalized_order.append(int(item.strip()))
            except ValueError:
                return None
            continue
        return None
    return normalized_order


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


def _truncate_response(response_text: str, *, limit: int = 200) -> str:
    normalized = " ".join(response_text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _clone_chunk(
    chunk: RetrievedChunk,
    *,
    metadata_updates: dict[str, object] | None = None,
) -> RetrievedChunk:
    metadata = dict(chunk.metadata)
    if metadata_updates:
        metadata.update(metadata_updates)
    return RetrievedChunk(
        id=chunk.id,
        source=chunk.source,
        section=chunk.section,
        content=chunk.content,
        similarity=chunk.similarity,
        metadata=metadata,
    )
