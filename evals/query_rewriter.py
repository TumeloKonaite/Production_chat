from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter

from app.config import Settings
from app.infrastructure.llm import LLMChatMessage, ModelRegistry, OpenAIClient, TokenUsage

QUERY_REWRITE_STATUS_DISABLED = "disabled"
QUERY_REWRITE_STATUS_SUCCESS = "success"
QUERY_REWRITE_STATUS_EMPTY_FALLBACK = "empty_fallback"
QUERY_REWRITE_STATUS_ERROR_FALLBACK = "error_fallback"
QUERY_REWRITE_STATUS_TIMEOUT_FALLBACK = "timeout_fallback"
QUERY_REWRITE_STATUS_INVALID_RESPONSE_FALLBACK = "invalid_response_fallback"

QUERY_REWRITE_PROMPTS = {
    "v1": """Rewrite the user question into a clear, standalone search query for retrieval.

Rules:
- Preserve the user's intent.
- Do not answer the question.
- Do not add facts that are not present in the original question.
- Do not add expected answer terms.
- Do not invent names, roles, projects, skills, dates, companies, or technologies.
- Only resolve pronouns or vague references when the missing reference is explicitly available in the provided context.
- Return exactly one rewritten query.
- Return plain text only.

Original question:
{query}

Optional context:
{context}
""",
}


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    original_query: str
    rewrite_context: str | None
    rewritten_query: str | None
    query_used_for_retrieval: str
    query_rewriting_enabled: bool
    query_rewrite_status: str
    query_rewrite_model: str | None
    query_rewrite_prompt_version: str | None
    query_rewrite_latency_ms: int | None
    query_rewrite_prompt_tokens: int | None
    query_rewrite_completion_tokens: int | None
    query_rewrite_total_tokens: int | None
    query_rewrite_estimated_cost: float | None
    query_rewrite_error: str | None


class QueryRewriter:
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
        )
        self._clients = clients or {
            "openai": OpenAIClient.from_settings(settings, provider="openai"),
            "openrouter": OpenAIClient.from_settings(settings, provider="openrouter"),
        }

    def rewrite_query(
        self,
        original_query: str,
        *,
        context: str | None = None,
    ) -> QueryRewriteResult:
        if not self._settings.enable_query_rewriting:
            return build_disabled_query_rewrite_result(
                original_query=original_query,
                rewrite_context=context,
            )
        return asyncio.run(self._rewrite_query(original_query, context=context))

    async def _rewrite_query(
        self,
        original_query: str,
        *,
        context: str | None = None,
    ) -> QueryRewriteResult:
        prompt_version = self._settings.query_rewrite_prompt_version
        prompt = render_query_rewrite_prompt(
            original_query,
            context=context,
            prompt_version=prompt_version,
        )
        model_config = self._model_registry.resolve(self._settings.query_rewrite_model)
        client = self._clients.get(model_config.provider)
        if client is None:
            raise ValueError(f"No query rewrite client configured for provider: {model_config.provider}")

        started_at = perf_counter()
        try:
            response = await asyncio.wait_for(
                client.generate(
                    [LLMChatMessage(role="user", content=prompt)],
                    model=model_config.model,
                    temperature=self._settings.query_rewrite_temperature,
                    max_tokens=self._settings.query_rewrite_max_tokens,
                    timeout_seconds=float(self._settings.query_rewrite_timeout_seconds),
                ),
                timeout=float(self._settings.query_rewrite_timeout_seconds) + 1.0,
            )
        except TimeoutError as exc:
            return _build_fallback_result(
                original_query=original_query,
                rewrite_context=context,
                status=QUERY_REWRITE_STATUS_TIMEOUT_FALLBACK,
                query_rewrite_model=model_config.model,
                prompt_version=prompt_version,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error=str(exc) or exc.__class__.__name__,
            )
        except Exception as exc:
            return _build_fallback_result(
                original_query=original_query,
                rewrite_context=context,
                status=QUERY_REWRITE_STATUS_ERROR_FALLBACK,
                query_rewrite_model=model_config.model,
                prompt_version=prompt_version,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error=_stringify_exception(exc),
            )

        normalized_query, invalid_error = _normalize_rewritten_query(response.content)
        estimated_cost = self._model_registry.estimate_cost(
            model_config.config_id,
            TokenUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
        )
        common_kwargs = {
            "original_query": original_query,
            "rewrite_context": _normalize_optional_text(context),
            "query_rewriting_enabled": True,
            "query_rewrite_model": response.model or model_config.model,
            "query_rewrite_prompt_version": prompt_version,
            "query_rewrite_latency_ms": response.latency_ms,
            "query_rewrite_prompt_tokens": response.input_tokens,
            "query_rewrite_completion_tokens": response.output_tokens,
            "query_rewrite_total_tokens": response.total_tokens,
            "query_rewrite_estimated_cost": estimated_cost,
        }
        if normalized_query is None:
            fallback_status = (
                QUERY_REWRITE_STATUS_EMPTY_FALLBACK
                if response.content.strip() == ""
                else QUERY_REWRITE_STATUS_INVALID_RESPONSE_FALLBACK
            )
            return QueryRewriteResult(
                rewritten_query=None,
                query_used_for_retrieval=original_query,
                query_rewrite_status=fallback_status,
                query_rewrite_error=invalid_error,
                **common_kwargs,
            )

        return QueryRewriteResult(
            rewritten_query=normalized_query,
            query_used_for_retrieval=normalized_query,
            query_rewrite_status=QUERY_REWRITE_STATUS_SUCCESS,
            query_rewrite_error=None,
            **common_kwargs,
        )


def build_disabled_query_rewrite_result(
    *,
    original_query: str,
    rewrite_context: str | None = None,
) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=original_query,
        rewrite_context=_normalize_optional_text(rewrite_context),
        rewritten_query=None,
        query_used_for_retrieval=original_query,
        query_rewriting_enabled=False,
        query_rewrite_status=QUERY_REWRITE_STATUS_DISABLED,
        query_rewrite_model=None,
        query_rewrite_prompt_version=None,
        query_rewrite_latency_ms=None,
        query_rewrite_prompt_tokens=None,
        query_rewrite_completion_tokens=None,
        query_rewrite_total_tokens=None,
        query_rewrite_estimated_cost=None,
        query_rewrite_error=None,
    )


def get_query_rewrite_prompt_template(prompt_version: str) -> str:
    normalized_version = prompt_version.strip().casefold()
    template = QUERY_REWRITE_PROMPTS.get(normalized_version)
    if template is None:
        supported_versions = ", ".join(sorted(QUERY_REWRITE_PROMPTS))
        raise ValueError(
            f"Unsupported query rewrite prompt version: {prompt_version}. "
            f"Supported versions: {supported_versions}."
        )
    return template


def render_query_rewrite_prompt(
    original_query: str,
    *,
    context: str | None = None,
    prompt_version: str,
) -> str:
    return get_query_rewrite_prompt_template(prompt_version).format(
        query=original_query.strip(),
        context=_normalize_optional_text(context) or "None",
    )


def _build_fallback_result(
    *,
    original_query: str,
    rewrite_context: str | None,
    status: str,
    query_rewrite_model: str,
    prompt_version: str,
    latency_ms: int,
    error: str,
) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=original_query,
        rewrite_context=_normalize_optional_text(rewrite_context),
        rewritten_query=None,
        query_used_for_retrieval=original_query,
        query_rewriting_enabled=True,
        query_rewrite_status=status,
        query_rewrite_model=query_rewrite_model,
        query_rewrite_prompt_version=prompt_version,
        query_rewrite_latency_ms=latency_ms,
        query_rewrite_prompt_tokens=None,
        query_rewrite_completion_tokens=None,
        query_rewrite_total_tokens=None,
        query_rewrite_estimated_cost=None,
        query_rewrite_error=error,
    )


def _normalize_rewritten_query(value: str) -> tuple[str | None, str | None]:
    text = value.strip()
    if not text:
        return None, "Query rewrite returned an empty response."
    if text.startswith(("```", "{", "[")):
        return None, "Query rewrite must return one plain-text query."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "Query rewrite must return exactly one query."

    normalized = " ".join(lines[0].split())
    if not normalized:
        return None, "Query rewrite returned an empty response."
    return normalized, None


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _stringify_exception(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
