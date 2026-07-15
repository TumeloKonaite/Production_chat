from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.infrastructure.llm import JudgeClient, OpenAIClient, normalize_llm_text
from app.infrastructure.llm.base import LLMChatMessage, LLMResponse
from app.services.llm import LLMServiceError
from app.services.llm.service import LLMService


def build_test_settings(
    *,
    openai_base_url: str = "https://api.openai.com/v1",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    default_model_config_id: str = "openai:gpt-4.1-mini",
    model_configs_json: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_prompt_cost_per_1m_tokens: float | None = None,
    llm_completion_cost_per_1m_tokens: float | None = None,
) -> Settings:
    inferred_provider, inferred_model = default_model_config_id.split(":", 1)
    resolved_llm_provider = llm_provider or inferred_provider
    resolved_llm_model = llm_model or inferred_model
    resolved_llm_base_url = llm_base_url or (
        openrouter_base_url if resolved_llm_provider == "openrouter" else openai_base_url
    )

    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_base_url=openai_base_url,
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url=openrouter_base_url,
        tavus_api_key="tavus-test-key",
        tavus_base_url="https://tavus.example",
        tavus_face_id="face_123",
        tavus_pal_id="pal_123",
        public_backend_url="https://backend.example",
        tavus_tool_secret="tool-secret",
        ingestion_api_secret="ingestion-secret",
        eval_admin_token="eval-secret",
        default_model_config_id=default_model_config_id,
        model_configs_json=model_configs_json,
        embedding_provider="hf",
        knowledge_embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version="v1_professional",
        conversation_history_limit=10,
        retriever_type="vector",
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="personal-chatbot-model-comparison",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
        llm_provider=resolved_llm_provider,
        llm_model=resolved_llm_model,
        llm_base_url=resolved_llm_base_url,
        llm_api_key=llm_api_key,
        llm_prompt_cost_per_1m_tokens=llm_prompt_cost_per_1m_tokens,
        llm_completion_cost_per_1m_tokens=llm_completion_cost_per_1m_tokens,
    )


def test_openai_client_uses_configured_base_url() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Configured endpoint response"}}],
                "model": "anthropic/claude-3.5-sonnet",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    client = OpenAIClient.from_settings(
        build_test_settings(openrouter_base_url="https://openrouter.ai/api/v1"),
        provider="openrouter",
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(
        client.generate(
            [LLMChatMessage(role="user", content="Test message")],
            model="anthropic/claude-3.5-sonnet",
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert response.content == "Configured endpoint response"


def test_judge_client_uses_configured_base_url() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "context_relevance": {"score": 2, "reason": "Grounded."},
                                    "faithfulness": {"score": 2, "reason": "Supported."},
                                    "answer_relevance": {"score": 2, "reason": "Relevant."},
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 18,
                    "total_tokens": 42,
                },
            },
        )

    settings = build_test_settings(
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model_config_id="openrouter:anthropic/claude-3.5-sonnet",
        model_configs_json=json.dumps(
            [
                {
                    "config_id": "openrouter:anthropic/claude-3.5-sonnet",
                    "provider": "openrouter",
                    "model": "anthropic/claude-3.5-sonnet",
                    "input_cost_per_1m_tokens": 3.0,
                    "output_cost_per_1m_tokens": 15.0,
                }
            ]
        ),
    )
    client = JudgeClient(settings=settings, transport=httpx.MockTransport(handler))

    evaluation, token_usage, latency_ms, model = asyncio.run(
        client.evaluate(prompt="Judge this answer.")
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert evaluation.context_relevance.score == 2
    assert token_usage.total_tokens == 42
    assert latency_ms >= 0
    assert model == "anthropic/claude-3.5-sonnet"


def test_judge_client_accepts_fenced_json_and_string_scores() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                + json.dumps(
                                    {
                                        "context_relevance": {
                                            "score": "2",
                                            "reason": "Grounded.",
                                        },
                                        "faithfulness": {
                                            "score": "2",
                                            "reason": "Supported.",
                                        },
                                        "answer_relevance": {
                                            "score": "0",
                                            "reason": "Ignored.",
                                        },
                                    }
                                )
                                + "\n```"
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    client = JudgeClient(settings=build_test_settings(), transport=httpx.MockTransport(handler))

    evaluation, token_usage, latency_ms, model = asyncio.run(
        client.evaluate(prompt="Judge this retrieval context.")
    )

    assert evaluation.context_relevance.score == 2
    assert evaluation.answer_relevance.score == 0
    assert token_usage.total_tokens == 30
    assert latency_ms >= 0
    assert model == "gpt-4.1-mini"


def test_llm_service_routes_openrouter_models_to_openrouter_client() -> None:
    class FakeClient:
        def __init__(self, response: LLMResponse) -> None:
            self.response = response
            self.calls: list[dict[str, Any]] = []

        async def generate(
            self,
            messages: list[LLMChatMessage],
            *,
            model: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            self.calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return self.response

    service = LLMService(
        settings=build_test_settings(
            default_model_config_id="openrouter:anthropic/claude-3.5-sonnet",
            model_configs_json=json.dumps(
                [
                    {
                        "config_id": "openrouter:anthropic/claude-3.5-sonnet",
                        "provider": "openrouter",
                        "model": "anthropic/claude-3.5-sonnet",
                        "input_cost_per_1m_tokens": 3.0,
                        "output_cost_per_1m_tokens": 15.0,
                    }
                ]
            ),
        )
    )
    openai_client = FakeClient(
        LLMResponse(content="openai", model="gpt-4.1-mini", latency_ms=100)
    )
    openrouter_client = FakeClient(
        LLMResponse(
            content="hello from openrouter",
            model="anthropic/claude-3.5-sonnet",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            latency_ms=250,
        )
    )
    service._clients = {
        "openai": openai_client,
        "openrouter": openrouter_client,
    }

    response = asyncio.run(
        service.generate_response(
            [LLMChatMessage(role="user", content="Say hello")],
            system_prompt="Be concise.",
            prompt_version="v1_professional",
        )
    )

    assert openai_client.calls == []
    assert len(openrouter_client.calls) == 1
    assert openrouter_client.calls[0]["model"] == "anthropic/claude-3.5-sonnet"
    assert response.message == "hello from openrouter"
    assert response.model_provider == "openrouter"
    assert response.model_config_id == "openrouter:anthropic/claude-3.5-sonnet"


def test_llm_service_builds_default_model_from_generic_llm_settings() -> None:
    service = LLMService(
        settings=build_test_settings(
            default_model_config_id="openrouter:google/gemini-2.5-flash",
            model_configs_json=None,
            llm_provider="openrouter",
            llm_model="google/gemini-2.5-flash",
            llm_base_url="https://openrouter.ai/api/v1",
            llm_api_key="generic-openrouter-key",
            llm_prompt_cost_per_1m_tokens=0.3,
            llm_completion_cost_per_1m_tokens=2.5,
        )
    )

    model = service.get_model_config()

    assert model.config_id == "openrouter:google/gemini-2.5-flash"
    assert model.provider == "openrouter"
    assert model.model == "google/gemini-2.5-flash"
    assert model.input_cost_per_1m_tokens == 0.3
    assert model.output_cost_per_1m_tokens == 2.5
    assert service.get_model_base_url() == "https://openrouter.ai/api/v1"


def test_normalize_llm_text_repairs_mojibake_and_unicode_punctuation() -> None:
    normalized = normalize_llm_text(
        "Hello, welcome to Tumelo Konaite\u00e2\u20ac\u2122s "
        "portfolio\u00e2\u20ac\u201dI\u2019m here to assist\u2026"
    )

    assert normalized == "Hello, welcome to Tumelo Konaite's portfolio-I'm here to assist..."


def test_openai_client_normalizes_response_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Hello, welcome to Tumelo Konaite\u00e2\u20ac\u2122s "
                                "portfolio\u00e2\u20ac\u201dI\u2019m here."
                            )
                        }
                    }
                ],
                "model": "openai/gpt-4.1-mini",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    client = OpenAIClient.from_settings(
        build_test_settings(),
        provider="openai",
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(
        client.generate(
            [LLMChatMessage(role="user", content="Test message")],
            model="gpt-4.1-mini",
        )
    )

    assert response.content == "Hello, welcome to Tumelo Konaite's portfolio-I'm here."


def test_openai_client_surfaces_http_status_and_response_excerpt() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error",
                }
            },
        )

    client = OpenAIClient.from_settings(
        build_test_settings(),
        provider="openai",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMServiceError) as exc_info:
        asyncio.run(
            client.generate(
                [LLMChatMessage(role="user", content="Test message")],
                model="gpt-4.1-mini",
            )
        )

    assert "HTTP 401" in str(exc_info.value)
    assert "https://api.openai.com/v1/chat/completions" in str(exc_info.value)
    assert "Incorrect API key provided." in str(exc_info.value)
