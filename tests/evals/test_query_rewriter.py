from __future__ import annotations

from app.config import Settings
from app.infrastructure.llm import LLMResponse
from evals.runners.query_rewriter import (
    QUERY_REWRITE_STATUS_EMPTY_FALLBACK,
    QUERY_REWRITE_STATUS_SUCCESS,
    QueryRewriter,
    render_query_rewrite_prompt,
)


class CapturingClient:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.prompts: list[str] = []

    async def generate(
        self,
        messages,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        del model, temperature, max_tokens, timeout_seconds
        self.prompts.append(messages[0].content)
        return self._response


def test_render_query_rewrite_prompt_stays_conservative_without_explicit_context() -> None:
    prompt = render_query_rewrite_prompt(
        "What does he do?",
        prompt_version="v1",
    )

    assert "What does he do?" in prompt
    assert "Optional context:\nNone" in prompt
    assert "Do not add expected answer terms." in prompt
    assert "Do not invent names, roles, projects, skills, dates, companies, or technologies." in prompt
    assert "Tumelo Konaite" not in prompt
    assert "data scientist" not in prompt
    assert "software engineer" not in prompt


def test_query_rewriter_allows_entity_resolution_from_explicit_context_only() -> None:
    client = CapturingClient(
        LLMResponse(
            content="What does Tumelo Konaite do?",
            model="gpt-4.1-mini",
            input_tokens=80,
            output_tokens=7,
            total_tokens=87,
            latency_ms=412,
        )
    )
    rewriter = QueryRewriter(
        _build_settings(),
        clients={"openai": client, "openrouter": client},
    )

    result = rewriter.rewrite_query(
        "What does he do?",
        context="Subject: Tumelo Konaite",
    )

    assert result.query_rewrite_status == QUERY_REWRITE_STATUS_SUCCESS
    assert result.query_used_for_retrieval == "What does Tumelo Konaite do?"
    assert "Subject: Tumelo Konaite" in client.prompts[0]
    assert "data scientist" not in client.prompts[0]
    assert "software engineer" not in client.prompts[0]


def test_query_rewriter_falls_back_when_the_model_returns_an_empty_query() -> None:
    client = CapturingClient(
        LLMResponse(
            content="   ",
            model="gpt-4.1-mini",
            latency_ms=125,
        )
    )
    rewriter = QueryRewriter(
        _build_settings(),
        clients={"openai": client, "openrouter": client},
    )

    result = rewriter.rewrite_query("What does Tumelo do?")

    assert result.query_rewrite_status == QUERY_REWRITE_STATUS_EMPTY_FALLBACK
    assert result.query_used_for_retrieval == "What does Tumelo do?"
    assert result.rewritten_query is None


def _build_settings() -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        tavus_api_key=None,
        tavus_base_url="https://tavus.example",
        tavus_face_id=None,
        tavus_pal_id=None,
        public_backend_url=None,
        tavus_tool_secret=None,
        ingestion_api_secret=None,
        eval_admin_token=None,
        default_model_config_id="openai:gpt-4.1-mini",
        model_configs_json=None,
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
        enable_query_rewriting=True,
        query_rewrite_model="openai:gpt-4.1-mini",
        query_rewrite_temperature=0.0,
        query_rewrite_prompt_version="v1",
        query_rewrite_timeout_seconds=10,
        query_rewrite_max_tokens=128,
    )
