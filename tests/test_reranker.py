from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.services.retrieval import InvalidRerankerResultError, RetrievedChunk
from app.services.retrieval.reranker import LLMReranker, NoOpReranker


def build_settings(
    *,
    default_model_config_id: str = "openai:gpt-4.1-mini",
    model_configs_json: str | None = None,
) -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        ingestion_api_secret=None,
        eval_admin_token=None,
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
        mlflow_experiment_name="test-experiment",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
        enable_reranking=True,
        reranker_type="llm",
        reranker_model=default_model_config_id,
        reranker_initial_top_k=10,
        reranker_final_top_k=5,
    )


def build_chunk(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=source,
        section="Section",
        content=f"content from {source}",
        similarity=0.9,
        metadata={"chunk_id": chunk_id, "source": source},
    )


def test_noop_reranker_preserves_original_order() -> None:
    reranker = NoOpReranker()

    results = reranker.rerank(
        question="Tell me about Tumelo",
        chunks=[
            build_chunk("chunk-1", "projects.md"),
            build_chunk("chunk-2", "skills.md"),
        ],
        final_top_k=1,
    )

    assert [chunk.id for chunk in results] == ["chunk-1"]
    assert results[0].metadata["chunk_id"] == "chunk-1"


def test_llm_reranker_returns_requested_final_top_k_with_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"order":[3,1,2]}'}}],
                "model": "gpt-4.1-mini",
            },
        )

    settings = build_settings()
    reranker = LLMReranker(
        settings=settings,
        clients={
            "openai": httpx_client_to_openai_client(settings, "openai", handler),
            "openrouter": httpx_client_to_openai_client(settings, "openrouter", handler),
        },
    )

    results = reranker.rerank(
        question="Tell me about Tumelo",
        chunks=[
            build_chunk("chunk-1", "projects.md"),
            build_chunk("chunk-2", "skills.md"),
            build_chunk("chunk-3", "profile.md"),
        ],
        final_top_k=2,
    )

    assert [chunk.id for chunk in results] == ["chunk-3", "chunk-1"]
    assert results[0].metadata["reranker_rank"] == 1
    assert results[0].metadata["reranker_type"] == "llm"
    assert results[0].metadata["reranker_model"] == "gpt-4.1-mini"


def test_llm_reranker_accepts_string_ids_inside_fenced_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '```json\n{"order":["3","1","2"]}\n```'}}],
                "model": "gpt-4.1-mini",
            },
        )

    settings = build_settings()
    reranker = LLMReranker(
        settings=settings,
        clients={
            "openai": httpx_client_to_openai_client(settings, "openai", handler),
            "openrouter": httpx_client_to_openai_client(settings, "openrouter", handler),
        },
    )

    results = reranker.rerank(
        question="Tell me about Tumelo",
        chunks=[
            build_chunk("chunk-1", "projects.md"),
            build_chunk("chunk-2", "skills.md"),
            build_chunk("chunk-3", "profile.md"),
        ],
        final_top_k=2,
    )

    assert [chunk.id for chunk in results] == ["chunk-3", "chunk-1"]


def test_llm_reranker_rejects_missing_or_duplicate_chunk_ids() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"order":[1,1,2]}'}}],
                "model": "gpt-4.1-mini",
            },
        )

    settings = build_settings()
    reranker = LLMReranker(
        settings=settings,
        clients={
            "openai": httpx_client_to_openai_client(settings, "openai", handler),
            "openrouter": httpx_client_to_openai_client(settings, "openrouter", handler),
        },
    )

    try:
        reranker.rerank(
            question="Tell me about Tumelo",
            chunks=[
                build_chunk("chunk-1", "projects.md"),
                build_chunk("chunk-2", "skills.md"),
                build_chunk("chunk-3", "profile.md"),
            ],
            final_top_k=2,
        )
    except InvalidRerankerResultError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected InvalidRerankerResultError")

    assert "Invalid reranker order" in message


def httpx_client_to_openai_client(
    settings: Settings,
    provider: str,
    handler,
):
    from app.infrastructure.llm import OpenAIClient

    return OpenAIClient.from_settings(
        settings,
        provider=provider,
        transport=httpx.MockTransport(handler),
    )
