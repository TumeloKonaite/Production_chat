from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.services.cache import (
    CacheLookupRequest,
    CacheScope,
    CacheStoreEntry,
    NoOpResponseCache,
    hash_scope,
    normalize_question,
    stable_hash,
)


def test_normalize_question_collapses_whitespace_and_casefolds() -> None:
    assert normalize_question("  What   DOES Tumelo do?  ") == "what does tumelo do?"


def test_hash_scope_is_stable_for_equivalent_payloads() -> None:
    scope = CacheScope(
        knowledge_base_version="kb-v1",
        prompt_version="v1_professional",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash="retriever-hash",
    )

    assert hash_scope(scope) == hash_scope(scope)
    assert stable_hash("abc") == stable_hash("abc")


def test_noop_response_cache_returns_disabled_outcomes() -> None:
    cache = NoOpResponseCache()
    request = CacheLookupRequest(
        request_hash="request-hash",
        normalized_question="what does tumelo do?",
        question_hash="question-hash",
        metadata_scope_hash="scope-hash",
        metadata_scope=CacheScope(
            knowledge_base_version="kb-v1",
            prompt_version="v1_professional",
            llm_provider="openai",
            llm_model="gpt-4.1-mini",
            embedding_provider="hf",
            embedding_model="all-MiniLM-L6-v2",
            retriever_type="vector",
            top_k=5,
            query_rewrite_enabled=False,
            reranker_enabled=False,
            retriever_config_hash="retriever-hash",
        ),
    )
    store_entry = CacheStoreEntry(
        entry_id="entry-1",
        normalized_question=request.normalized_question,
        question_hash=request.question_hash,
        question_embedding=None,
        answer_text="Tumelo builds AI systems.",
        source_documents=[],
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        prompt_version="v1_professional",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        knowledge_base_version="kb-v1",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash="retriever-hash",
        metadata_scope_hash="scope-hash",
        retrieval_config="default",
        created_at=datetime.now(UTC),
        expires_at=None,
        total_latency_ms=1200,
        embedding_latency_ms=15,
        retrieval_latency_ms=120,
        llm_latency_ms=842,
    )

    exact_outcome = asyncio.run(cache.get_exact(request))
    store_outcome = asyncio.run(cache.store(store_entry))

    assert exact_outcome.reason == "disabled"
    assert exact_outcome.hit is False
    assert store_outcome.reason == "write_skipped"
    assert store_outcome.success is False
