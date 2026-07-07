from __future__ import annotations

from datetime import datetime, timezone
import os
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, text
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.ingestion.chunker import ChunkedDocument
from app.knowledge.ingestion.service import prepare_knowledge_ingestion_storage
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.models import KnowledgeChunk
from app.services.retrieval import RetrievalService

RUN_DB_INTEGRATION_TESTS = os.getenv("RUN_DB_INTEGRATION_TESTS", "").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not RUN_DB_INTEGRATION_TESTS or not TEST_DATABASE_URL,
    reason="Set RUN_DB_INTEGRATION_TESTS=true and TEST_DATABASE_URL to run Postgres smoke tests.",
)


class DeterministicEmbeddingProvider(Embeddings):
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        normalized = text.casefold()
        vector = [0.0] * self._dimension
        if "retrieval" in normalized or "chatbot" in normalized:
            vector[0] = 1.0
        elif "dashboard" in normalized:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector


def build_settings(*, database_url: str, collection_name: str, dimension: int = 384) -> object:
    return type(
        "Settings",
        (),
        {
            "retriever_type": "vector",
            "retrieval_top_k": 5,
            "retrieval_min_similarity": 0.5,
            "embedding_provider": "openai",
            "knowledge_embedding_model": "text-embedding-3-small",
            "embedding_dimension": dimension,
            "knowledge_collection_name": collection_name,
            "database_url": database_url,
            "vector_store_provider": "supabase_pgvector",
            "openrouter_api_key": None,
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "enable_reranking": False,
            "reranker_type": "none",
            "reranker_initial_top_k": 5,
            "reranker_final_top_k": 5,
            "reranker_model": "openai:gpt-4.1-mini",
        },
    )()


def test_supabase_pgvector_smoke_round_trip() -> None:
    assert TEST_DATABASE_URL is not None
    PGVector = pytest.importorskip("langchain_postgres").PGVector

    collection_name = f"smoke_{uuid4().hex}"
    source_name = f"integration_{uuid4().hex}.md"
    embedding_provider = DeterministicEmbeddingProvider()
    settings = build_settings(database_url=TEST_DATABASE_URL, collection_name=collection_name)
    engine = create_engine(TEST_DATABASE_URL, future=True)
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )

    prepare_knowledge_ingestion_storage(engine)
    vectorstore = PGVector(
        embeddings=embedding_provider,
        collection_name=collection_name,
        connection=TEST_DATABASE_URL,
        embedding_length=settings.embedding_dimension,
        collection_metadata={
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.knowledge_embedding_model,
            "embedding_dimension": settings.embedding_dimension,
        },
        use_jsonb=True,
    )
    retrieval_service = RetrievalService(settings=settings, vectorstore=vectorstore)

    try:
        with session_factory() as session:
            repository = KnowledgeRepository(session)
            stored_chunks = repository.replace_source_chunks(
                source=source_name,
                chunks=[
                    ChunkedDocument(
                        source=source_name,
                        source_type="markdown",
                        section="Portfolio Chatbot",
                        content="Tumelo built a retrieval chatbot with FastAPI and RAG.",
                        metadata={
                            "chunk_index": 0,
                            "source": source_name,
                            "section": "Portfolio Chatbot",
                        },
                        updated_at=datetime.now(timezone.utc),
                    ),
                    ChunkedDocument(
                        source=source_name,
                        source_type="markdown",
                        section="Analytics",
                        content="Tumelo also built internal dashboard tooling.",
                        metadata={
                            "chunk_index": 1,
                            "source": source_name,
                            "section": "Analytics",
                        },
                        updated_at=datetime.now(timezone.utc),
                    ),
                ],
            )

        retrieval_service.replace_all_chunks(list(stored_chunks))
        results = retrieval_service.retrieve(
            "Which project used retrieval chatbot patterns?",
            top_k=1,
            query_embedding=embedding_provider.embed_query("retrieval chatbot"),
        )

        assert len(results) == 1
        assert results[0].source == source_name
        assert results[0].section == "Portfolio Chatbot"
        assert "retrieval chatbot" in results[0].content
        assert results[0].metadata["chunk_index"] == 0
        assert results[0].metadata["source"] == source_name
        assert results[0].metadata["embedding_dimension"] == 384

        with session_factory() as session:
            embedding_type = session.execute(
                text(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
                    FROM pg_attribute AS a
                    JOIN pg_class AS c
                      ON a.attrelid = c.oid
                    WHERE c.relname = 'langchain_pg_embedding'
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    LIMIT 1
                    """
                )
            ).scalar_one()
            indexed_row_count = session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding AS e
                    JOIN langchain_pg_collection AS c
                      ON c.uuid = e.collection_id
                    WHERE c.name = :collection_name
                    """
                ),
                {"collection_name": collection_name},
            ).scalar_one()
            ivfflat_index_present = session.execute(
                text(
                    """
                    SELECT 1
                    FROM pg_indexes
                    WHERE tablename = 'langchain_pg_embedding'
                      AND indexname = 'ix_langchain_pg_embedding_embedding_cosine_ivfflat'
                    """
                )
            ).scalar_one_or_none()

        assert embedding_type == "vector(384)"
        assert indexed_row_count == 2
        assert ivfflat_index_present == 1
    finally:
        vectorstore.delete_collection()
        with session_factory() as session:
            session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source == source_name))
            session.commit()
        engine.dispose()
