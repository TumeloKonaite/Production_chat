from __future__ import annotations

from langchain_core.documents import Document

from app.repositories.models import KnowledgeChunk
from app.services.retrieval import (
    RetrievalService,
    UnsupportedRetrieverError,
    VectorIndexConfigurationError,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.results: list[tuple[Document, float]] = []
        self.deleted_filters: list[dict[str, object]] = []
        self.added_documents: list[Document] = []
        self.added_ids: list[str] = []
        self.collection_deleted = False
        self.collection_created = False

    def as_retriever(self, **_: object) -> "FakeVectorStore":
        return self

    def delete_collection(self) -> None:
        self.collection_deleted = True

    def create_collection(self) -> None:
        self.collection_created = True

    def delete(self, where: dict[str, object]) -> None:
        self.deleted_filters.append(where)

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.added_documents.extend(documents)
        self.added_ids.extend(ids)

    def similarity_search_with_relevance_scores(
        self,
        _: str,
        *,
        k: int,
    ) -> list[tuple[Document, float]]:
        return self.results[:k]


class FakeCollection:
    def __init__(self, uuid: str, cmetadata: dict[str, object] | None) -> None:
        self.uuid = uuid
        self.cmetadata = cmetadata


class MetadataAwareVectorStore(FakeVectorStore):
    def __init__(
        self,
        *,
        collection_metadata: dict[str, object] | None,
        document_count: int = 1,
        vector_dimension: str | None = "vector(384)",
    ) -> None:
        super().__init__()
        self._collection = FakeCollection("collection-1", collection_metadata)
        self._document_count = document_count
        self._vector_dimension = vector_dimension

    def _make_sync_session(self) -> "MetadataAwareVectorStore":
        return self

    def __enter__(self) -> "MetadataAwareVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_collection(self, _: object) -> FakeCollection:
        return self._collection

    def get_indexed_document_count(self, _: object) -> int:
        return self._document_count

    def get_vector_store_dimension(self) -> int | None:
        if self._vector_dimension is None:
            return None

        if not self._vector_dimension.startswith("vector(") or not self._vector_dimension.endswith(")"):
            return None
        return int(self._vector_dimension.removeprefix("vector(").removesuffix(")"))


class FakeKnowledgeRepository:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)

    def list_all(self) -> list[KnowledgeChunk]:
        return list(self._chunks)


class FakeReranker:
    def __init__(self, ordered_ids: list[str] | None = None) -> None:
        self.ordered_ids = ordered_ids
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        *,
        question: str,
        chunks: list[object],
        final_top_k: int,
    ) -> list[object]:
        self.calls.append(
            {
                "question": question,
                "chunk_ids": [chunk.id for chunk in chunks],
                "final_top_k": final_top_k,
            }
        )
        if self.ordered_ids is None:
            return list(chunks)[:final_top_k]
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        return [chunk_by_id[chunk_id] for chunk_id in self.ordered_ids[:final_top_k]]


def build_settings(
    *,
    retriever_type: str = "vector",
    enable_reranking: bool = False,
    reranker_type: str = "none",
    reranker_initial_top_k: int = 20,
    reranker_final_top_k: int = 5,
) -> object:
    return type(
        "Settings",
        (),
        {
            "retriever_type": retriever_type,
            "retrieval_top_k": 5,
            "retrieval_min_similarity": 0.6,
            "embedding_provider": "hf",
            "knowledge_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "knowledge_collection_name": "test_collection",
            "database_url": "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
            "openrouter_api_key": None,
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "enable_reranking": enable_reranking,
            "reranker_type": reranker_type,
            "reranker_initial_top_k": reranker_initial_top_k,
            "reranker_final_top_k": reranker_final_top_k,
            "reranker_model": "openai:gpt-4.1-mini",
        },
    )()


def test_retrieval_service_returns_only_relevant_chunks() -> None:
    vectorstore = FakeVectorStore()
    vectorstore.results = [
        (
            Document(
                page_content="Tumelo built a FastAPI retrieval chatbot.",
                metadata={
                    "chunk_id": "chunk-1",
                    "source": "projects.md",
                    "section": "Projects",
                },
            ),
            0.91,
        ),
        (
            Document(
                page_content="Tumelo also worked on dashboards.",
                metadata={
                    "chunk_id": "chunk-2",
                    "source": "projects.md",
                    "section": "Other",
                },
            ),
            0.24,
        ),
    ]
    retrieval_service = RetrievalService(
        settings=build_settings(),
        vectorstore=vectorstore,
    )

    results = retrieval_service.retrieve("Tell me about the retrieval chatbot")

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].source == "projects.md"
    assert results[0].section == "Projects"
    assert results[0].content == "Tumelo built a FastAPI retrieval chatbot."
    assert results[0].similarity == 0.91


def test_retrieval_service_keyword_strategy_prefers_project_name_matches() -> None:
    retrieval_service = RetrievalService(
        settings=build_settings(retriever_type="keyword"),
        knowledge_repository=FakeKnowledgeRepository(
            [
                KnowledgeChunk(
                    id="chunk-1",
                    source="experience.md",
                    source_type="markdown",
                    section="Engineering Experience Themes",
                    content="Tumelo has experience building production-grade AI systems.",
                    chunk_metadata={},
                ),
                KnowledgeChunk(
                    id="chunk-2",
                    source="projects.md",
                    source_type="markdown",
                    section="BeautyVerse - Beauty Services Marketplace",
                    content="BeautyVerse is a marketplace for beauty service providers and customers.",
                    chunk_metadata={},
                ),
            ]
        ),
    )

    results = retrieval_service.retrieve("Tell me about Tumelo's BeautyVerse project")

    assert retrieval_service.retriever_type == "keyword"
    assert [chunk.id for chunk in results] == ["chunk-2"]
    assert results[0].source == "projects.md"
    assert "BeautyVerse" in results[0].section


def test_retrieval_service_hybrid_strategy_merges_and_deduplicates_results() -> None:
    vectorstore = FakeVectorStore()
    vectorstore.results = [
        (
            Document(
                page_content="BeautyVerse is a marketplace for beauty services.",
                metadata={
                    "chunk_id": "chunk-2",
                    "source": "projects.md",
                    "section": "BeautyVerse - Beauty Services Marketplace",
                },
            ),
            0.83,
        ),
        (
            Document(
                page_content="Tumelo has built production-grade AI systems.",
                metadata={
                    "chunk_id": "chunk-1",
                    "source": "experience.md",
                    "section": "Engineering Experience Themes",
                },
            ),
            0.79,
        ),
    ]
    retrieval_service = RetrievalService(
        settings=build_settings(retriever_type="hybrid"),
        vectorstore=vectorstore,
        knowledge_repository=FakeKnowledgeRepository(
            [
                KnowledgeChunk(
                    id="chunk-2",
                    source="projects.md",
                    source_type="markdown",
                    section="BeautyVerse - Beauty Services Marketplace",
                    content="BeautyVerse is a marketplace for beauty service providers and customers.",
                    chunk_metadata={},
                ),
                KnowledgeChunk(
                    id="chunk-3",
                    source="skills.md",
                    source_type="markdown",
                    section="BeautyVerse Skills",
                    content="Tumelo used FastAPI, retrieval, and embeddings in BeautyVerse.",
                    chunk_metadata={},
                ),
            ]
        ),
    )

    results = retrieval_service.retrieve("Tell me about Tumelo's BeautyVerse project", top_k=3)

    assert retrieval_service.retriever_type == "hybrid"
    assert [chunk.id for chunk in results] == ["chunk-2", "chunk-3", "chunk-1"]
    assert results[0].similarity <= 0.99


def test_replace_all_chunks_records_embedding_metadata_on_vector_documents() -> None:
    vectorstore = FakeVectorStore()
    retrieval_service = RetrievalService(settings=build_settings(), vectorstore=vectorstore)

    retrieval_service.replace_all_chunks(
        [
            KnowledgeChunk(
                id="chunk-1",
                source="projects.md",
                source_type="markdown",
                section="Projects",
                content="Tumelo built a FastAPI retrieval chatbot.",
                chunk_metadata={"chunk_index": 0},
            )
        ]
    )

    assert vectorstore.collection_deleted is True
    assert vectorstore.collection_created is True
    assert len(vectorstore.added_documents) == 1
    assert vectorstore.added_documents[0].metadata["embedding_provider"] == "hf"
    assert vectorstore.added_documents[0].metadata["embedding_model"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert vectorstore.added_documents[0].metadata["embedding_dimension"] == 384


def test_retrieval_service_rejects_collection_metadata_mismatches() -> None:
    vectorstore = MetadataAwareVectorStore(
        collection_metadata={
            "embedding_provider": "openrouter",
            "embedding_model": "openai/text-embedding-3-small",
            "embedding_dimension": 1536,
        }
    )
    retrieval_service = RetrievalService(settings=build_settings(), vectorstore=vectorstore)

    try:
        retrieval_service.retrieve("Tell me about the retrieval chatbot")
    except VectorIndexConfigurationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected VectorIndexConfigurationError")

    assert "Configured embedding provider/model does not match the existing vector index" in message
    assert "Configured: hf/sentence-transformers/all-MiniLM-L6-v2/384" in message
    assert "Indexed: openrouter/openai/text-embedding-3-small/1536" in message


def test_retrieval_service_rejects_vector_dimension_mismatches() -> None:
    vectorstore = MetadataAwareVectorStore(
        collection_metadata={
            "embedding_provider": "hf",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
        },
        vector_dimension="vector(1536)",
    )

    try:
        retrieval_service = RetrievalService(settings=build_settings(), vectorstore=vectorstore)
        retrieval_service.retrieve("Tell me about the retrieval chatbot")
    except VectorIndexConfigurationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected VectorIndexConfigurationError")

    assert "Configured embedding dimension does not match the pgvector storage dimension" in message
    assert "Configured: 384" in message
    assert "Vector store: 1536" in message


def test_retrieval_service_rejects_unsupported_retriever_type() -> None:
    try:
        RetrievalService(settings=build_settings(retriever_type="semantic"))
    except UnsupportedRetrieverError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected UnsupportedRetrieverError")

    assert message == "Unsupported retriever type: semantic."


def test_retrieval_service_reranks_retrieved_candidates_before_returning_final_top_k() -> None:
    vectorstore = FakeVectorStore()
    vectorstore.results = [
        (
            Document(
                page_content="Chunk one",
                metadata={"chunk_id": "chunk-1", "source": "projects.md", "section": "One"},
            ),
            0.91,
        ),
        (
            Document(
                page_content="Chunk two",
                metadata={"chunk_id": "chunk-2", "source": "skills.md", "section": "Two"},
            ),
            0.89,
        ),
        (
            Document(
                page_content="Chunk three",
                metadata={"chunk_id": "chunk-3", "source": "profile.md", "section": "Three"},
            ),
            0.88,
        ),
    ]
    reranker = FakeReranker(ordered_ids=["chunk-3", "chunk-1", "chunk-2"])
    retrieval_service = RetrievalService(
        settings=build_settings(
            enable_reranking=True,
            reranker_type="llm",
            reranker_initial_top_k=3,
            reranker_final_top_k=1,
        ),
        vectorstore=vectorstore,
        reranker=reranker,
    )

    results = retrieval_service.retrieve("Tell me about Tumelo", top_k=1)

    assert reranker.calls == [
        {"question": "Tell me about Tumelo", "chunk_ids": ["chunk-1", "chunk-2", "chunk-3"], "final_top_k": 1}
    ]
    assert [chunk.id for chunk in results] == ["chunk-3"]
    assert results[0].metadata["retrieval_rank"] == 3
    assert results[0].metadata["final_rank"] == 1
