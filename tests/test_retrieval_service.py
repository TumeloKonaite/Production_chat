from __future__ import annotations

from langchain_core.documents import Document

from app.repositories.models import KnowledgeChunk
from app.services.retrieval import RetrievalService, VectorIndexConfigurationError


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


def build_settings() -> object:
    return type(
        "Settings",
        (),
        {
            "retrieval_top_k": 5,
            "retrieval_min_similarity": 0.6,
            "embedding_provider": "hf",
            "knowledge_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "knowledge_collection_name": "test_collection",
            "database_url": "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
            "openrouter_api_key": None,
            "openrouter_base_url": "https://openrouter.ai/api/v1",
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
        RetrievalService(settings=build_settings(), vectorstore=vectorstore)
    except VectorIndexConfigurationError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected VectorIndexConfigurationError")

    assert "Configured embedding dimension does not match the pgvector storage dimension" in message
    assert "Configured: 384" in message
    assert "Vector store: 1536" in message
