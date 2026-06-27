from __future__ import annotations

from langchain_core.documents import Document

from app.services.retrieval import RetrievalService


class FakeEmbeddingFunction:
    def embed_query(self, _: str) -> list[float]:
        return [1.0, 0.0, 0.0]


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
        settings=type(
            "Settings",
            (),
            {
                "retrieval_top_k": 5,
                "retrieval_min_similarity": 0.6,
                "knowledge_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "knowledge_collection_name": "test_collection",
                "database_url": "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
            },
        )(),
        vectorstore=vectorstore,
    )

    results = retrieval_service.retrieve("Tell me about the retrieval chatbot")

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].source == "projects.md"
    assert results[0].section == "Projects"
    assert results[0].content == "Tumelo built a FastAPI retrieval chatbot."
    assert results[0].similarity == 0.91
