from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.config import Settings
from app.repositories.models import KnowledgeChunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    source: str
    section: str
    content: str
    similarity: float
    metadata: dict[str, object]


class RetrievalService:
    def __init__(self, settings: Settings, vectorstore: Any | None = None) -> None:
        self._settings = settings
        self._default_top_k = settings.retrieval_top_k
        self._min_similarity = settings.retrieval_min_similarity

        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_postgres import PGVector

            embeddings = HuggingFaceEmbeddings(model_name=settings.knowledge_embedding_model)
            self.vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=settings.knowledge_collection_name,
                connection=settings.database_url,
                use_jsonb=True,
            )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self._default_top_k}
        )

    def replace_all_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self.vectorstore.delete_collection()
        self.vectorstore.create_collection()
        if not chunks:
            return

        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    **chunk.chunk_metadata,
                    "chunk_id": chunk.id,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "section": chunk.section,
                },
            )
            for chunk in chunks
        ]
        self.vectorstore.add_documents(documents=documents, ids=[chunk.id for chunk in chunks])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        limit = top_k or self._default_top_k
        results = self.vectorstore.similarity_search_with_relevance_scores(
            normalized_query,
            k=limit,
        )

        retrieved_chunks: list[RetrievedChunk] = []
        for document, similarity in results:
            if float(similarity) < self._min_similarity:
                continue

            metadata = dict(document.metadata)
            retrieved_chunks.append(
                RetrievedChunk(
                    id=str(metadata.get("chunk_id", "")),
                    source=str(metadata.get("source", "unknown")),
                    section=str(metadata.get("section", "Document")),
                    content=document.page_content,
                    similarity=float(similarity),
                    metadata=metadata,
                )
            )

        return retrieved_chunks
