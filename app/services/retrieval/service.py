from __future__ import annotations

from collections.abc import Callable, Sequence
import re
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import func, select, text

from app.config import Settings
from app.infrastructure.embeddings import EmbeddingDescriptor, create_embedding_provider
from app.repositories import KnowledgeRepository
from app.repositories.db.session import get_session_factory
from app.repositories.models import KnowledgeChunk
from app.services.retrieval.errors import (
    UnsupportedRerankerError,
    UnsupportedRetrieverError,
    VectorIndexConfigurationError,
)
from app.services.retrieval.reranker import (
    LLMReranker,
    NoOpReranker,
    RERANKER_TYPE_LLM,
    RERANKER_TYPE_NONE,
)
from app.services.retrieval.strategies import HybridRetriever, KeywordRetriever, VectorRetriever
from app.services.retrieval.types import RetrievalResult, RetrievedChunk, Retriever

VECTOR_TYPE_PATTERN = re.compile(r"vector\((\d+)\)")


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vectorstore: Any | None = None,
        knowledge_repository: KnowledgeRepository | None = None,
        reranker: Any | None = None,
    ) -> None:
        self._settings = settings
        self._default_top_k = settings.retrieval_top_k
        self._min_similarity = settings.retrieval_min_similarity
        self._retriever_type = settings.retriever_type
        self._reranker_enabled = bool(getattr(settings, "enable_reranking", False))
        self._configured_reranker_type = str(
            getattr(settings, "reranker_type", RERANKER_TYPE_NONE)
        ).casefold()
        self._reranker_initial_top_k = int(
            getattr(settings, "reranker_initial_top_k", self._default_top_k)
        )
        self._default_final_top_k = int(
            getattr(settings, "reranker_final_top_k", self._default_top_k)
        )
        self._reranker_model = getattr(settings, "reranker_model", None)
        self._embedding_descriptor = EmbeddingDescriptor(
            provider=settings.embedding_provider,
            model=settings.knowledge_embedding_model,
            dimension=settings.embedding_dimension,
        )
        self._vectorstore = vectorstore
        self._chunk_loader = self._build_chunk_loader(knowledge_repository)
        self._retriever = self._build_retriever()
        self._reranker = reranker or self._build_reranker()

    @property
    def retriever_type(self) -> str:
        return self._retriever_type

    @property
    def embedding_descriptor(self) -> EmbeddingDescriptor:
        return self._embedding_descriptor

    @property
    def embedding_metadata(self) -> dict[str, object]:
        return self._embedding_descriptor.as_metadata()

    @property
    def vector_store_name(self) -> str | None:
        if self._retriever_type not in {"vector", "hybrid"}:
            return None
        return "pgvector"

    @property
    def vectorstore(self) -> Any:
        if self._vectorstore is None:
            from langchain_postgres import PGVector

            embedding_provider = create_embedding_provider(self._settings)
            self._vectorstore = PGVector(
                embeddings=embedding_provider,
                collection_name=self._settings.knowledge_collection_name,
                connection=self._settings.database_url,
                embedding_length=self._settings.embedding_dimension,
                collection_metadata=self.embedding_metadata,
                use_jsonb=True,
            )
        return self._vectorstore

    def get_vector_store_dimension(self) -> int | None:
        return self._get_vector_column_dimension()

    def replace_all_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self._ensure_vector_dimension_is_supported()
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
                    **self.embedding_metadata,
                },
            )
            for chunk in chunks
        ]
        self.vectorstore.add_documents(documents=documents, ids=[chunk.id for chunk in chunks])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_with_diagnostics(query, top_k=top_k).final_chunks

    def retrieve_with_diagnostics(self, query: str, top_k: int | None = None) -> RetrievalResult:
        final_top_k = top_k if top_k is not None else self._default_final_top_k
        initial_top_k = self._resolve_initial_top_k(final_top_k)
        initial_chunks = self._with_retrieval_ranks(
            self._retriever.retrieve(query, top_k=initial_top_k)
        )
        final_chunks = self._with_final_ranks(
            self._reranker.rerank(
                question=query,
                chunks=initial_chunks,
                final_top_k=final_top_k,
            )
        )
        return RetrievalResult(
            query=query,
            initial_chunks=initial_chunks,
            final_chunks=final_chunks,
            reranker_enabled=self.reranker_enabled,
            reranker_type=self.reranker_type,
            reranker_model=self._resolved_reranker_model(),
            initial_top_k=initial_top_k,
            final_top_k=final_top_k,
        )

    @property
    def reranker_enabled(self) -> bool:
        return self._reranker_enabled and self._configured_reranker_type != RERANKER_TYPE_NONE

    @property
    def reranker_type(self) -> str:
        if not self.reranker_enabled:
            return RERANKER_TYPE_NONE
        return self._configured_reranker_type

    def _build_retriever(self) -> Retriever:
        vector_retriever = VectorRetriever(
            default_top_k=self._default_top_k,
            search=self._run_vector_search,
        )
        if self._retriever_type == "vector":
            return vector_retriever

        keyword_retriever = KeywordRetriever(
            default_top_k=self._default_top_k,
            chunk_loader=self._chunk_loader,
        )
        if self._retriever_type == "keyword":
            return keyword_retriever
        if self._retriever_type == "hybrid":
            return HybridRetriever(
                default_top_k=self._default_top_k,
                vector_retriever=vector_retriever,
                keyword_retriever=keyword_retriever,
            )

        raise UnsupportedRetrieverError(
            f"Unsupported retriever type: {self._retriever_type}."
        )

    def _build_reranker(self) -> Any:
        if not self._reranker_enabled or self._configured_reranker_type == RERANKER_TYPE_NONE:
            return NoOpReranker()
        if self._configured_reranker_type == RERANKER_TYPE_LLM:
            return LLMReranker(settings=self._settings)
        raise UnsupportedRerankerError(
            f"Unsupported reranker type: {self._configured_reranker_type}."
        )

    def _build_chunk_loader(
        self,
        knowledge_repository: KnowledgeRepository | None,
    ) -> Callable[[], Sequence[KnowledgeChunk]]:
        if knowledge_repository is not None:
            return lambda: knowledge_repository.list_all()

        def load_chunks() -> Sequence[KnowledgeChunk]:
            session = get_session_factory()()
            try:
                repository = KnowledgeRepository(session=session)
                return repository.list_all()
            finally:
                session.close()

        return load_chunks

    def _run_vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_index_compatible()
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=top_k,
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

    def _resolve_initial_top_k(self, final_top_k: int) -> int:
        if not self.reranker_enabled:
            return final_top_k
        return max(final_top_k, self._reranker_initial_top_k)

    def _resolved_reranker_model(self) -> str | None:
        if not self.reranker_enabled:
            return None
        if self.reranker_type == RERANKER_TYPE_LLM and isinstance(self._reranker_model, str):
            return self._reranker_model
        return None

    def _with_retrieval_ranks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ranked_chunks: list[RetrievedChunk] = []
        for rank, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata)
            metadata["retrieval_rank"] = rank
            ranked_chunks.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    similarity=chunk.similarity,
                    metadata=metadata,
                )
            )
        return ranked_chunks

    def _with_final_ranks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ranked_chunks: list[RetrievedChunk] = []
        for rank, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata)
            metadata["final_rank"] = rank
            ranked_chunks.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    similarity=chunk.similarity,
                    metadata=metadata,
                )
            )
        return ranked_chunks

    def _ensure_index_compatible(self) -> None:
        self._ensure_vector_dimension_is_supported()
        if not self._supports_collection_metadata():
            return

        with self.vectorstore._make_sync_session() as session:
            collection = self.vectorstore.get_collection(session)
            if collection is None:
                return

            metadata = collection.cmetadata or {}
            indexed_document_count = self._get_indexed_document_count(
                session=session,
                collection_uuid=collection.uuid,
            )
            if not indexed_document_count:
                return

            indexed_descriptor = self._descriptor_from_metadata(metadata)
            if indexed_descriptor is None:
                raise VectorIndexConfigurationError(
                    "The existing vector index is missing embedding metadata. "
                    "Rebuild the knowledge index before running retrieval."
                )

            if indexed_descriptor != self._embedding_descriptor:
                raise VectorIndexConfigurationError(
                    "Configured embedding provider/model does not match the existing vector index. "
                    f"Configured: {self._embedding_descriptor.as_config_string()}. "
                    f"Indexed: {indexed_descriptor.as_config_string()}. "
                    "Rebuild the knowledge index before running retrieval."
                )

    def _ensure_vector_dimension_is_supported(self) -> None:
        actual_dimension = self._get_vector_column_dimension()
        if actual_dimension is None:
            return

        if actual_dimension != self._embedding_descriptor.dimension:
            raise VectorIndexConfigurationError(
                "Configured embedding dimension does not match the pgvector storage dimension. "
                f"Configured: {self._embedding_descriptor.dimension}. "
                f"Vector store: {actual_dimension}. "
                "Update EMBEDDING_DIMENSION and rebuild the knowledge index, or run the required "
                "database migration before using this embedding configuration."
            )

    def _get_vector_column_dimension(self) -> int | None:
        if not self._supports_collection_metadata():
            return None

        custom_dimension = getattr(self.vectorstore, "get_vector_store_dimension", None)
        if callable(custom_dimension):
            value = custom_dimension()
            return int(value) if isinstance(value, int) else None

        query = text(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
            FROM pg_attribute AS a
            JOIN pg_class AS c
              ON a.attrelid = c.oid
            JOIN pg_namespace AS n
              ON c.relnamespace = n.oid
            WHERE c.relname = 'langchain_pg_embedding'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END, n.nspname
            LIMIT 1
            """
        )
        with self.vectorstore._make_sync_session() as session:
            embedding_type = session.execute(query).scalar_one_or_none()

        if not isinstance(embedding_type, str):
            return None

        match = VECTOR_TYPE_PATTERN.fullmatch(embedding_type.strip())
        if match is None:
            return None
        return int(match.group(1))

    def _supports_collection_metadata(self) -> bool:
        if not all(
            hasattr(self.vectorstore, attribute)
            for attribute in ("_make_sync_session", "get_collection")
        ):
            return False

        return hasattr(self.vectorstore, "EmbeddingStore") or callable(
            getattr(self.vectorstore, "get_indexed_document_count", None)
        )

    def _get_indexed_document_count(self, *, session: Any, collection_uuid: object) -> int:
        custom_counter = getattr(self.vectorstore, "get_indexed_document_count", None)
        if callable(custom_counter):
            return int(custom_counter(collection_uuid))

        return int(
            session.scalar(
                select(func.count())
                .select_from(self.vectorstore.EmbeddingStore)
                .where(self.vectorstore.EmbeddingStore.collection_id == collection_uuid)
            )
            or 0
        )

    def _descriptor_from_metadata(self, metadata: dict[str, object]) -> EmbeddingDescriptor | None:
        provider = metadata.get("embedding_provider")
        model = metadata.get("embedding_model")
        dimension = metadata.get("embedding_dimension")
        if not isinstance(provider, str) or not isinstance(model, str):
            return None
        if not isinstance(dimension, int):
            return None

        return EmbeddingDescriptor(
            provider=provider,
            model=model,
            dimension=dimension,
        )
