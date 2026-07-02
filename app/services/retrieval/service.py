from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import func, select, text

from app.config import Settings
from app.infrastructure.embeddings import EmbeddingDescriptor, create_embedding_provider
from app.repositories.models import KnowledgeChunk
from app.services.retrieval.errors import VectorIndexConfigurationError

VECTOR_TYPE_PATTERN = re.compile(r"vector\((\d+)\)")


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
        self._embedding_descriptor = EmbeddingDescriptor(
            provider=settings.embedding_provider,
            model=settings.knowledge_embedding_model,
            dimension=settings.embedding_dimension,
        )

        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            from langchain_postgres import PGVector

            embedding_provider = create_embedding_provider(settings)
            self.vectorstore = PGVector(
                embeddings=embedding_provider,
                collection_name=settings.knowledge_collection_name,
                connection=settings.database_url,
                embedding_length=settings.embedding_dimension,
                collection_metadata=self.embedding_metadata,
                use_jsonb=True,
            )

        self._ensure_vector_dimension_is_supported()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self._default_top_k}
        )

    @property
    def embedding_descriptor(self) -> EmbeddingDescriptor:
        return self._embedding_descriptor

    @property
    def embedding_metadata(self) -> dict[str, object]:
        return self._embedding_descriptor.as_metadata()

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
        normalized_query = query.strip()
        if not normalized_query:
            return []

        self._ensure_index_compatible()
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
