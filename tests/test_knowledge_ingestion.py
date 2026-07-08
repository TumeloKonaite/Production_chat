from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.knowledge.schemas import KnowledgeIngestionRequest
from app.infrastructure.storage import StorageError, SupabaseKnowledgeFileStorage
from app.knowledge.ingestion import (
    KnowledgeIngestionConflictError,
    KnowledgeIngestionServiceError,
    KnowledgeIngestionValidationError,
    KnowledgeIngestionService,
    UploadedKnowledgeFileLoader,
    chunk_markdown_document,
    clean_markdown_text,
    ingest_knowledge,
    load_source_documents,
)
from app.knowledge.ingestion.loader import SourceDocument
from app.repositories.db.base import Base
from app.repositories.models import KnowledgeChunk, KnowledgeFile
from app.repositories.knowledge_repository import KnowledgeRepository


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_session_factory(tmp_path) -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def write_source_file(base_dir: Path, name: str, content: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / name).write_text(content, encoding="utf-8")


class FakeRetrievalService:
    def __init__(self) -> None:
        self.replaced_chunk_ids: list[str] = []

    def replace_all_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self.replaced_chunk_ids = [chunk.id for chunk in chunks]


class MappingStorage:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def upload_file(
        self,
        *,
        file_bytes: bytes,
        storage_path: str,
        content_type: str | None = None,
    ):
        raise AssertionError("upload_file should not be called during ingestion tests")

    def download_file(self, *, storage_path: str) -> bytes:
        if storage_path not in self._files:
            raise StorageError("missing storage object") from FileNotFoundError(storage_path)
        return self._files[storage_path]

    def delete_file(self, *, storage_path: str) -> None:
        self._files.pop(storage_path, None)


def create_uploaded_knowledge_file(
    session: SASession,
    *,
    filename: str,
    storage_path: str | None = None,
    status: str = "uploaded",
    storage_provider: str = "local",
    storage_bucket: str = "knowledge-files",
) -> KnowledgeFile:
    knowledge_file = KnowledgeFile(
        id=str(uuid4()),
        original_filename=filename,
        content_type="text/markdown" if filename.endswith(".md") else "text/plain",
        file_size_bytes=128,
        storage_provider=storage_provider,
        storage_bucket=storage_bucket,
        storage_path=storage_path or f"uploads/{uuid4()}/{filename}",
        checksum="checksum",
        status=status,
    )
    session.add(knowledge_file)
    session.commit()
    session.refresh(knowledge_file)
    return knowledge_file


def test_loader_reads_markdown_documents(tmp_path) -> None:
    source_dir = tmp_path / "source"
    write_source_file(source_dir, "profile.md", "# Profile\n\nTumelo builds AI products.\n")
    write_source_file(source_dir, "skills.md", "# Skills\n\nFastAPI and SQLAlchemy.\n")

    documents = load_source_documents(source_dir)

    assert [document.source for document in documents] == ["profile.md", "skills.md"]
    assert documents[0].text.startswith("# Profile")
    assert documents[0].updated_at.tzinfo is not None


def test_cleaner_normalizes_whitespace_without_removing_headings() -> None:
    raw_text = "  #   Profile  \n\nTumelo    builds   AI systems. \n\n\n##  Focus \n  backend APIs\tand data products  "

    cleaned_text = clean_markdown_text(raw_text)

    assert cleaned_text == (
        "# Profile\n\n"
        "Tumelo builds AI systems.\n\n"
        "## Focus\n\n"
        "backend APIs and data products"
    )


def test_chunker_splits_by_heading_and_keeps_metadata() -> None:
    document = SourceDocument(
        source="projects.md",
        text=clean_markdown_text(
            "# Projects\n\n"
            "Tumelo builds grounded assistants.\n\n"
            "## Portfolio Chatbot\n\n"
            + " ".join(["retrieval"] * 820)
        ),
        updated_at=utcnow(),
    )

    chunks = chunk_markdown_document(document, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 2
    assert chunks[0].section == "Projects"
    assert chunks[0].source_type == "markdown"
    assert chunks[0].metadata["source"] == "projects.md"
    assert chunks[0].metadata["content_type"] == "projects"
    assert chunks[1].content.startswith("## Portfolio Chatbot")


def test_repository_replaces_existing_chunks_for_source(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        repository = KnowledgeRepository(session)
        initial_chunks = [
            SourceDocument(
                source="profile.md",
                text="# Profile\n\nTumelo builds AI systems.",
                updated_at=utcnow(),
            )
        ]
        repository.replace_source_chunks(
            source="profile.md",
            chunks=chunk_markdown_document(initial_chunks[0]),
        )
        repository.replace_source_chunks(
            source="profile.md",
            chunks=chunk_markdown_document(
                SourceDocument(
                    source="profile.md",
                    text="# Profile\n\nTumelo builds backend APIs.",
                    updated_at=utcnow(),
                )
            ),
        )

        stored_chunks = repository.list_by_source("profile.md")

    assert len(stored_chunks) == 1
    assert stored_chunks[0].content.endswith("Tumelo builds backend APIs.")


def test_ingest_knowledge_loads_and_persists_all_documents(tmp_path) -> None:
    source_dir = tmp_path / "source"
    write_source_file(source_dir, "profile.md", "# Profile\n\nTumelo builds AI products.\n")
    write_source_file(source_dir, "contact.md", "# Contact\n\nReach out for AI and backend work.\n")
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        documents, results = ingest_knowledge(
            session,
            retrieval_service,
            source_dir=source_dir,
        )
        repository = KnowledgeRepository(session)
        stored_chunks = repository.list_all()

    assert len(documents) == 2
    assert [result.source for result in results] == ["contact.md", "profile.md"]
    assert len(stored_chunks) == 2
    assert all(isinstance(chunk, KnowledgeChunk) for chunk in stored_chunks)
    assert all(chunk.source_type == "markdown" for chunk in stored_chunks)
    assert len(retrieval_service.replaced_chunk_ids) == 2


def test_ingest_knowledge_uses_configured_chunking_values(tmp_path) -> None:
    source_dir = tmp_path / "source"
    write_source_file(
        source_dir,
        "projects.md",
        "# Projects\n\n## Retrieval\n\n" + " ".join(["retrieval"] * 900),
    )
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        _, fine_grained_results = ingest_knowledge(
            session,
            retrieval_service,
            source_dir=source_dir,
            chunk_size=300,
            chunk_overlap=50,
        )
        _, coarse_results = ingest_knowledge(
            session,
            retrieval_service,
            source_dir=source_dir,
            chunk_size=1000,
            chunk_overlap=200,
        )

    assert fine_grained_results[0].chunk_count > coarse_results[0].chunk_count


def test_knowledge_ingestion_service_returns_summary(tmp_path) -> None:
    source_dir = tmp_path / "source"
    write_source_file(source_dir, "profile.md", "# Profile\n\nTumelo builds AI products.\n")
    write_source_file(source_dir, "projects.md", "# Projects\n\nTumelo builds grounded assistants.\n")
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()
    ingestion_service = KnowledgeIngestionService(
        retrieval_service=retrieval_service,
        source_dir=source_dir,
    )

    with session_factory() as session:
        result = ingestion_service.run(session)

    assert result.status == "ok"
    assert result.source_type == "local_directory"
    assert result.file_id is None
    assert result.documents_loaded == 2
    assert result.chunks_created == 2
    assert result.chunks_updated == 0
    assert result.chunks_skipped == 0
    assert [(item.source, item.chunk_count) for item in result.results] == [
        ("profile.md", 1),
        ("projects.md", 1),
    ]


def test_knowledge_ingestion_service_ingests_uploaded_markdown_file(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_path="uploaded/company-profile.md",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage(
                    {"uploaded/company-profile.md": b"# Company\n\nGrounded answers.\n"}
                ),
                storage_provider="local",
            ),
        )

        result = ingestion_service.run(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
        )
        repository = KnowledgeRepository(session)
        stored_chunks = list(repository.list_all())
        refreshed_file = session.get(KnowledgeFile, knowledge_file.id)

    assert result.status == "ingested"
    assert result.source_type == "uploaded_file"
    assert result.file_id == knowledge_file.id
    assert result.documents_loaded == 1
    assert result.chunks_created == 1
    assert [(item.source, item.chunk_count) for item in result.results] == [
        ("company-profile.md", 1)
    ]
    assert len(stored_chunks) == 1
    assert stored_chunks[0].source == "uploaded/company-profile.md"
    assert stored_chunks[0].source_type == "uploaded_file"
    assert stored_chunks[0].chunk_metadata["file_id"] == knowledge_file.id
    assert stored_chunks[0].chunk_metadata["original_filename"] == "company-profile.md"
    assert stored_chunks[0].chunk_metadata["storage_provider"] == "local"
    assert stored_chunks[0].chunk_metadata["storage_bucket"] == "knowledge-files"
    assert stored_chunks[0].chunk_metadata["storage_path"] == "uploaded/company-profile.md"
    assert refreshed_file is not None
    assert refreshed_file.status == "ingested"
    assert refreshed_file.ingested_at is not None


def test_knowledge_ingestion_service_ingests_uploaded_text_file(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="notes.txt",
            storage_path="uploaded/notes.txt",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage({"uploaded/notes.txt": b"Operational notes"}),
                storage_provider="local",
            ),
        )

        result = ingestion_service.run(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
        )
        repository = KnowledgeRepository(session)
        stored_chunks = list(repository.list_all())

    assert result.status == "ingested"
    assert result.chunks_created == 1
    assert len(stored_chunks) == 1
    assert stored_chunks[0].chunk_metadata["content_type"] == "text"
    assert stored_chunks[0].section == "Document"


def test_knowledge_ingestion_service_rejects_already_ingesting_uploaded_file(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            status="ingesting",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage({}),
                storage_provider="local",
            ),
        )

        with pytest.raises(
            KnowledgeIngestionConflictError,
            match="already being ingested",
        ):
            ingestion_service.run(
                session,
                request=KnowledgeIngestionRequest(
                    source_type="uploaded_file",
                    file_id=UUID(knowledge_file.id),
                ),
            )


def test_knowledge_ingestion_service_allows_reingesting_previously_ingested_uploaded_file(
    tmp_path,
) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_path="uploaded/company-profile.md",
            status="ingested",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage(
                    {"uploaded/company-profile.md": b"# Company\n\nUpdated content.\n"}
                ),
                storage_provider="local",
            ),
        )

        result = ingestion_service.run(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
        )

    assert result.status == "ingested"
    assert result.chunks_created == 1


def test_knowledge_ingestion_service_marks_missing_storage_object_as_failed(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_path="uploaded/missing.md",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage({}),
                storage_provider="local",
            ),
        )

        with pytest.raises(KnowledgeIngestionServiceError, match="Storage object not found."):
            ingestion_service.run(
                session,
                request=KnowledgeIngestionRequest(
                    source_type="uploaded_file",
                    file_id=UUID(knowledge_file.id),
                ),
            )

        refreshed_file = session.get(KnowledgeFile, knowledge_file.id)

    assert refreshed_file is not None
    assert refreshed_file.status == "failed"
    assert refreshed_file.error_message == "Storage object not found."
    assert refreshed_file.ingested_at is None


def test_knowledge_ingestion_service_marks_decode_failure_as_failed(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_path="uploaded/invalid.md",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage({"uploaded/invalid.md": b"\xff\xfe\x00"}),
                storage_provider="local",
            ),
        )

        with pytest.raises(KnowledgeIngestionValidationError, match="valid UTF-8 text"):
            ingestion_service.run(
                session,
                request=KnowledgeIngestionRequest(
                    source_type="uploaded_file",
                    file_id=UUID(knowledge_file.id),
                ),
            )

        refreshed_file = session.get(KnowledgeFile, knowledge_file.id)

    assert refreshed_file is not None
    assert refreshed_file.status == "failed"
    assert refreshed_file.error_message == "Uploaded knowledge files must be valid UTF-8 text."


def test_knowledge_ingestion_service_downloads_uploaded_file_from_supabase_storage(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/storage/v1/object/knowledge-files/uploaded/company-profile.md"
        return httpx.Response(200, content=b"# Company\n\nGrounded answers.\n")

    storage = SupabaseKnowledgeFileStorage(
        url="https://project.supabase.co",
        service_role_key="service-role",
        bucket="knowledge-files",
        http_client=httpx.Client(
            base_url="https://project.supabase.co/storage/v1",
            headers={
                "apiKey": "service-role",
                "Authorization": "Bearer service-role",
            },
            transport=httpx.MockTransport(handler),
        ),
    )

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_path="uploaded/company-profile.md",
            storage_provider="supabase",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=storage,
                storage_provider="supabase",
            ),
        )

        result = ingestion_service.run(
            session,
            request=KnowledgeIngestionRequest(
                source_type="uploaded_file",
                file_id=UUID(knowledge_file.id),
            ),
        )
        repository = KnowledgeRepository(session)
        stored_chunks = list(repository.list_all())

    assert result.status == "ingested"
    assert len(stored_chunks) == 1
    assert stored_chunks[0].chunk_metadata["storage_provider"] == "supabase"
    assert stored_chunks[0].chunk_metadata["storage_bucket"] == "knowledge-files"


def test_knowledge_ingestion_service_rejects_storage_provider_mismatch(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)
    retrieval_service = FakeRetrievalService()

    with session_factory() as session:
        knowledge_file = create_uploaded_knowledge_file(
            session,
            filename="company-profile.md",
            storage_provider="supabase",
        )
        ingestion_service = KnowledgeIngestionService(
            retrieval_service=retrieval_service,
            uploaded_file_loader=UploadedKnowledgeFileLoader(
                storage=MappingStorage({}),
                storage_provider="minio",
            ),
        )

        with pytest.raises(
            KnowledgeIngestionServiceError,
            match="storage_provider=supabase, but current STORAGE_PROVIDER=minio",
        ):
            ingestion_service.run(
                session,
                request=KnowledgeIngestionRequest(
                    source_type="uploaded_file",
                    file_id=UUID(knowledge_file.id),
                ),
            )

        refreshed_file = session.get(KnowledgeFile, knowledge_file.id)

    assert refreshed_file is not None
    assert refreshed_file.status == "failed"
    assert (
        refreshed_file.error_message
        == "File was uploaded with storage_provider=supabase, but current STORAGE_PROVIDER=minio."
    )
