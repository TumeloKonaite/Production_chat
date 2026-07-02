from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.ingestion import (
    KnowledgeIngestionService,
    chunk_markdown_document,
    clean_markdown_text,
    ingest_knowledge,
    load_source_documents,
)
from app.knowledge.ingestion.loader import SourceDocument
from app.repositories.db.base import Base
from app.repositories.models import KnowledgeChunk
from app.repositories.knowledge_repository import KnowledgeRepository


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_session_factory(tmp_path) -> sessionmaker[Session]:
    database_path = tmp_path / "test_knowledge.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
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
    assert result.documents_loaded == 2
    assert [(item.source, item.chunk_count) for item in result.results] == [
        ("profile.md", 1),
        ("projects.md", 1),
    ]
