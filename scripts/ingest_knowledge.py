from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.knowledge.ingestion import ingest_knowledge
from app.config import get_settings
from app.repositories.db.base import Base
from app.repositories.db.session import get_engine, get_session_factory
from app.services.retrieval import RetrievalService


def main() -> None:
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory()
    settings = get_settings()
    retrieval_service = RetrievalService(settings=settings)

    with session_factory() as session:
        documents, results = ingest_knowledge(session, retrieval_service)

    print(f"Loaded {len(documents)} source documents")
    for result in results:
        print(f"Ingested {result.source}: {result.chunk_count} chunks")
    print("Knowledge ingestion complete")


if __name__ == "__main__":
    main()
