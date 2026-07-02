from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.knowledge.ingestion import KnowledgeIngestionService, prepare_knowledge_ingestion_storage
from app.repositories.db.session import get_engine, get_session_factory
from app.services.retrieval import RetrievalService


def main() -> None:
    settings = get_settings()
    engine = get_engine()
    prepare_knowledge_ingestion_storage(engine)
    session_factory = get_session_factory()
    retrieval_service = RetrievalService(settings=settings)
    ingestion_service = KnowledgeIngestionService(
        retrieval_service=retrieval_service,
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )

    with session_factory() as session:
        result = ingestion_service.run(session)

    print(
        "Using chunk config: "
        f"chunk_size={settings.knowledge_chunk_size}, "
        f"chunk_overlap={settings.knowledge_chunk_overlap}"
    )
    print(f"Loaded {result.documents_loaded} source documents")
    for document_result in result.results:
        print(f"Ingested {document_result.source}: {document_result.chunk_count} chunks")
    print("Knowledge ingestion complete")


if __name__ == "__main__":
    main()
