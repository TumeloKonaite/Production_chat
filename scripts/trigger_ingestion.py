from __future__ import annotations

import argparse
from pathlib import Path
import sys
from uuid import UUID

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.dependencies.knowledge_dependencies import get_knowledge_ingestion_orchestrator
from app.api.knowledge.schemas import KnowledgeIngestionRequest
from app.config import Settings, get_settings
from app.repositories.db.session import get_session_factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trigger a knowledge ingestion job.")
    parser.add_argument(
        "--source-type",
        choices=("local_directory", "uploaded_file"),
        default="uploaded_file",
        help="Knowledge source type to ingest.",
    )
    parser.add_argument(
        "--source-id",
        "--file-id",
        dest="file_id",
        help="Uploaded file ID when source_type=uploaded_file.",
    )
    parser.add_argument("--experiment-name", help="Optional experiment label for this trigger.")
    parser.add_argument(
        "--embedding-provider",
        choices=("hf", "openai", "openrouter"),
        help="Optional embedding provider override. Requires model, dimension, and reset flag.",
    )
    parser.add_argument("--embedding-model", help="Optional embedding model override.")
    parser.add_argument("--embedding-dimension", type=int, help="Optional embedding dimension override.")
    parser.add_argument(
        "--reset-existing-vectors",
        action="store_true",
        help="Required when overriding the embedding configuration.",
    )
    return parser


def build_request(args: argparse.Namespace) -> KnowledgeIngestionRequest:
    file_id = UUID(args.file_id) if args.file_id else None
    return KnowledgeIngestionRequest(
        source_type=args.source_type,
        file_id=file_id,
        experiment_name=args.experiment_name,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        reset_existing_vectors=args.reset_existing_vectors or None,
    )


def build_effective_settings(
    *,
    settings: Settings,
    request: KnowledgeIngestionRequest,
) -> Settings:
    if not request.has_embedding_override:
        return settings

    from dataclasses import replace

    return replace(
        settings,
        embedding_provider=str(request.embedding_provider),
        knowledge_embedding_model=str(request.embedding_model),
        embedding_dimension=int(request.embedding_dimension),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()
    request = build_request(args)
    effective_settings = build_effective_settings(settings=settings, request=request)
    orchestrator = get_knowledge_ingestion_orchestrator(settings)
    session_factory = get_session_factory()

    with session_factory() as session:
        result = orchestrator.trigger(
            session,
            request=request,
            effective_settings=effective_settings,
        )

    print(f"job_id={result.job_id}")
    print(f"status={result.status}")
    print(f"source_type={result.source_type}")
    if result.file_id is not None:
        print(f"file_id={result.file_id}")


if __name__ == "__main__":
    main()
