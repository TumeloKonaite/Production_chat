from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
import secrets

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import get_knowledge_ingestion_service_factory
from app.api.knowledge.schemas import (
    KnowledgeIngestionDocumentResponse,
    KnowledgeIngestionRequest,
    KnowledgeIngestionResponse,
)
from app.config import Settings
from app.infrastructure.tracking import TrackingSetupError, create_experiment_tracker
from app.knowledge.ingestion import KnowledgeIngestionService

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestionResponse)
def ingest_knowledge_endpoint(
    request: KnowledgeIngestionRequest | None = Body(default=None),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
    ingestion_service_factory: Callable[[Settings | None], KnowledgeIngestionService] = Depends(
        get_knowledge_ingestion_service_factory
    ),
    ingestion_secret: str | None = Header(default=None, alias="x-ingestion-secret"),
) -> KnowledgeIngestionResponse:
    _validate_ingestion_secret(
        provided_secret=ingestion_secret,
        expected_secret=settings.ingestion_api_secret,
    )
    payload = request or KnowledgeIngestionRequest()
    effective_settings = _resolve_effective_settings(settings=settings, request=payload)
    ingestion_service = ingestion_service_factory(effective_settings)
    with _ingestion_tracking_run(settings=effective_settings, request=payload):
        result = ingestion_service.run(session)

    chunks_created = sum(document_result.chunk_count for document_result in result.results)
    return KnowledgeIngestionResponse(
        status=result.status,
        experiment_name=payload.experiment_name,
        embedding_provider=effective_settings.embedding_provider,
        embedding_model=effective_settings.knowledge_embedding_model,
        embedding_dimension=effective_settings.embedding_dimension,
        documents_loaded=result.documents_loaded,
        chunks_created=chunks_created,
        results=[
            KnowledgeIngestionDocumentResponse(
                source=document_result.source,
                chunk_count=document_result.chunk_count,
            )
            for document_result in result.results
        ],
    )


def _resolve_effective_settings(
    *,
    settings: Settings,
    request: KnowledgeIngestionRequest,
) -> Settings:
    if not request.has_embedding_override:
        return settings

    return replace(
        settings,
        embedding_provider=str(request.embedding_provider),
        knowledge_embedding_model=str(request.embedding_model),
        embedding_dimension=int(request.embedding_dimension),
    )


@contextmanager
def _ingestion_tracking_run(
    *,
    settings: Settings,
    request: KnowledgeIngestionRequest,
):
    # Tracking is best-effort for ingestion so experiments do not fail due to
    # optional observability configuration.
    try:
        tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)
    except TrackingSetupError:
        yield
        return

    if not tracker.enabled:
        yield
        return

    run_name = request.experiment_name or f"knowledge-ingest-{secrets.token_hex(4)}"
    with tracker.run(run_name=run_name):
        tracker.log_params(
            {
                "experiment_name": request.experiment_name,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.knowledge_embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "reset_existing_vectors": request.reset_existing_vectors,
            }
        )
        yield


def _validate_ingestion_secret(
    *,
    provided_secret: str | None,
    expected_secret: str | None,
) -> None:
    if not expected_secret or not provided_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ingestion secret.",
        )
    if not secrets.compare_digest(provided_secret, expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid ingestion secret.",
        )
