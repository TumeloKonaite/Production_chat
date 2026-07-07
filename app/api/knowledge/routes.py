from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
import secrets

from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import (
    get_knowledge_file_upload_service,
    get_knowledge_ingestion_service_factory,
)
from app.api.knowledge.schemas import (
    KnowledgeFileUploadResponse,
    KnowledgeIngestionDocumentResponse,
    KnowledgeIngestionRequest,
    KnowledgeIngestionResponse,
)
from app.config import Settings
from app.infrastructure.tracking import TrackingSetupError, create_experiment_tracker
from app.knowledge.ingestion import KnowledgeIngestionService
from app.services.knowledge_files import KnowledgeFileUploadService

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
        result = ingestion_service.run(session, request=payload)

    return KnowledgeIngestionResponse(
        status=result.status,
        source_type=result.source_type,
        file_id=result.file_id,
        experiment_name=payload.experiment_name,
        embedding_provider=effective_settings.embedding_provider,
        embedding_model=effective_settings.knowledge_embedding_model,
        embedding_dimension=effective_settings.embedding_dimension,
        documents_loaded=result.documents_loaded,
        chunks_created=result.chunks_created,
        chunks_updated=result.chunks_updated,
        chunks_skipped=result.chunks_skipped,
        results=[
            KnowledgeIngestionDocumentResponse(
                source=document_result.source,
                chunk_count=document_result.chunk_count,
            )
            for document_result in result.results
        ],
    )


@router.post(
    "/files",
    response_model=KnowledgeFileUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_knowledge_file_endpoint(
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
    upload_service: KnowledgeFileUploadService = Depends(get_knowledge_file_upload_service),
) -> KnowledgeFileUploadResponse:
    record = upload_service.upload_file(
        session,
        filename=file.filename,
        content_type=file.content_type,
        file_bytes=await file.read(),
    )
    return KnowledgeFileUploadResponse.model_validate(record)


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
