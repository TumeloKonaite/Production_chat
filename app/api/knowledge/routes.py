from __future__ import annotations

from dataclasses import replace
import secrets

from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import (
    get_knowledge_file_upload_service,
    get_knowledge_ingestion_orchestrator,
)
from app.api.knowledge.schemas import (
    KnowledgeFileUploadResponse,
    KnowledgeIngestionRequest,
    KnowledgeIngestionResponse,
)
from app.config import Settings
from app.knowledge.ingestion import KnowledgeIngestionOrchestrator
from app.services.knowledge_files import KnowledgeFileUploadService

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestionResponse)
def ingest_knowledge_endpoint(
    response: Response,
    request: KnowledgeIngestionRequest | None = Body(default=None),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
    orchestrator: KnowledgeIngestionOrchestrator = Depends(
        get_knowledge_ingestion_orchestrator
    ),
    ingestion_secret: str | None = Header(default=None, alias="x-ingestion-secret"),
) -> KnowledgeIngestionResponse:
    _validate_ingestion_secret(
        provided_secret=ingestion_secret,
        expected_secret=settings.ingestion_api_secret,
    )
    payload = request or KnowledgeIngestionRequest()
    effective_settings = _resolve_effective_settings(settings=settings, request=payload)
    result = orchestrator.trigger(
        session,
        request=payload,
        effective_settings=effective_settings,
    )
    response.status_code = (
        status.HTTP_200_OK if result.status == "skipped" else status.HTTP_202_ACCEPTED
    )
    return KnowledgeIngestionResponse(
        job_id=result.job_id,
        status=result.status,
        source_type=result.source_type,
        file_id=result.file_id,
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
