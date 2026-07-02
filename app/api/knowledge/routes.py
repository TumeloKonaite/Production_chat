from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.knowledge_dependencies import get_knowledge_ingestion_service_factory
from app.api.knowledge.schemas import (
    KnowledgeIngestionDocumentResponse,
    KnowledgeIngestionResponse,
)
from app.config import Settings
from app.knowledge.ingestion import KnowledgeIngestionService
from collections.abc import Callable

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestionResponse)
def ingest_knowledge_endpoint(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
    ingestion_service_factory: Callable[[], KnowledgeIngestionService] = Depends(
        get_knowledge_ingestion_service_factory
    ),
    ingestion_secret: str | None = Header(default=None, alias="x-ingestion-secret"),
) -> KnowledgeIngestionResponse:
    _validate_ingestion_secret(
        provided_secret=ingestion_secret,
        expected_secret=settings.ingestion_api_secret,
    )
    ingestion_service = ingestion_service_factory()
    result = ingestion_service.run(session)
    return KnowledgeIngestionResponse(
        status=result.status,
        documents_loaded=result.documents_loaded,
        results=[
            KnowledgeIngestionDocumentResponse(
                source=document_result.source,
                chunk_count=document_result.chunk_count,
            )
            for document_result in result.results
        ],
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
