from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import KnowledgeIngestionJob


class KnowledgeIngestionJobRepositoryError(Exception):
    """Raised when knowledge-ingestion-job persistence fails."""


class KnowledgeIngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, job: KnowledgeIngestionJob) -> KnowledgeIngestionJob:
        return self.save(job)

    def save(self, job: KnowledgeIngestionJob) -> KnowledgeIngestionJob:
        try:
            self._session.add(job)
            self._session.commit()
            self._session.refresh(job)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeIngestionJobRepositoryError() from exc

        return job

    def get_by_id(self, job_id: str) -> KnowledgeIngestionJob | None:
        try:
            statement = select(KnowledgeIngestionJob).where(KnowledgeIngestionJob.id == job_id)
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise KnowledgeIngestionJobRepositoryError() from exc

    def find_latest_active_by_idempotency_key(self, idempotency_key: str) -> KnowledgeIngestionJob | None:
        return self._find_latest_by_statuses(idempotency_key, ("pending", "running"))

    def find_latest_terminal_by_idempotency_key(self, idempotency_key: str) -> KnowledgeIngestionJob | None:
        return self._find_latest_by_statuses(idempotency_key, ("completed", "skipped"))

    def mark_running(self, *, job_id: str, started_at: datetime) -> bool:
        return self._update_status(
            job_id=job_id,
            current_statuses=("pending", "failed"),
            next_status="running",
            started_at=started_at,
            completed_at=None,
            chunk_count=None,
            error_message=None,
        )

    def mark_completed(
        self,
        *,
        job_id: str,
        completed_at: datetime,
        chunk_count: int,
    ) -> bool:
        return self._update_status(
            job_id=job_id,
            current_statuses=("running",),
            next_status="completed",
            completed_at=completed_at,
            chunk_count=chunk_count,
            error_message=None,
        )

    def mark_failed(
        self,
        *,
        job_id: str,
        completed_at: datetime,
        error_message: str,
    ) -> bool:
        return self._update_status(
            job_id=job_id,
            current_statuses=("pending", "running", "failed"),
            next_status="failed",
            completed_at=completed_at,
            error_message=error_message,
        )

    def _find_latest_by_statuses(
        self,
        idempotency_key: str,
        statuses: tuple[str, ...],
    ) -> KnowledgeIngestionJob | None:
        try:
            statement = (
                select(KnowledgeIngestionJob)
                .where(KnowledgeIngestionJob.idempotency_key == idempotency_key)
                .where(KnowledgeIngestionJob.status.in_(statuses))
                .order_by(
                    KnowledgeIngestionJob.created_at.desc(),
                    KnowledgeIngestionJob.id.desc(),
                )
            )
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise KnowledgeIngestionJobRepositoryError() from exc

    def _update_status(
        self,
        *,
        job_id: str,
        current_statuses: tuple[str, ...],
        next_status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        values: dict[str, object | None] = {
            "status": next_status,
            "error_message": error_message,
        }
        if started_at is not None:
            values["started_at"] = started_at
        if completed_at is not None:
            values["completed_at"] = completed_at
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if next_status == "running":
            values["completed_at"] = None
            values["chunk_count"] = None

        try:
            statement = (
                update(KnowledgeIngestionJob)
                .where(KnowledgeIngestionJob.id == job_id)
                .where(KnowledgeIngestionJob.status.in_(current_statuses))
                .values(**values)
            )
            result = self._session.execute(statement)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeIngestionJobRepositoryError() from exc

        return result.rowcount > 0
