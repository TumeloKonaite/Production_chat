from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import KnowledgeFile


class KnowledgeFileRepositoryError(Exception):
    """Raised when knowledge-file persistence fails."""


class KnowledgeFileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, file_id: str) -> KnowledgeFile | None:
        try:
            statement = select(KnowledgeFile).where(KnowledgeFile.id == file_id)
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise KnowledgeFileRepositoryError() from exc

    def create(self, knowledge_file: KnowledgeFile) -> KnowledgeFile:
        return self.save(knowledge_file)

    def save(self, knowledge_file: KnowledgeFile) -> KnowledgeFile:
        try:
            self._session.add(knowledge_file)
            self._session.commit()
            self._session.refresh(knowledge_file)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeFileRepositoryError() from exc

        return knowledge_file
