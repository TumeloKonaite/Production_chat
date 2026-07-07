from __future__ import annotations

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import KnowledgeFile


class KnowledgeFileRepositoryError(Exception):
    """Raised when knowledge-file persistence fails."""


class KnowledgeFileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, knowledge_file: KnowledgeFile) -> KnowledgeFile:
        try:
            self._session.add(knowledge_file)
            self._session.commit()
            self._session.refresh(knowledge_file)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise KnowledgeFileRepositoryError() from exc

        return knowledge_file
