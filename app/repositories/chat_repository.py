from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import Conversation, Message, utcnow


class ConversationRepositoryError(Exception):
    """Raised when chat persistence fails."""


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_conversation(
        self,
        *,
        model: str,
        prompt_version: str,
        visitor_id: str | None = None,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            visitor_id=visitor_id,
            title=title,
            model=model,
            prompt_version=prompt_version,
        )
        self._session.add(conversation)
        return self._commit_and_refresh(conversation)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        statement = select(Conversation).where(Conversation.id == conversation_id)
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError() from exc

    def add_message(
        self,
        *,
        conversation: Conversation,
        role: str,
        content: str,
        model: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        model_config_id: str | None = None,
        prompt_version: str | None = None,
        retrieval_config: str | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> Message:
        conversation.updated_at = utcnow()
        message = Message(
            conversation=conversation,
            role=role,
            content=content,
            model=model,
            model_provider=model_provider,
            model_name=model_name,
            model_config_id=model_config_id,
            prompt_version=prompt_version,
            retrieval_config=retrieval_config,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self._session.add(message)
        return self._commit_and_refresh(message)

    def list_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> Sequence[Message]:
        descending_statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        try:
            recent_messages = list(self._session.scalars(descending_statement))
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError() from exc
        # Query in descending order to apply the limit efficiently, then restore chat order for the LLM.
        recent_messages.reverse()
        return recent_messages

    def _commit_and_refresh(self, instance: Conversation | Message) -> Conversation | Message:
        try:
            self._session.commit()
            self._session.refresh(instance)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise ConversationRepositoryError() from exc

        return instance
