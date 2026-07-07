from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.models import ChatTrace, Message, MessageFeedback, utcnow


class MessageFeedbackRepositoryError(Exception):
    """Raised when message feedback persistence fails."""


class MessageFeedbackRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_message(self, message_id: str) -> Message | None:
        statement = select(Message).where(Message.id == message_id)
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise MessageFeedbackRepositoryError() from exc

    def get_feedback_by_message_id(self, message_id: str) -> MessageFeedback | None:
        statement = select(MessageFeedback).where(MessageFeedback.message_id == message_id)
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise MessageFeedbackRepositoryError() from exc

    def get_trace(self, trace_id: str) -> ChatTrace | None:
        statement = select(ChatTrace).where(ChatTrace.id == trace_id)
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise MessageFeedbackRepositoryError() from exc

    def find_trace_for_message(self, message: Message) -> ChatTrace | None:
        trace_id = message.message_metadata.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            trace = self.get_trace(trace_id)
            if trace is not None:
                return trace

        statement = (
            select(ChatTrace)
            .where(
                ChatTrace.conversation_id == message.conversation_id,
                ChatTrace.output_text == message.content,
            )
            .order_by(ChatTrace.created_at.desc(), ChatTrace.id.desc())
            .limit(1)
        )
        try:
            return self._session.scalar(statement)
        except SQLAlchemyError as exc:
            raise MessageFeedbackRepositoryError() from exc

    def upsert_feedback(
        self,
        *,
        message: Message,
        trace: ChatTrace | None,
        rating: str,
        comment: str | None,
        user_id: str | None,
        feedback_metadata: dict[str, object],
    ) -> tuple[MessageFeedback, bool]:
        feedback = self.get_feedback_by_message_id(message.id)
        created = feedback is None

        if feedback is None:
            feedback = MessageFeedback(
                id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                message_id=message.id,
                trace_id=(trace.id if trace is not None else None),
                rating=rating,
                comment=comment,
                user_id=user_id,
                feedback_metadata=dict(feedback_metadata),
            )
            self._session.add(feedback)
        else:
            feedback.conversation_id = message.conversation_id
            feedback.trace_id = trace.id if trace is not None else None
            feedback.rating = rating
            feedback.comment = comment
            feedback.user_id = user_id
            feedback.feedback_metadata = dict(feedback_metadata)
            feedback.updated_at = utcnow()

        if trace is not None:
            trace.trace_metadata = self._build_trace_feedback_metadata(
                trace=trace,
                feedback=feedback,
                rating=rating,
                comment=comment,
            )
            trace.updated_at = utcnow()

        return self._commit_and_refresh(feedback), created

    def _build_trace_feedback_metadata(
        self,
        *,
        trace: ChatTrace,
        feedback: MessageFeedback,
        rating: str,
        comment: str | None,
    ) -> dict[str, object]:
        metadata = dict(trace.trace_metadata or {})
        metadata["feedback"] = {
            "rating": rating,
            "thumb_rating": rating,
            "feedback_rating": "positive" if rating == "up" else "negative",
            "comment": comment,
            "feedback_comment": comment,
            "message_feedback_id": feedback.id,
            "message_id": feedback.message_id,
            "conversation_id": feedback.conversation_id,
            "trace_id": trace.id,
        }
        return metadata

    def _commit_and_refresh(self, feedback: MessageFeedback) -> MessageFeedback:
        try:
            self._session.commit()
            self._session.refresh(feedback)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise MessageFeedbackRepositoryError() from exc
        return feedback
