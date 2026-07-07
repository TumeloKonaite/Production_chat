from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

from app.repositories.feedback_repository import (
    MessageFeedbackRepository,
    MessageFeedbackRepositoryError,
)
from app.repositories.models import ChatTrace, Message
from app.services.feedback.errors import (
    InvalidFeedbackTargetError,
    MessageFeedbackPersistenceError,
    MessageFeedbackTargetNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubmittedMessageFeedback:
    id: str
    message_id: str
    rating: str
    comment: str | None
    created_at: datetime
    updated_at: datetime


class MessageFeedbackService:
    def __init__(self, repository: MessageFeedbackRepository) -> None:
        self._repository = repository

    def submit_feedback(
        self,
        *,
        message_id: str,
        rating: str,
        comment: str | None,
    ) -> SubmittedMessageFeedback:
        try:
            message = self._repository.get_message(message_id)
        except MessageFeedbackRepositoryError as exc:
            raise MessageFeedbackPersistenceError() from exc

        if message is None:
            raise MessageFeedbackTargetNotFoundError("Message not found.")
        if message.role != "assistant":
            raise InvalidFeedbackTargetError(
                "Feedback can only be submitted for assistant messages."
            )

        normalized_comment = self._normalize_optional_string(comment)
        try:
            trace = self._repository.find_trace_for_message(message)
            feedback, created = self._repository.upsert_feedback(
                message=message,
                trace=trace,
                rating=rating,
                comment=normalized_comment,
                user_id=self._extract_optional_string(message.message_metadata, "user_id"),
                feedback_metadata=self._build_feedback_metadata(message=message, trace=trace),
            )
        except MessageFeedbackRepositoryError as exc:
            raise MessageFeedbackPersistenceError() from exc

        logger.info(
            "Message feedback submitted.",
            extra={
                "feedback_id": feedback.id,
                "message_id": message.id,
                "conversation_id": message.conversation_id,
                "trace_id": feedback.trace_id,
                "rating": rating,
                "created": created,
            },
        )
        return SubmittedMessageFeedback(
            id=feedback.id,
            message_id=feedback.message_id,
            rating=feedback.rating,
            comment=feedback.comment,
            created_at=feedback.created_at,
            updated_at=feedback.updated_at,
        )

    def _build_feedback_metadata(
        self,
        *,
        message: Message,
        trace: ChatTrace | None,
    ) -> dict[str, object]:
        metadata = {
            "channel": message.channel,
            "model": message.model,
            "model_provider": message.model_provider,
            "model_name": message.model_name,
            "model_config_id": message.model_config_id,
            "prompt_version": message.prompt_version,
            "retrieval_config": message.retrieval_config,
            "latency_ms": message.latency_ms,
            "input_tokens": message.input_tokens,
            "output_tokens": message.output_tokens,
            "total_tokens": message.total_tokens,
            "estimated_cost_usd": message.estimated_cost_usd,
            "session_id": (
                trace.session_id
                if trace is not None
                else self._extract_optional_string(message.message_metadata, "session_id")
            ),
            "request_id": (
                trace.request_id
                if trace is not None
                else self._extract_optional_string(message.message_metadata, "request_id")
            ),
            "trace_status": (trace.status if trace is not None else None),
            "route": (
                self._extract_optional_string(trace.trace_metadata, "route")
                if trace is not None
                else None
            ),
            "environment": (
                self._extract_optional_string(trace.trace_metadata, "environment")
                if trace is not None
                else None
            ),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _extract_optional_string(
        self,
        metadata: dict[str, object],
        key: str,
    ) -> str | None:
        return self._normalize_optional_string(metadata.get(key))

    def _normalize_optional_string(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None
