from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import sqltypes

from app.domain.tracing.enums import TRACE_STEP_TYPE_VALUES, TRACE_STATUS_VALUES
from app.repositories.db.base import Base
from app.repositories.models.common import utcnow

if TYPE_CHECKING:
    from app.repositories.models.chat_trace import ChatTrace

JSON_VARIANT = sqltypes.JSON().with_variant(JSONB, "postgresql")
TRACE_STEP_TYPE_CHECK = (
    "step_type IN ('request_received', 'retrieval_started', 'retrieval_completed', "
    "'prompt_built', 'llm_call_started', 'llm_call_completed', 'response_generated', 'error')"
)
TRACE_STATUS_CHECK = "status IN ('started', 'success', 'error', 'cancelled')"


class ChatTraceStep(Base):
    __tablename__ = "chat_trace_steps"
    __table_args__ = (
        CheckConstraint(TRACE_STEP_TYPE_CHECK, name="ck_chat_trace_steps_type_valid"),
        CheckConstraint(TRACE_STATUS_CHECK, name="ck_chat_trace_steps_status_valid"),
        UniqueConstraint("trace_id", "step_index", name="uq_chat_trace_steps_trace_step_index"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    trace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_traces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        default=TRACE_STEP_TYPE_VALUES[0],
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=TRACE_STATUS_VALUES[1])
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_payload: Mapped[dict[str, object] | None] = mapped_column(JSON_VARIANT, nullable=True)
    output_payload: Mapped[dict[str, object] | None] = mapped_column(JSON_VARIANT, nullable=True)
    step_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSON_VARIANT,
        nullable=False,
        default=dict,
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )

    trace: Mapped["ChatTrace"] = relationship(back_populates="steps")
