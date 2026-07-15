from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import CheckConstraint, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import sqltypes

from app.domain.tracing.enums import TRACE_STATUS_VALUES
from app.repositories.db.base import Base
from app.repositories.models.common import utcnow

if TYPE_CHECKING:
    from app.repositories.models.chat_trace_step import ChatTraceStep

TRACE_STATUS_CHECK = "status IN ('started', 'success', 'error', 'cancelled')"
JSON_VARIANT = sqltypes.JSON().with_variant(JSONB, "postgresql")


class ChatTrace(Base):
    __tablename__ = "chat_traces"
    __table_args__ = (
        CheckConstraint(TRACE_STATUS_CHECK, name="ck_chat_traces_status_valid"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        default=TRACE_STATUS_VALUES[0],
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    observability_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    external_trace_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retriever_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSON_VARIANT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
    )

    steps: Mapped[list["ChatTraceStep"]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="ChatTraceStep.step_index",
    )
