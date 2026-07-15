from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.repositories.db.base import Base
from app.repositories.models.common import utcnow

if TYPE_CHECKING:
    from app.repositories.models.conversation import Conversation

MESSAGE_ROLE_CHECK = "role IN ('user', 'assistant', 'system')"


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(MESSAGE_ROLE_CHECK, name="ck_messages_role_valid"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False, default="web_chat")
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_config_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retrieval_config: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    message_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
