from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.repositories.db.base import Base
from app.repositories.models.common import utcnow


class RagEvalResult(Base):
    __tablename__ = "rag_eval_results"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rag_eval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_source_documents: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    retrieved_source_documents: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    precision_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    recall_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    mrr: Mapped[float] = mapped_column(Float, nullable=False)
    ndcg_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    context_relevance_score: Mapped[int] = mapped_column(Integer, nullable=False)
    context_relevance_reason: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness_score: Mapped[int] = mapped_column(Integer, nullable=False)
    faithfulness_reason: Mapped[str] = mapped_column(Text, nullable=False)
    answer_relevance_score: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_relevance_reason: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_usage: Mapped[dict[str, int | None]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
