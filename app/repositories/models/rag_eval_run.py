from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.repositories.db.base import Base
from app.repositories.models.common import utcnow


class RagEvalRun(Base):
    __tablename__ = "rag_eval_runs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieval_config: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    retrieval_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_precision_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    avg_recall_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    avg_mrr: Mapped[float] = mapped_column(Float, nullable=False)
    avg_ndcg_at_k: Mapped[float] = mapped_column(Float, nullable=False)
    avg_context_relevance: Mapped[float] = mapped_column(Float, nullable=False)
    avg_faithfulness: Mapped[float] = mapped_column(Float, nullable=False)
    avg_answer_relevance: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
