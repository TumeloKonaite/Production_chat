from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
import pytest

from app.repositories.db.base import Base
from app.repositories.models import ChatTrace, ChatTraceStep, Conversation, KnowledgeChunk, Message, RetrievalLog
from evals.feedback.feedback_dataset import FEEDBACK_DATASET_SOURCE
from evals.feedback.feedback_exporter import (
    ProductionFeedbackExportDisabledError,
    ProductionFeedbackExportFilters,
    export_feedback_dataset,
    list_feedback_examples,
)


def build_session_factory(tmp_path: Path) -> sessionmaker[Session]:
    database_path = tmp_path / "feedback_exporter.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def test_feedback_exporter_redacts_text_by_default(tmp_path: Path) -> None:
    session_factory = build_session_factory(tmp_path)
    output_path = tmp_path / "feedback.jsonl"

    with session_factory() as session:
        _seed_feedback_trace(session)
        result = export_feedback_dataset(
            session=session,
            settings=type(
                "Settings",
                (),
                {
                    "enable_production_feedback_export": True,
                    "allow_raw_production_text_in_evals": False,
                },
            )(),
            filters=ProductionFeedbackExportFilters(rating="negative", limit=10),
            output_path=output_path,
            allow_raw_text=False,
            append=False,
            overwrite=True,
        )

    assert result.written_rows == 1
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == FEEDBACK_DATASET_SOURCE
    assert row["question"] == "[redacted production question]"
    assert row["actual_answer"] == "[redacted production answer]"
    assert row["feedback_comment"] == "[redacted production feedback comment]"
    assert row["context"][0]["content"] == "[redacted production context]"
    assert row["expected_facts"] == ["backend APIs"]
    assert row["expected_source_documents"] == ["profile.md"]


def test_feedback_exporter_filters_by_reason_and_date(tmp_path: Path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        _seed_feedback_trace(
            session,
            trace_id="trace-negative",
            created_at=datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
            feedback_rating="negative",
            feedback_reason="incorrect_answer",
        )
        _seed_feedback_trace(
            session,
            trace_id="trace-positive",
            created_at=datetime(2026, 7, 1, 9, 0, tzinfo=UTC),
            feedback_rating="positive",
            feedback_reason="helpful_answer",
        )

        examples = list_feedback_examples(
            session=session,
            filters=ProductionFeedbackExportFilters(
                from_timestamp=datetime(2026, 7, 5, 0, 0, tzinfo=UTC),
                rating="negative",
                reason="incorrect_answer",
                limit=10,
            ),
            allow_raw_text=True,
        )

    assert [example.trace_id for example in examples] == ["trace-negative"]
    assert examples[0].question == "What does Tumelo do?"
    assert examples[0].actual_answer == "Tumelo is mainly a frontend developer."


def test_feedback_exporter_requires_explicit_enablement(tmp_path: Path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        _seed_feedback_trace(session)
        with pytest.raises(
            ProductionFeedbackExportDisabledError,
            match="ENABLE_PRODUCTION_FEEDBACK_EXPORT=true",
        ):
            export_feedback_dataset(
                session=session,
                settings=type(
                    "Settings",
                    (),
                    {
                        "enable_production_feedback_export": False,
                        "allow_raw_production_text_in_evals": False,
                    },
                )(),
                filters=ProductionFeedbackExportFilters(limit=10),
                output_path=tmp_path / "feedback.jsonl",
                allow_raw_text=False,
                append=False,
                overwrite=True,
            )


def test_feedback_exporter_blocks_raw_text_without_config_flag(tmp_path: Path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        _seed_feedback_trace(session)
        with pytest.raises(
            ProductionFeedbackExportDisabledError,
            match="ALLOW_RAW_PRODUCTION_TEXT_IN_EVALS=true",
        ):
            export_feedback_dataset(
                session=session,
                settings=type(
                    "Settings",
                    (),
                    {
                        "enable_production_feedback_export": True,
                        "allow_raw_production_text_in_evals": False,
                    },
                )(),
                filters=ProductionFeedbackExportFilters(limit=10),
                output_path=tmp_path / "feedback.jsonl",
                allow_raw_text=True,
                append=False,
                overwrite=True,
            )


def _seed_feedback_trace(
    session: Session,
    *,
    trace_id: str | None = None,
    created_at: datetime | None = None,
    feedback_rating: str = "negative",
    feedback_reason: str = "incorrect_answer",
) -> None:
    conversation = Conversation(
        id=str(uuid.uuid4()),
        model="openai:gpt-4.1-mini",
        prompt_version="v1_professional",
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
        updated_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    knowledge_chunk = KnowledgeChunk(
        id=str(uuid.uuid4()),
        source="profile.md",
        source_type="markdown",
        section="Summary",
        content="Tumelo builds practical AI systems and backend APIs.",
        chunk_metadata={},
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
        updated_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    user_message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        role="user",
        content="What does Tumelo do?",
        channel="web_chat",
        message_metadata={},
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    assistant_message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        role="assistant",
        content="Tumelo is mainly a frontend developer.",
        channel="web_chat",
        model="gpt-4.1-mini",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        model_config_id="openai:gpt-4.1-mini",
        prompt_version="v1_professional",
        retrieval_config="default",
        message_metadata={},
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    retrieval_log = RetrievalLog(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        message_id=user_message.id,
        query="What does Tumelo do?",
        top_k=5,
        retrieved_chunk_ids=[knowledge_chunk.id],
        retrieved_sources=["profile.md"],
        similarity_scores=[0.92],
        used_fallback=False,
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    trace = ChatTrace(
        id=trace_id or str(uuid.uuid4()),
        conversation_id=conversation.id,
        request_id="request-123",
        session_id="session-123",
        input_text="What does Tumelo do?",
        output_text="Tumelo is mainly a frontend developer.",
        status="success",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        prompt_version="v1_professional",
        retriever_type="vector",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        trace_metadata={
            "environment": "production",
            "route": "/chat",
            "feedback": {
                "rating": feedback_rating,
                "reason": feedback_reason,
                "comment": "The answer missed the backend work.",
                "expected_facts": ["backend APIs"],
                "expected_source_documents": ["profile.md"],
                "langfuse_trace_id": "lf-trace-123",
            },
        },
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
        updated_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )
    trace_step = ChatTraceStep(
        id=str(uuid.uuid4()),
        trace_id=trace.id,
        step_index=1,
        step_type="retrieval_completed",
        status="success",
        output_payload={
            "retrieved_chunks": [
                {
                    "source": "profile.md",
                    "section": "Summary",
                    "score": 0.92,
                }
            ]
        },
        step_metadata={},
        created_at=created_at or datetime(2026, 7, 6, 20, 0, tzinfo=UTC),
    )

    session.add_all(
        [
            conversation,
            knowledge_chunk,
            user_message,
            assistant_message,
            retrieval_log,
            trace,
            trace_step,
        ]
    )
    session.commit()
