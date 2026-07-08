from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.domain.tracing import ChatTraceCreate, ChatTraceStepCreate, ChatTraceUpdate, TraceStatus, TraceStepType
from app.repositories.db.base import Base
from app.repositories.tracing_repository import TraceRepository


def build_session_factory(tmp_path) -> sessionmaker[Session]:
    database_path = tmp_path / "test_trace_repository.db"
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


def test_repository_creates_and_updates_trace(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        repository = TraceRepository(session)
        trace = repository.create_trace(
            ChatTraceCreate(
                conversation_id="conv-123",
                input_text="What projects has Tumelo built?",
                status=TraceStatus.STARTED,
                llm_provider="openrouter",
                llm_model="openai/gpt-4o-mini",
                observability_provider="langfuse",
                external_trace_id="lf-trace-123",
                retriever_type="vector",
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                metadata={"route": "/chat", "channel": "web_chat"},
            )
        )
        updated = repository.update_trace(
            trace.id,
            ChatTraceUpdate(
                output_text="Tumelo has built several AI and backend projects.",
                status=TraceStatus.SUCCESS,
                input_tokens=1200,
                output_tokens=320,
                total_tokens=1520,
                latency_ms=1840,
                estimated_cost_usd=0.001234,
                metadata={"route": "/chat", "channel": "web_chat", "environment": "test"},
            ),
        )

    assert trace.id == updated.id
    assert updated.status == "success"
    assert updated.output_text == "Tumelo has built several AI and backend projects."
    assert updated.total_tokens == 1520
    assert float(updated.estimated_cost_usd) == 0.001234
    assert updated.observability_provider == "langfuse"
    assert updated.external_trace_id == "lf-trace-123"
    assert updated.trace_metadata["environment"] == "test"


def test_repository_attaches_steps_in_incrementing_order(tmp_path) -> None:
    session_factory = build_session_factory(tmp_path)

    with session_factory() as session:
        repository = TraceRepository(session)
        trace = repository.create_trace(
            ChatTraceCreate(
                input_text="Tell me about BeautyVerse",
                status=TraceStatus.STARTED,
            )
        )
        first_step = repository.create_step(
            ChatTraceStepCreate(
                trace_id=trace.id,
                step_type=TraceStepType.RETRIEVAL_STARTED,
                status=TraceStatus.STARTED,
                input_payload={"query": "Tell me about BeautyVerse", "top_k": 5},
            )
        )
        second_step = repository.create_step(
            ChatTraceStepCreate(
                trace_id=trace.id,
                step_type=TraceStepType.RETRIEVAL_COMPLETED,
                status=TraceStatus.SUCCESS,
                output_payload={"retrieved_chunks": [{"source": "projects.md", "score": 0.82}]},
                latency_ms=220,
            )
        )
        stored_steps = repository.list_steps(trace.id)

    assert first_step.step_index == 1
    assert second_step.step_index == 2
    assert [step.step_type for step in stored_steps] == [
        "retrieval_started",
        "retrieval_completed",
    ]
    assert stored_steps[1].output_payload == {
        "retrieved_chunks": [{"source": "projects.md", "score": 0.82}]
    }
