from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.domain.tracing import TraceStatus, TraceStepType
from app.repositories.db.base import Base
from app.services.tracing import TraceService


def build_session_factory(tmp_path) -> sessionmaker[Session]:
    database_path = tmp_path / "test_trace_service.db"
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


def test_trace_service_records_trace_lifecycle(tmp_path) -> None:
    service = TraceService(session_factory=build_session_factory(tmp_path))

    trace = service.start_trace(
        conversation_id=str(uuid.uuid4()),
        input_text="Tell me about Tumelo's chatbot",
        status=TraceStatus.STARTED,
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        prompt_version="v1_professional",
        retriever_type="vector",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        metadata={"route": "/chat"},
    )
    service.add_step(
        trace_id=trace.id,
        step_type=TraceStepType.REQUEST_RECEIVED,
        status=TraceStatus.SUCCESS,
        input_payload={"message": "Tell me about Tumelo's chatbot"},
        metadata={"channel": "web_chat"},
    )
    service.complete_trace(
        trace.id,
        output_text="Tumelo built a production-ready FastAPI chatbot.",
        status=TraceStatus.SUCCESS,
        input_tokens=1200,
        output_tokens=180,
        total_tokens=1380,
        estimated_cost_usd=0.000768,
        latency_ms=842,
        metadata={"route": "/chat", "channel": "web_chat"},
    )

    stored_trace = service.get_trace(trace.id)

    assert stored_trace is not None
    assert stored_trace.status == TraceStatus.SUCCESS
    assert stored_trace.output_text == "Tumelo built a production-ready FastAPI chatbot."
    assert stored_trace.total_tokens == 1380
    assert float(stored_trace.estimated_cost_usd) == 0.000768
    assert stored_trace.steps[0].step_type == TraceStepType.REQUEST_RECEIVED


def test_trace_service_sanitizes_non_json_payload_values(tmp_path) -> None:
    service = TraceService(session_factory=build_session_factory(tmp_path))
    trace = service.start_trace(input_text="hello", metadata={"created_by": uuid.uuid4()})
    started_at = datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc)

    service.add_step(
        trace_id=trace.id,
        step_type=TraceStepType.PROMPT_BUILT,
        status=TraceStatus.SUCCESS,
        input_payload={
            "conversation_uuid": uuid.uuid4(),
            "started_at": started_at,
            "labels": ("one", "two"),
        },
    )

    stored_trace = service.get_trace(trace.id)

    assert stored_trace is not None
    assert isinstance(stored_trace.metadata["created_by"], str)
    assert isinstance(stored_trace.steps[0].input_payload["conversation_uuid"], str)
    assert stored_trace.steps[0].input_payload["started_at"] == started_at.isoformat()
    assert stored_trace.steps[0].input_payload["labels"] == ["one", "two"]
