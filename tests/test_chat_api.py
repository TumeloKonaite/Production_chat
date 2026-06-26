from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.chat_dependencies import get_llm_service
from app.api.dependencies.common_dependencies import get_db_session
from app.main import app
from app.repositories.db.base import Base
from app.repositories.models import Conversation, Message
from app.services.llm import LLMChatMessage, LLMGeneratedResponse, LLMServiceError, TokenUsage


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class FakeLLMService:
    def __init__(
        self,
        *,
        reply: str = "Mocked assistant response.",
        fail: bool = False,
    ) -> None:
        self.reply = reply
        self.fail = fail
        self.model = "gpt-4.1-mini"
        self.prompt_version = "v1"
        self.calls: list[list[LLMChatMessage]] = []

    async def generate_response(
        self,
        messages: list[LLMChatMessage],
    ) -> LLMGeneratedResponse:
        self.calls.append(list(messages))
        if self.fail:
            raise LLMServiceError("sk-test-should-not-leak")

        return LLMGeneratedResponse(
            message=self.reply,
            model=self.model,
            prompt_version=self.prompt_version,
            latency_ms=842,
            token_usage=TokenUsage(
                input_tokens=1200,
                output_tokens=180,
                total_tokens=1380,
            ),
        )


def build_test_client(tmp_path, fake_llm: FakeLLMService) -> tuple[TestClient, sessionmaker[Session]]:
    database_path = tmp_path / "test_chatbot.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )

    def override_db_session() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_llm_service] = lambda: fake_llm
    return TestClient(app), session_factory


def fetch_messages(session_factory: sessionmaker[Session]) -> list[Message]:
    with session_factory() as session:
        statement = select(Message).order_by(Message.created_at.asc(), Message.id.asc())
        return list(session.scalars(statement))


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_creates_conversation_and_returns_conversation_id(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo has worked on AI systems.")
    client, session_factory = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Tell me about Tumelo's AI projects"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "conversation_id": body["conversation_id"],
        "message": "Tumelo has worked on AI systems.",
        "model": "gpt-4.1-mini",
        "prompt_version": "v1",
        "latency_ms": 842,
        "token_usage": {
            "input_tokens": 1200,
            "output_tokens": 180,
            "total_tokens": 1380,
        },
    }

    with session_factory() as session:
        conversation = session.get(Conversation, body["conversation_id"])
        assert conversation is not None
        assert conversation.model == "gpt-4.1-mini"
        assert conversation.prompt_version == "v1"


def test_chat_stores_user_and_assistant_messages(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Production readiness is strongest in the RAG stack.")
    client, session_factory = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": " Which project best shows production readiness? "},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    messages = fetch_messages(session_factory)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Which project best shows production readiness?"
    assert messages[0].model is None
    assert messages[1].content == "Production readiness is strongest in the RAG stack."
    assert messages[1].model == "gpt-4.1-mini"
    assert messages[1].prompt_version == "v1"
    assert messages[1].latency_ms == 842
    assert messages[1].input_tokens == 1200
    assert messages[1].output_tokens == 180
    assert messages[1].total_tokens == 1380


def test_chat_with_existing_conversation_appends_messages_and_loads_history(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, session_factory = build_test_client(tmp_path, fake_llm)

    first_response = client.post("/chat", json={"message": "Tell me about Tumelo's AI projects"})
    conversation_id = first_response.json()["conversation_id"]

    second_response = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "message": "Which one best shows production readiness?",
        },
    )

    app.dependency_overrides.clear()

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    messages = fetch_messages(session_factory)
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]
    assert len(fake_llm.calls) == 2
    assert [(message.role, message.content) for message in fake_llm.calls[0]] == [
        ("user", "Tell me about Tumelo's AI projects"),
    ]
    assert [(message.role, message.content) for message in fake_llm.calls[1]] == [
        ("user", "Tell me about Tumelo's AI projects"),
        ("assistant", "Mocked assistant response."),
        ("user", "Which one best shows production readiness?"),
    ]


def test_chat_rejects_empty_message(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_test_client(tmp_path, fake_llm)
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_message(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "   "})

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "Chat message cannot be empty."}


def test_chat_rejects_invalid_conversation_id(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Hello", "conversation_id": "not-a-uuid"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "conversation_id must be a valid UUID."}


def test_chat_returns_not_found_for_missing_conversation(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Hello", "conversation_id": str(uuid.uuid4())},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_llm_failures_leave_user_message_persisted_without_assistant_message(tmp_path) -> None:
    fake_llm = FakeLLMService(fail=True)
    client, session_factory = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Hello"})

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Unable to generate assistant response. Please try again."
    }
    assert "sk-test-should-not-leak" not in response.text

    messages = fetch_messages(session_factory)
    assert [message.role for message in messages] == ["user"]
    assert messages[0].content == "Hello"


def test_chat_loads_last_ten_messages_for_follow_up(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, session_factory = build_test_client(tmp_path, fake_llm)
    conversation_id = str(uuid.uuid4())
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    with session_factory() as session:
        conversation = Conversation(
            id=conversation_id,
            model="gpt-4.1-mini",
            prompt_version="v1",
            created_at=base_time,
            updated_at=base_time,
        )
        session.add(conversation)

        for index in range(12):
            session.add(
                Message(
                    id=str(uuid.uuid4()),
                    conversation=conversation,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"message-{index + 1}",
                    created_at=base_time + timedelta(minutes=index + 1),
                )
            )

        session.commit()

    response = client.post(
        "/chat",
        json={"conversation_id": conversation_id, "message": "new-follow-up"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(fake_llm.calls) == 1
    assert [message.content for message in fake_llm.calls[0]] == [
        "message-4",
        "message-5",
        "message-6",
        "message-7",
        "message-8",
        "message-9",
        "message-10",
        "message-11",
        "message-12",
        "new-follow-up",
    ]
