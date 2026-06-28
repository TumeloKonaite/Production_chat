from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.chat_dependencies import get_llm_service, get_retrieval_service
from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.config import Settings
from app.main import app
from app.repositories.db.base import Base
from app.repositories.models import Conversation, Message, RetrievalLog
from app.services.llm import LLMChatMessage, LLMGeneratedResponse, LLMServiceError, TokenUsage
from app.services.retrieval import RetrievedChunk


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
        self.calls: list[list[LLMChatMessage]] = []
        self.system_prompts: list[str] = []
        self.prompt_versions: list[str] = []

    async def generate_response(
        self,
        messages: list[LLMChatMessage],
        *,
        system_prompt: str,
        prompt_version: str,
        temperature: float | None = None,
    ) -> LLMGeneratedResponse:
        self.calls.append(list(messages))
        self.system_prompts.append(system_prompt)
        self.prompt_versions.append(prompt_version)
        if self.fail:
            raise LLMServiceError("sk-test-should-not-leak")

        return LLMGeneratedResponse(
            message=self.reply,
            model=self.model,
            prompt_version=prompt_version,
            latency_ms=842,
            token_usage=TokenUsage(
                input_tokens=1200,
                output_tokens=180,
                total_tokens=1380,
            ),
        )


class FakeRetrievalService:
    def __init__(self, retrieved_chunks: list[RetrievedChunk] | None = None) -> None:
        self.retrieved_chunks = (
            [build_retrieved_chunk()] if retrieved_chunks is None else list(retrieved_chunks)
        )
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return list(self.retrieved_chunks)


def build_retrieved_chunk(
    *,
    content: str = "Tumelo built a FastAPI chatbot backed by a curated knowledge base.",
    source: str = "projects.md",
    section: str = "Portfolio Chatbot",
    similarity: float = 0.91,
) -> RetrievedChunk:
    return RetrievedChunk(
        id="chunk-1",
        source=source,
        section=section,
        content=content,
        similarity=similarity,
        metadata={
            "chunk_id": "chunk-1",
            "source": source,
            "section": section,
            "source_type": "markdown",
        },
    )


def build_test_settings(*, default_prompt_version: str = "v1_professional") -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_model="gpt-4.1-mini",
        knowledge_embedding_model="all-MiniLM-L6-v2",
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version=default_prompt_version,
        conversation_history_limit=10,
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="portfolio-chatbot-prompt-experiments",
    )


def build_test_client(
    tmp_path,
    fake_llm: FakeLLMService,
    fake_retrieval: FakeRetrievalService | None = None,
    *,
    default_prompt_version: str = "v1_professional",
) -> tuple[TestClient, sessionmaker[Session], FakeRetrievalService]:
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

    retrieval_service = fake_retrieval or FakeRetrievalService()
    settings = build_test_settings(default_prompt_version=default_prompt_version)
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_llm_service] = lambda: fake_llm
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    return TestClient(app), session_factory, retrieval_service


def fetch_messages(session_factory: sessionmaker[Session]) -> list[Message]:
    with session_factory() as session:
        statement = select(Message).order_by(Message.created_at.asc(), Message.id.asc())
        return list(session.scalars(statement))


def fetch_retrieval_logs(session_factory: sessionmaker[Session]) -> list[RetrievalLog]:
    with session_factory() as session:
        statement = select(RetrievalLog).order_by(RetrievalLog.created_at.asc(), RetrievalLog.id.asc())
        return list(session.scalars(statement))


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_creates_conversation_and_returns_conversation_id(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo has worked on AI systems.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Tell me about Tumelo's AI projects"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "conversation_id": body["conversation_id"],
        "message": "Tumelo has worked on AI systems.",
        "model": "gpt-4.1-mini",
        "prompt_version": "v1_professional",
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
        assert conversation.prompt_version == "v1_professional"


def test_chat_stores_user_and_assistant_messages(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Production readiness is strongest in the RAG stack.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

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
    assert messages[1].prompt_version == "v1_professional"
    assert messages[1].latency_ms == 842
    assert messages[1].input_tokens == 1200
    assert messages[1].output_tokens == 180
    assert messages[1].total_tokens == 1380


def test_chat_with_existing_conversation_appends_messages_and_loads_history(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

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
    assert fake_llm.prompt_versions == ["v1_professional", "v1_professional"]
    assert [(message.role, message.content) for message in fake_llm.calls[1]] == [
        ("user", "Tell me about Tumelo's AI projects"),
        ("assistant", "Mocked assistant response."),
        ("user", "Which one best shows production readiness?"),
    ]


def test_chat_rejects_empty_message(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_message(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "   "})

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "Chat message cannot be empty."}


def test_chat_rejects_invalid_conversation_id(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Hello", "conversation_id": "not-a-uuid"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": "conversation_id must be a valid UUID."}


def test_chat_returns_not_found_for_missing_conversation(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Hello", "conversation_id": str(uuid.uuid4())},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}


def test_llm_failures_leave_user_message_persisted_without_assistant_message(tmp_path) -> None:
    fake_llm = FakeLLMService(fail=True)
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

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
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)
    conversation_id = str(uuid.uuid4())
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    with session_factory() as session:
        conversation = Conversation(
            id=conversation_id,
            model="gpt-4.1-mini",
            prompt_version="v1_professional",
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


def test_chat_injects_retrieved_context_into_system_prompt(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo built a retrieval-grounded chatbot.")
    fake_retrieval = FakeRetrievalService(
        [build_retrieved_chunk(content="Tumelo built a retrieval-grounded FastAPI chatbot.")]
    )
    client, _, _ = build_test_client(tmp_path, fake_llm, fake_retrieval)

    response = client.post("/chat", json={"message": "Tell me about Tumelo's chatbot project"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(fake_llm.system_prompts) == 1
    assert "Approved Tumelo knowledge base context" in fake_llm.system_prompts[0]
    assert "Tumelo built a retrieval-grounded FastAPI chatbot." in fake_llm.system_prompts[0]
    assert "Answer as a professional assistant representing Tumelo." in fake_llm.system_prompts[0]
    assert "Do not invent experience, projects, employers, dates, tools, certifications, or achievements." in fake_llm.system_prompts[0]


def test_chat_personal_query_without_context_returns_safe_fallback(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="This should not be used.")
    fake_retrieval = FakeRetrievalService([])
    client, session_factory, _ = build_test_client(tmp_path, fake_llm, fake_retrieval)

    response = client.post("/chat", json={"message": "What companies has Tumelo worked for?"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == (
        "I do not have enough approved information about that in Tumelo's knowledge base yet."
    )
    assert fake_llm.calls == []

    logs = fetch_retrieval_logs(session_factory)
    assert len(logs) == 1
    assert logs[0].retrieved_chunk_ids == []
    assert logs[0].used_fallback is True


def test_chat_general_question_without_context_still_uses_llm(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="RAG combines retrieval with generation.")
    fake_retrieval = FakeRetrievalService([])
    client, session_factory, _ = build_test_client(tmp_path, fake_llm, fake_retrieval)

    response = client.post("/chat", json={"message": "What is retrieval augmented generation?"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "RAG combines retrieval with generation."
    assert len(fake_llm.calls) == 1
    assert "No relevant approved Tumelo context was retrieved for this turn" in fake_llm.system_prompts[0]

    logs = fetch_retrieval_logs(session_factory)
    assert len(logs) == 1
    assert logs[0].used_fallback is False


def test_chat_uses_requested_prompt_version(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo builds practical AI systems.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={
            "message": "Tell me about Tumelo's work.",
            "prompt_version": "v2_warm_conversational",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v2_warm_conversational"
    assert fake_llm.prompt_versions == ["v2_warm_conversational"]
    assert "Tone:\n- warm" in fake_llm.system_prompts[0]

    messages = fetch_messages(session_factory)
    assert messages[1].prompt_version == "v2_warm_conversational"

    with session_factory() as session:
        conversation = session.get(Conversation, response.json()["conversation_id"])
        assert conversation is not None
        assert conversation.prompt_version == "v2_warm_conversational"


def test_chat_rejects_unknown_prompt_version(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Tell me about Tumelo.", "prompt_version": "v999_unknown"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unknown prompt version: v999_unknown. Available versions: v1_professional, v2_warm_conversational"
    }
    assert fetch_messages(session_factory) == []
