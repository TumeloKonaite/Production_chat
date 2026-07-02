from __future__ import annotations

import asyncio
from collections.abc import Generator
import uuid

from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.chat_dependencies import get_llm_service, get_retrieval_service
from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
from app.api.dependencies.tavus_dependencies import get_tavus_service
from app.config import Settings
from app.infrastructure.tavus import TavusClient
from app.main import app
from app.repositories.db.base import Base
from app.repositories.models import Conversation, Message
from app.services.retrieval import RetrievedChunk
from tests.test_chat_api import FakeLLMService, FakeRetrievalService


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def build_test_settings() -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        tavus_api_key="tavus-test-key",
        tavus_base_url="https://tavus.example",
        tavus_face_id="face_123",
        tavus_pal_id="pal_123",
        public_backend_url="https://backend.example",
        tavus_tool_secret="tool-secret",
        ingestion_api_secret="ingestion-secret",
        default_model_config_id="openai:gpt-4.1-mini",
        knowledge_embedding_model="all-MiniLM-L6-v2",
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version="v1_professional",
        conversation_history_limit=10,
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="personal-chatbot-model-comparison",
    )


def build_sqlite_client(
    tmp_path,
    fake_llm: FakeLLMService,
    *,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> tuple[TestClient, sessionmaker[Session]]:
    database_path = tmp_path / "test_tavus.db"
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
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_llm_service] = lambda: fake_llm
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(
        retrieved_chunks=retrieved_chunks
    )
    return TestClient(app), session_factory


class FakeTavusService:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, str | None]] = []
        self.end_calls: list[str] = []

    async def create_conversation(
        self,
        *,
        visitor_name: str,
        backend_conversation_id: str | None = None,
    ):
        self.create_calls.append((visitor_name, backend_conversation_id))
        return type(
            "Session",
            (),
            {
                "conversation_id": "tavus-conversation-123",
                "conversation_url": "https://tavus.example/conversations/123",
            },
        )()

    async def end_conversation(self, *, conversation_id: str) -> dict[str, object]:
        self.end_calls.append(conversation_id)
        return {"conversation_id": conversation_id, "status": "ended"}


def test_tavus_client_sends_expected_payload() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = request.content.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "conversation_id": "conv_123",
                "conversation_url": "https://tavus.example/conversations/conv_123",
            },
        )

    client = TavusClient(
        settings=build_test_settings(),
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(
        client.create_conversation(
            face_id="face_123",
            pal_id="pal_123",
            conversational_context="Use the backend tool.",
        )
    )

    assert response["conversation_id"] == "conv_123"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://tavus.example/v2/conversations"
    assert captured["headers"]["x-api-key"] == "tavus-test-key"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["json"] == (
        '{"face_id":"face_123","pal_id":"pal_123","conversational_context":"Use the backend tool."}'
    )


def test_create_tavus_conversation_returns_session_details() -> None:
    fake_tavus_service = FakeTavusService()
    app.dependency_overrides[get_tavus_service] = lambda: fake_tavus_service

    client = TestClient(app)
    response = client.post(
        "/api/tavus/conversations",
        json={"visitor_name": "Portfolio visitor", "conversation_id": str(uuid.uuid4())},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "tavus-conversation-123",
        "conversation_url": "https://tavus.example/conversations/123",
    }
    assert len(fake_tavus_service.create_calls) == 1
    assert fake_tavus_service.create_calls[0][0] == "Portfolio visitor"


def test_tavus_tool_endpoint_rejects_missing_secret(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_sqlite_client(tmp_path, fake_llm)

    response = client.post(
        "/api/tavus/tools/ask-tumelo",
        json={"message": "What projects has Tumelo built?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid Tavus tool secret."}


def test_tavus_tool_endpoint_rejects_invalid_secret(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _ = build_sqlite_client(tmp_path, fake_llm)

    response = client.post(
        "/api/tavus/tools/ask-tumelo",
        headers={"x-tavus-tool-secret": "wrong-secret"},
        json={"message": "What projects has Tumelo built?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid Tavus tool secret."}


def test_tavus_tool_endpoint_accepts_valid_secret_and_calls_chat_service(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo has built AI chatbots.")
    client, session_factory = build_sqlite_client(tmp_path, fake_llm)

    response = client.post(
        "/api/tavus/tools/ask-tumelo",
        headers={"x-tavus-tool-secret": "tool-secret"},
        json={
            "message": "What projects has Tumelo built?",
            "tavus_conversation_id": "tavus-external-conv-1",
            "visitor_name": "Amina",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Tumelo has built AI chatbots."}
    assert len(fake_llm.calls) == 1

    with session_factory() as session:
        conversation = session.scalar(
            select(Conversation).where(Conversation.visitor_id == "tavus-external-conv-1")
        )
        assert conversation is not None
        assert conversation.title == "Amina"

        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.asc(), Message.id.asc())
            )
        )

    assert [message.role for message in messages] == ["user", "assistant"]
    assert all(message.channel == "tavus_video" for message in messages)
    assert messages[0].message_metadata == {
        "visitor_name": "Amina",
        "source": "tavus_tool_call",
        "tavus_conversation_id": "tavus-external-conv-1",
    }
    assert messages[1].message_metadata == messages[0].message_metadata


def test_end_tavus_conversation_calls_service() -> None:
    fake_tavus_service = FakeTavusService()
    app.dependency_overrides[get_tavus_service] = lambda: fake_tavus_service

    client = TestClient(app)
    response = client.post(
        "/api/tavus/conversations/end",
        json={"conversation_id": "tavus-conversation-123"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "tavus-conversation-123",
        "status": "ended",
    }
    assert fake_tavus_service.end_calls == ["tavus-conversation-123"]


def test_tavus_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/tavus/conversations" in paths
    assert "/api/tavus/tools/ask-tumelo" in paths
    assert "/api/tavus/conversations/end" in paths
