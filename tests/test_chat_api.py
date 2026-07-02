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
from app.infrastructure.llm import UnknownModelError
from app.main import app
from app.repositories.db.base import Base
from app.repositories.models import Conversation, KnowledgeChunk, Message, RetrievalLog
from app.services.llm import (
    LLMChatMessage,
    LLMGeneratedResponse,
    LLMServiceError,
    ModelConfig,
    TokenUsage,
)
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
        self.default_model_config_id = "openai:gpt-4.1-mini"
        self.calls: list[list[LLMChatMessage]] = []
        self.system_prompts: list[str] = []
        self.prompt_versions: list[str] = []
        self.model_config_ids: list[str] = []

    @property
    def model(self) -> str:
        return self.get_model_config().model

    def get_model_config(self, model_config_id: str | None = None) -> ModelConfig:
        configs = {
            "openai:gpt-4.1-mini": ModelConfig(
                config_id="openai:gpt-4.1-mini",
                provider="openai",
                model="gpt-4.1-mini",
                input_cost_per_1m_tokens=0.40,
                output_cost_per_1m_tokens=1.60,
            ),
            "openai:gpt-4.1": ModelConfig(
                config_id="openai:gpt-4.1",
                provider="openai",
                model="gpt-4.1",
                input_cost_per_1m_tokens=2.00,
                output_cost_per_1m_tokens=8.00,
            ),
        }
        normalized_model_config_id = model_config_id or self.default_model_config_id
        if ":" not in normalized_model_config_id:
            normalized_model_config_id = f"openai:{normalized_model_config_id}"
        model_config = configs.get(normalized_model_config_id)
        if model_config is None:
            raise UnknownModelError(normalized_model_config_id, sorted(configs))
        return model_config

    async def generate_response(
        self,
        messages: list[LLMChatMessage],
        *,
        system_prompt: str,
        prompt_version: str,
        retrieval_config: str = "default",
        temperature: float | None = None,
        model_config_id: str | None = None,
    ) -> LLMGeneratedResponse:
        self.calls.append(list(messages))
        self.system_prompts.append(system_prompt)
        self.prompt_versions.append(prompt_version)
        if self.fail:
            raise LLMServiceError("sk-test-should-not-leak")

        selected_model = self.get_model_config(model_config_id)
        self.model_config_ids.append(selected_model.config_id)
        estimated_cost_usd = (
            0.000768 if selected_model.config_id == "openai:gpt-4.1-mini" else 0.00384
        )

        return LLMGeneratedResponse(
            message=self.reply,
            model=selected_model.model,
            model_provider=selected_model.provider,
            model_name=selected_model.model,
            model_config_id=selected_model.config_id,
            prompt_version=prompt_version,
            retrieval_config=retrieval_config,
            latency_ms=842,
            token_usage=TokenUsage(
                input_tokens=1200,
                output_tokens=180,
                total_tokens=1380,
            ),
            estimated_cost_usd=estimated_cost_usd,
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
        default_prompt_version=default_prompt_version,
        conversation_history_limit=10,
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="personal-chatbot-model-comparison",
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


def store_knowledge_chunk(
    session_factory: sessionmaker[Session],
    *,
    source: str,
    section: str,
    content: str,
    chunk_index: int = 0,
    section_chunk_index: int = 0,
) -> None:
    with session_factory() as session:
        session.add(
            KnowledgeChunk(
                source=source,
                source_type="markdown",
                section=section,
                content=content,
                chunk_metadata={
                    "source": source,
                    "section": section,
                    "source_type": "markdown",
                    "chunk_index": chunk_index,
                    "section_chunk_index": section_chunk_index,
                },
            )
        )
        session.commit()


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
        "model_provider": "openai",
        "model_name": "gpt-4.1-mini",
        "model_config_id": "openai:gpt-4.1-mini",
        "prompt_version": "v1_professional",
        "retrieval_config": "default",
        "latency_ms": 842,
        "token_usage": {
            "input_tokens": 1200,
            "output_tokens": 180,
            "total_tokens": 1380,
        },
        "estimated_cost_usd": 0.000768,
    }

    with session_factory() as session:
        conversation = session.get(Conversation, body["conversation_id"])
        assert conversation is not None
        assert conversation.model == "openai:gpt-4.1-mini"
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
    assert messages[0].channel == "web_chat"
    assert messages[0].message_metadata == {}
    assert messages[1].content == "Production readiness is strongest in the RAG stack."
    assert messages[1].model == "gpt-4.1-mini"
    assert messages[1].model_provider == "openai"
    assert messages[1].model_name == "gpt-4.1-mini"
    assert messages[1].model_config_id == "openai:gpt-4.1-mini"
    assert messages[1].channel == "web_chat"
    assert messages[1].prompt_version == "v1_professional"
    assert messages[1].retrieval_config == "default"
    assert messages[1].latency_ms == 842
    assert messages[1].input_tokens == 1200
    assert messages[1].output_tokens == 180
    assert messages[1].total_tokens == 1380
    assert messages[1].estimated_cost_usd == pytest.approx(0.000768)
    assert messages[1].message_metadata == {}


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
    assert fake_llm.model_config_ids == ["openai:gpt-4.1-mini", "openai:gpt-4.1-mini"]
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
            model="openai:gpt-4.1-mini",
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


def test_chat_keyword_boost_prefers_project_name_matches_over_semantic_fallback(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="BeautyVerse is Tumelo's beauty services marketplace.")
    fake_retrieval = FakeRetrievalService([])
    client, session_factory, _ = build_test_client(tmp_path, fake_llm, fake_retrieval)
    store_knowledge_chunk(
        session_factory,
        source="experience.md",
        section="Engineering Experience Themes",
        content="Tumelo has experience building production-grade AI systems and software products.",
    )
    store_knowledge_chunk(
        session_factory,
        source="projects.md",
        section="BeautyVerse - Beauty Services Marketplace",
        content=(
            "BeautyVerse is a marketplace and full-stack web application that enables "
            "providers to manage service listings while customers browse beauty services."
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "Tell me about Tumelo's BeautyVerse project"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "BeautyVerse is Tumelo's beauty services marketplace."
    assert len(fake_llm.calls) == 1
    assert "BeautyVerse - Beauty Services Marketplace" in fake_llm.system_prompts[0]
    assert "Engineering Experience Themes" not in fake_llm.system_prompts[0]

    logs = fetch_retrieval_logs(session_factory)
    assert len(logs) == 1
    assert logs[0].retrieved_sources == ["projects.md"]
    assert logs[0].used_fallback is False


def test_chat_broad_project_query_prefers_multiple_project_chunks(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="LetsGo, BeautyVerse, and MedDesk are among Tumelo's projects.")
    fake_retrieval = FakeRetrievalService(
        [
            build_retrieved_chunk(
                source="experience.md",
                section="Engineering Experience Themes",
                content="Generic engineering summary.",
            )
        ]
    )
    client, session_factory, retrieval_service = build_test_client(tmp_path, fake_llm, fake_retrieval)
    store_knowledge_chunk(
        session_factory,
        source="projects.md",
        section="LetsGo South Africa - AI-Powered Tourism Platform",
        content="LetsGo South Africa is an AI-powered tourism platform.",
        chunk_index=0,
    )
    store_knowledge_chunk(
        session_factory,
        source="projects.md",
        section="BeautyVerse - Beauty Services Marketplace",
        content="BeautyVerse is a beauty services marketplace.",
        chunk_index=1,
    )
    store_knowledge_chunk(
        session_factory,
        source="projects.md",
        section="MedDesk - AI Clinical Intake Proof of Concept",
        content="MedDesk is an AI clinical intake proof of concept.",
        chunk_index=2,
        section_chunk_index=0,
    )
    store_knowledge_chunk(
        session_factory,
        source="projects.md",
        section="MedDesk - AI Clinical Intake Proof of Concept",
        content="Additional MedDesk detail chunk.",
        chunk_index=3,
        section_chunk_index=1,
    )
    store_knowledge_chunk(
        session_factory,
        source="experience.md",
        section="Engineering Experience Themes",
        content="Tumelo has broad AI engineering experience.",
        chunk_index=4,
    )

    response = client.post(
        "/chat",
        json={"message": "Tell me about Tumelo's projects"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "LetsGo, BeautyVerse, and MedDesk are among Tumelo's projects."
    assert retrieval_service.calls == []
    assert "LetsGo South Africa - AI-Powered Tourism Platform" in fake_llm.system_prompts[0]
    assert "BeautyVerse - Beauty Services Marketplace" in fake_llm.system_prompts[0]
    assert "MedDesk - AI Clinical Intake Proof of Concept" in fake_llm.system_prompts[0]
    assert "Engineering Experience Themes" not in fake_llm.system_prompts[0]
    assert (
        "summarize the most relevant projects from the approved project context"
        in fake_llm.system_prompts[0]
    )

    logs = fetch_retrieval_logs(session_factory)
    assert len(logs) == 1
    assert logs[0].retrieved_sources == ["projects.md", "projects.md", "projects.md"]
    assert logs[0].used_fallback is False


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
    assert response.json()["model_config_id"] == "openai:gpt-4.1-mini"
    assert response.json()["estimated_cost_usd"] is None
    assert fake_llm.calls == []

    logs = fetch_retrieval_logs(session_factory)
    assert len(logs) == 1
    assert logs[0].retrieved_chunk_ids == []
    assert logs[0].used_fallback is True

    messages = fetch_messages(session_factory)
    assert messages[1].model_config_id == "openai:gpt-4.1-mini"
    assert messages[1].estimated_cost_usd is None


def test_chat_general_question_without_context_still_uses_llm(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="RAG combines retrieval with generation.")
    fake_retrieval = FakeRetrievalService([])
    client, session_factory, _ = build_test_client(tmp_path, fake_llm, fake_retrieval)

    response = client.post("/chat", json={"message": "What is retrieval augmented generation?"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "RAG combines retrieval with generation."
    assert len(fake_llm.calls) == 1
    assert response.json()["model_config_id"] == "openai:gpt-4.1-mini"
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


def test_chat_normalizes_legacy_prompt_version_alias(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo builds practical AI systems.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={
            "message": "Tell me about Tumelo's work.",
            "prompt_version": "v1",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v1_professional"
    assert fake_llm.prompt_versions == ["v1_professional"]

    messages = fetch_messages(session_factory)
    assert messages[1].prompt_version == "v1_professional"

    with session_factory() as session:
        conversation = session.get(Conversation, response.json()["conversation_id"])
        assert conversation is not None
        assert conversation.prompt_version == "v1_professional"


def test_chat_uses_requested_model_config_id(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo also builds higher-capability AI systems.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={
            "message": "Tell me about Tumelo's AI systems.",
            "model_config_id": "openai:gpt-4.1",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gpt-4.1"
    assert body["model_name"] == "gpt-4.1"
    assert body["model_config_id"] == "openai:gpt-4.1"
    assert body["estimated_cost_usd"] == 0.00384
    assert fake_llm.model_config_ids == ["openai:gpt-4.1"]

    messages = fetch_messages(session_factory)
    assert messages[1].model_name == "gpt-4.1"
    assert messages[1].model_config_id == "openai:gpt-4.1"
    assert messages[1].estimated_cost_usd == pytest.approx(0.00384)

    with session_factory() as session:
        conversation = session.get(Conversation, body["conversation_id"])
        assert conversation is not None
        assert conversation.model == "openai:gpt-4.1"


def test_chat_rejects_unknown_model_config_id(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        "/chat",
        json={"message": "Tell me about Tumelo.", "model_config_id": "openai:nope"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unknown model config ID: openai:nope. Available models: openai:gpt-4.1, openai:gpt-4.1-mini"
    }
    assert fetch_messages(session_factory) == []


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
