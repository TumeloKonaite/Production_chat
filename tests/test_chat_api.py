from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
import json
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies.chat_dependencies import (
    get_llm_service,
    get_observability_tracer,
    get_request_lock,
    get_response_cache,
    get_retrieval_service,
    get_trace_service,
)
from app.api.dependencies.common_dependencies import get_app_settings, get_db_session
import app.api.health.routes as health_routes
from app.config import Settings
from app.domain.tracing import TraceStatus, TraceStepType
from app.infrastructure.observability import ObservabilityTrace
from app.infrastructure.llm import UnknownModelError
from app.main import app
from app.repositories.db.base import Base
from app.repositories.models import (
    ChatTrace,
    ChatTraceStep,
    Conversation,
    KnowledgeChunk,
    Message,
    MessageFeedback,
    RetrievalLog,
)
from app.services.cache import (
    CacheScope,
    CacheLookupResult,
    CacheStoreEntry,
    ResponseCacheLookupOutcome,
    ResponseCacheStoreOutcome,
    hash_scope,
    stable_hash,
)
from app.services.llm import (
    LLMChatMessage,
    LLMGeneratedResponse,
    LLMServiceError,
    ModelConfig,
    TokenUsage,
)
from app.services.retrieval import RetrievedChunk
from app.services.tracing import TraceServiceError


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
        self.response_model_name_overrides: dict[str, str] = {}

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
        response_model_name = self.response_model_name_overrides.get(
            selected_model.config_id,
            selected_model.model,
        )

        return LLMGeneratedResponse(
            message=self.reply,
            model=response_model_name,
            model_provider=selected_model.provider,
            model_name=response_model_name,
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
    def __init__(
        self,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> None:
        self.retrieved_chunks = (
            [build_retrieved_chunk()] if retrieved_chunks is None else list(retrieved_chunks)
        )
        self.calls: list[tuple[str, int | None]] = []
        self.query_embeddings_used: list[list[float] | None] = []
        self.embed_query_calls: list[str] = []
        self.query_embedding = [0.11, 0.22, 0.33] if query_embedding is None else list(query_embedding)
        self.reranker_enabled = False
        self.reranker_type = "none"
        self.vector_store_name = "pgvector"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        self.query_embeddings_used.append(query_embedding)
        return list(self.retrieved_chunks)

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls.append(query)
        return list(self.query_embedding)


class FakeResponseCache:
    def __init__(
        self,
        *,
        exact_outcome: ResponseCacheLookupOutcome | None = None,
        semantic_outcome: ResponseCacheLookupOutcome | None = None,
        store_outcome: ResponseCacheStoreOutcome | None = None,
    ) -> None:
        self.exact_outcome = exact_outcome or ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="disabled",
            latency_ms=0,
        )
        self.semantic_outcome = semantic_outcome or ResponseCacheLookupOutcome(
            cache_type="semantic",
            hit=False,
            reason="disabled",
            latency_ms=0,
        )
        self.store_outcome = store_outcome or ResponseCacheStoreOutcome(
            success=False,
            reason="write_skipped",
            latency_ms=0,
        )
        self.exact_requests: list[object] = []
        self.semantic_requests: list[object] = []
        self.store_entries: list[CacheStoreEntry] = []

    async def get_exact(self, request) -> ResponseCacheLookupOutcome:
        self.exact_requests.append(request)
        if callable(self.exact_outcome):
            return self.exact_outcome(request)
        return self.exact_outcome

    async def get_semantic(self, request) -> ResponseCacheLookupOutcome:
        self.semantic_requests.append(request)
        if callable(self.semantic_outcome):
            return self.semantic_outcome(request)
        return self.semantic_outcome

    async def store(self, entry: CacheStoreEntry) -> ResponseCacheStoreOutcome:
        self.store_entries.append(entry)
        return self.store_outcome


class FailingTraceService:
    def start_trace(self, **kwargs):
        raise TraceServiceError()

    def add_step(self, **kwargs):
        raise TraceServiceError()

    def complete_trace(self, *args, **kwargs):
        raise TraceServiceError()

    def fail_trace(self, *args, **kwargs):
        raise TraceServiceError()


class FailingObservabilityTracer:
    def start_chat_request(self, **kwargs):
        raise RuntimeError("Langfuse start failed")

    def start_retrieval(self, *args, **kwargs):
        raise RuntimeError("Langfuse retrieval failed")

    def complete_retrieval(self, *args, **kwargs):
        raise RuntimeError("Langfuse retrieval completion failed")

    def start_llm_call(self, *args, **kwargs):
        raise RuntimeError("Langfuse llm failed")

    def complete_llm_call(self, *args, **kwargs):
        raise RuntimeError("Langfuse llm completion failed")

    def complete_chat_request(self, *args, **kwargs):
        raise RuntimeError("Langfuse completion failed")

    def capture_error(self, *args, **kwargs):
        raise RuntimeError("Langfuse error failed")

    def flush(self):
        raise RuntimeError("Langfuse flush failed")


class ProviderObservabilityTracer:
    def start_chat_request(self, **kwargs):
        return ObservabilityTrace(
            provider="langfuse",
            external_trace_id="lf-trace-123",
        )

    def start_retrieval(self, *args, **kwargs):
        return None

    def complete_retrieval(self, *args, **kwargs):
        return None

    def start_llm_call(self, *args, **kwargs):
        return None

    def complete_llm_call(self, *args, **kwargs):
        return None

    def complete_chat_request(self, *args, **kwargs):
        return None

    def capture_error(self, *args, **kwargs):
        return None

    def flush(self):
        return None


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


def build_cache_lookup_result(
    *,
    answer_text: str = "Tumelo builds AI systems.",
    cache_type: str = "exact",
    metadata_scope_hash: str | None = None,
    prompt_version: str = "v1_professional",
    expires_at: datetime | None = None,
    total_latency_ms: int = 1000,
) -> CacheLookupResult:
    created_at = datetime.now(timezone.utc)
    retriever_config_hash = stable_hash(
        json.dumps(
            {
                "retriever_type": "vector",
                "top_k": 5,
                "retrieval_min_similarity": 0.55,
                "default_retrieval_config": "default",
                "query_rewrite_enabled": False,
                "query_rewrite_model": "openai:gpt-4.1-mini",
                "query_rewrite_prompt_version": "v1",
                "query_rewrite_temperature": 0.0,
                "reranker_enabled": False,
                "reranker_type": "none",
                "reranker_model": "openai:gpt-4.1-mini",
                "reranker_initial_top_k": 20,
                "reranker_final_top_k": 5,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    scope = CacheScope(
        knowledge_base_version="personal_knowledge_base",
        prompt_version=prompt_version,
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash=retriever_config_hash,
    )
    return CacheLookupResult(
        entry_id="entry-1",
        cache_type="semantic" if cache_type == "semantic" else "exact",
        normalized_question="tell me about tumelo's work.",
        question_hash="question-hash",
        answer_text=answer_text,
        source_documents=[{"chunk_id": "chunk-1", "source": "projects.md"}],
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        prompt_version=prompt_version,
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        knowledge_base_version="personal_knowledge_base",
        retriever_type="vector",
        top_k=5,
        query_rewrite_enabled=False,
        reranker_enabled=False,
        retriever_config_hash=retriever_config_hash,
        metadata_scope_hash=metadata_scope_hash or hash_scope(scope),
        retrieval_config="default",
        created_at=created_at,
        expires_at=expires_at or (created_at + timedelta(hours=1)),
        last_hit_at=None,
        hit_count=0,
        total_latency_ms=total_latency_ms,
        embedding_latency_ms=18,
        retrieval_latency_ms=120,
        llm_latency_ms=842,
    )


def build_test_settings(
    *,
    default_prompt_version: str = "v1_professional",
    **overrides: object,
) -> Settings:
    legacy_cache_enabled = bool(overrides.get("enable_response_cache", False))
    legacy_rate_limit_enabled = bool(overrides.get("enable_rate_limiting", False))
    enable_redis = bool(
        overrides.get(
            "enable_redis",
            legacy_cache_enabled or legacy_rate_limit_enabled or overrides.get("request_lock_enabled", False),
        )
    )
    values: dict[str, object] = {
        "database_url": "sqlite:///unused-for-tests.db",
        "openai_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": "openrouter-test-key",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "tavus_api_key": "tavus-test-key",
        "tavus_base_url": "https://tavus.example",
        "tavus_face_id": "face_123",
        "tavus_pal_id": "pal_123",
        "public_backend_url": "https://backend.example",
        "tavus_tool_secret": "tool-secret",
        "ingestion_api_secret": "ingestion-secret",
        "eval_admin_token": "eval-secret",
        "default_model_config_id": "openai:gpt-4.1-mini",
        "model_configs_json": None,
        "embedding_provider": "hf",
        "knowledge_embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "knowledge_collection_name": "personal_knowledge_base",
        "default_prompt_version": default_prompt_version,
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "personal-chatbot-model-comparison",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "enable_redis": enable_redis,
        "upstash_redis_rest_url": "https://redis.example.test",
        "upstash_redis_rest_token": "upstash-test-token",
        "rate_limit_enabled": legacy_rate_limit_enabled,
        "rate_limit_max_requests": 20,
        "rate_limit_window_seconds": 60,
        "exact_cache_enabled": legacy_cache_enabled,
        "exact_cache_ttl_seconds": 300,
        "request_lock_enabled": False,
        "request_lock_ttl_seconds": 30,
        "response_cache_knowledge_base_version": "personal_knowledge_base",
    }
    values.update(overrides)
    return Settings(**values)


def build_test_client(
    tmp_path,
    fake_llm: FakeLLMService,
    fake_retrieval: FakeRetrievalService | None = None,
    fake_response_cache: FakeResponseCache | None = None,
    *,
    default_prompt_version: str = "v1_professional",
    settings_overrides: dict[str, object] | None = None,
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
    settings = build_test_settings(
        default_prompt_version=default_prompt_version,
        **(settings_overrides or {}),
    )
    response_cache = fake_response_cache or FakeResponseCache()
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_llm_service] = lambda: fake_llm
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_response_cache] = lambda: response_cache
    return TestClient(app), session_factory, retrieval_service


def fetch_messages(session_factory: sessionmaker[Session]) -> list[Message]:
    with session_factory() as session:
        statement = select(Message).order_by(Message.created_at.asc(), Message.id.asc())
        return list(session.scalars(statement))


def fetch_retrieval_logs(session_factory: sessionmaker[Session]) -> list[RetrievalLog]:
    with session_factory() as session:
        statement = select(RetrievalLog).order_by(RetrievalLog.created_at.asc(), RetrievalLog.id.asc())
        return list(session.scalars(statement))


def fetch_message_feedback(session_factory: sessionmaker[Session]) -> list[MessageFeedback]:
    with session_factory() as session:
        statement = select(MessageFeedback).order_by(
            MessageFeedback.created_at.asc(),
            MessageFeedback.id.asc(),
        )
        return list(session.scalars(statement))


def fetch_chat_traces(session_factory: sessionmaker[Session]) -> list[ChatTrace]:
    with session_factory() as session:
        statement = select(ChatTrace).order_by(ChatTrace.created_at.asc(), ChatTrace.id.asc())
        return list(session.scalars(statement))


def fetch_chat_trace_steps(session_factory: sessionmaker[Session]) -> list[ChatTraceStep]:
    with session_factory() as session:
        statement = select(ChatTraceStep).order_by(ChatTraceStep.step_index.asc(), ChatTraceStep.created_at.asc())
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


def test_ready_returns_ok_when_database_is_reachable(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_ready_returns_service_unavailable_when_database_is_down() -> None:
    class FailingSession:
        def execute(self, *_args, **_kwargs) -> None:
            raise SQLAlchemyError("database unavailable")

    def override_db_session() -> Generator[FailingSession, None, None]:
        yield FailingSession()

    app.dependency_overrides[get_db_session] = override_db_session
    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}


def test_ready_returns_service_unavailable_when_database_driver_raises_non_sqlalchemy_error() -> None:
    class FailingSession:
        def execute(self, *_args, **_kwargs) -> None:
            raise UnicodeError("label empty or too long")

    def override_db_session() -> Generator[FailingSession, None, None]:
        yield FailingSession()

    app.dependency_overrides[get_db_session] = override_db_session
    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}


def test_ready_returns_ok_with_redis_when_enabled_and_reachable(tmp_path, monkeypatch) -> None:
    class HealthyRedisClient:
        async def get(self, key: str) -> str | None:
            return None

    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_redis": True,
            "upstash_redis_rest_url": "https://cache.example.com",
            "upstash_redis_rest_token": "redis-secret",
        },
    )
    monkeypatch.setattr(
        health_routes,
        "build_cache_client",
        lambda _settings: HealthyRedisClient(),
    )

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok"}


def test_ready_returns_service_unavailable_when_redis_enabled_without_url(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        settings_overrides={
            "enable_redis": True,
            "upstash_redis_rest_url": None,
            "upstash_redis_rest_token": None,
        },
    )

    response = client.get("/ready")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "ok",
        "redis": "misconfigured",
    }


def test_chat_creates_conversation_and_returns_conversation_id(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo has worked on AI systems.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Tell me about Tumelo's AI projects"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "conversation_id": body["conversation_id"],
        "message_id": body["message_id"],
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
        "response_cache_hit": False,
        "response_cache_type": None,
        "response_cache_reason": "disabled",
        "response_cache_distance": None,
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
    assert isinstance(messages[1].message_metadata.get("trace_id"), str)


def test_chat_creates_internal_trace_records(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tumelo built a production-ready chatbot.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    response = client.post("/chat", json={"message": "Tell me about Tumelo's chatbot project"})

    app.dependency_overrides.clear()

    assert response.status_code == 200

    traces = fetch_chat_traces(session_factory)
    steps = fetch_chat_trace_steps(session_factory)

    assert len(traces) == 1
    assert traces[0].status == TraceStatus.SUCCESS
    assert traces[0].input_text == "Tell me about Tumelo's chatbot project"
    assert traces[0].output_text == "Tumelo built a production-ready chatbot."
    assert traces[0].llm_provider == "openai"
    assert traces[0].llm_model == "gpt-4.1-mini"
    assert traces[0].prompt_version == "v1_professional"
    assert traces[0].retriever_type == "vector"
    assert traces[0].observability_provider is None
    assert traces[0].external_trace_id is None
    assert traces[0].trace_metadata["route"] == "/chat"
    assert traces[0].trace_metadata["channel"] == "web_chat"
    assert traces[0].trace_metadata["app_environment"] == "local"
    assert traces[0].trace_metadata["vector_store_provider"] == "pgvector"
    assert traces[0].trace_metadata["embedding_dimension"] == 384
    assert traces[0].trace_metadata["top_k"] == 5
    assert traces[0].trace_metadata["internal_trace_id"] == traces[0].id
    assert [step.step_type for step in steps] == [
        TraceStepType.REQUEST_RECEIVED,
        TraceStepType.RETRIEVAL_STARTED,
        TraceStepType.RETRIEVAL_COMPLETED,
        TraceStepType.PROMPT_BUILT,
        TraceStepType.LLM_CALL_STARTED,
        TraceStepType.LLM_CALL_COMPLETED,
        TraceStepType.RESPONSE_GENERATED,
    ]
    assert [step.step_index for step in steps] == [1, 2, 3, 4, 5, 6, 7]
    assert steps[2].output_payload == {
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-1",
                "source_document_id": None,
                "source": "projects.md",
                "source_type": "markdown",
                "section": "Portfolio Chatbot",
                "retrieval_rank": None,
                "final_rank": None,
                "score": 0.91,
            }
        ]
    }


def test_chat_persists_external_langfuse_trace_id_when_available(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Trace-aware response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)
    app.dependency_overrides[get_observability_tracer] = lambda: ProviderObservabilityTracer()

    response = client.post("/chat", json={"message": "Tell me about observability."})

    app.dependency_overrides.clear()

    assert response.status_code == 200

    traces = fetch_chat_traces(session_factory)
    messages = fetch_messages(session_factory)

    assert traces[0].observability_provider == "langfuse"
    assert traces[0].external_trace_id == "lf-trace-123"
    assert traces[0].trace_metadata["external_trace_id"] == "lf-trace-123"
    assert traces[0].trace_metadata["langfuse_trace_id"] == "lf-trace-123"
    assert messages[1].message_metadata["external_trace_id"] == "lf-trace-123"
    assert messages[1].message_metadata["langfuse_trace_id"] == "lf-trace-123"


def test_chat_exact_cache_hit_skips_embedding_retrieval_and_llm(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="This should not be used.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=lambda request: ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=4,
            entry=build_cache_lookup_result(
                answer_text="Cached exact answer.",
                metadata_scope_hash=request.metadata_scope_hash,
                prompt_version=request.metadata_scope.prompt_version,
            ),
            entry_id="entry-1",
        )
    )
    client, session_factory, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Cached exact answer."
    assert response.json()["latency_ms"] == 0
    assert response.json()["response_cache_hit"] is True
    assert response.json()["response_cache_type"] == "exact"
    assert response.json()["response_cache_reason"] == "exact_hit"
    assert response.json()["response_cache_distance"] is None
    assert fake_llm.calls == []
    assert retrieval_service.embed_query_calls == []
    assert retrieval_service.calls == []

    traces = fetch_chat_traces(session_factory)
    assert traces[0].trace_metadata["response_cache"]["response_cache_hit"] is True
    assert traces[0].trace_metadata["response_cache"]["response_cache_type"] == "exact"
    assert traces[0].trace_metadata["response_cache"]["response_cache_reason"] == "exact_hit"
    assert traces[0].trace_metadata["response_cache"]["response_cache_lookup_latency_ms"] == 4


def test_chat_cache_miss_writes_exact_cache_entry(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Freshly generated answer.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="miss_no_exact_entry",
            latency_ms=2,
        ),
        store_outcome=ResponseCacheStoreOutcome(
            success=True,
            reason="write_success",
            latency_ms=7,
            entry_id="entry-1",
        ),
    )
    client, session_factory, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["response_cache_hit"] is False
    assert response.json()["response_cache_type"] is None
    assert response.json()["response_cache_reason"] == "miss_no_exact_entry"
    assert response.json()["response_cache_distance"] is None
    assert fake_llm.calls != []
    assert retrieval_service.embed_query_calls == []
    assert retrieval_service.query_embeddings_used == [None]
    assert fake_cache.store_entries[0].question_embedding is None

    traces = fetch_chat_traces(session_factory)
    assert traces[0].trace_metadata["response_cache"]["response_cache_reason"] == "miss_no_exact_entry"
    assert traces[0].trace_metadata["response_cache"]["response_cache_write_reason"] == (
        "write_success"
    )


def test_chat_duplicate_inflight_request_returns_409(tmp_path) -> None:
    class LockAlreadyHeld:
        async def acquire(self, request_hash: str) -> bool:
            return False

        async def release(self, request_hash: str) -> None:
            return None

    fake_llm = FakeLLMService(reply="This should not be used.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="miss_no_exact_entry",
            latency_ms=1,
        )
    )
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={
            "enable_redis": True,
            "exact_cache_enabled": True,
            "request_lock_enabled": True,
        },
    )
    app.dependency_overrides[get_request_lock] = lambda: LockAlreadyHeld()

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json() == {
        "detail": "An identical request is already being processed. Please retry shortly."
    }
    assert fake_llm.calls == []


def test_chat_exact_cache_hit_works_with_versioned_runtime_model_name(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Versioned-model cached answer.")
    fake_llm.response_model_name_overrides["openai:gpt-4.1-mini"] = "gpt-4.1-mini-2025-04-14"
    fake_retrieval = FakeRetrievalService(query_embedding=[0.4, 0.5, 0.6])
    fake_cache = FakeResponseCache(
        exact_outcome=lambda request: ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=4,
            entry=build_cache_lookup_result(
                answer_text="Versioned-model cached answer.",
                metadata_scope_hash=request.metadata_scope_hash,
                prompt_version=request.metadata_scope.prompt_version,
            ),
            entry_id="entry-1",
        )
        if fake_cache.store_entries
        else ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="miss_no_exact_entry",
            latency_ms=2,
        ),
        store_outcome=ResponseCacheStoreOutcome(
            success=True,
            reason="write_success",
            latency_ms=7,
            entry_id="entry-1",
        ),
    )
    client, _, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    first_response = client.post("/chat", json={"message": "What does Tumelo do?"})
    second_response = client.post("/chat", json={"message": "What does Tumelo do?"})

    app.dependency_overrides.clear()

    assert first_response.status_code == 200
    assert first_response.json()["response_cache_hit"] is False
    assert first_response.json()["response_cache_reason"] == "miss_no_exact_entry"
    assert first_response.json()["response_cache_distance"] is None
    assert first_response.json()["model_name"] == "gpt-4.1-mini-2025-04-14"
    assert second_response.status_code == 200
    assert second_response.json()["response_cache_hit"] is True
    assert second_response.json()["response_cache_type"] == "exact"
    assert second_response.json()["response_cache_reason"] == "exact_hit"
    assert second_response.json()["response_cache_distance"] is None
    assert second_response.json()["message"] == "Versioned-model cached answer."
    assert fake_llm.calls != []
    assert retrieval_service.calls == [("What does Tumelo do?", 5)]
    assert fake_cache.store_entries[0].llm_model == "gpt-4.1-mini"


def test_chat_metadata_mismatch_cache_entry_is_treated_as_miss(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Fresh answer after mismatch.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=4,
            entry=build_cache_lookup_result(metadata_scope_hash="wrong-scope"),
            entry_id="entry-1",
        )
    )
    client, session_factory, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Fresh answer after mismatch."
    assert fake_llm.calls != []
    assert retrieval_service.calls == [("Tell me about Tumelo's work.", 5)]
    traces = fetch_chat_traces(session_factory)
    assert traces[0].trace_metadata["response_cache"]["response_cache_reason"] == (
        "miss_metadata_mismatch"
    )


def test_chat_expired_cache_entry_is_treated_as_miss(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Fresh answer after expiry.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=lambda request: ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=True,
            reason="exact_hit",
            latency_ms=4,
            entry=build_cache_lookup_result(
                metadata_scope_hash=request.metadata_scope_hash,
                prompt_version=request.metadata_scope.prompt_version,
                expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            ),
            entry_id="entry-1",
        )
    )
    client, session_factory, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Fresh answer after expiry."
    assert fake_llm.calls != []
    assert retrieval_service.calls == [("Tell me about Tumelo's work.", 5)]
    traces = fetch_chat_traces(session_factory)
    assert traces[0].trace_metadata["response_cache"]["response_cache_reason"] == (
        "miss_expired"
    )


def test_chat_redis_unavailable_does_not_break_normal_path(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Fallback to normal path.")
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="redis_unavailable",
            latency_ms=9,
        )
    )
    client, session_factory, retrieval_service = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Fallback to normal path."
    assert fake_llm.calls != []
    assert retrieval_service.calls == [("Tell me about Tumelo's work.", 5)]
    traces = fetch_chat_traces(session_factory)
    assert traces[0].trace_metadata["response_cache"]["response_cache_reason"] == (
        "redis_unavailable"
    )


def test_failed_llm_response_does_not_write_cache_entry(tmp_path) -> None:
    fake_llm = FakeLLMService(fail=True)
    fake_retrieval = FakeRetrievalService()
    fake_cache = FakeResponseCache(
        exact_outcome=ResponseCacheLookupOutcome(
            cache_type="exact",
            hit=False,
            reason="miss_no_exact_entry",
            latency_ms=1,
        )
    )
    client, _, _ = build_test_client(
        tmp_path,
        fake_llm,
        fake_retrieval,
        fake_cache,
        settings_overrides={"enable_response_cache": True},
    )

    response = client.post("/chat", json={"message": "Tell me about Tumelo's work."})

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert fake_cache.store_entries == []


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

    traces = fetch_chat_traces(session_factory)
    steps = fetch_chat_trace_steps(session_factory)
    assert len(traces) == 1
    assert traces[0].status == TraceStatus.ERROR
    assert traces[0].error_message == "Unable to generate assistant response. Please try again."
    assert "sk-test-should-not-leak" not in (traces[0].error_message or "")
    assert steps[-1].step_type == TraceStepType.ERROR
    assert steps[-1].error_message == "Unable to generate assistant response. Please try again."


def test_trace_failures_do_not_break_chat_response(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Tracing should not block this response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)
    app.dependency_overrides[get_trace_service] = lambda: FailingTraceService()

    response = client.post("/chat", json={"message": "Tell me about tracing."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Tracing should not block this response."
    assert len(fetch_messages(session_factory)) == 2
    assert fetch_chat_traces(session_factory) == []


def test_langfuse_failures_do_not_break_chat_response(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Langfuse should not block this response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)
    app.dependency_overrides[get_observability_tracer] = lambda: FailingObservabilityTracer()

    response = client.post("/chat", json={"message": "Tell me about Langfuse."})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message"] == "Langfuse should not block this response."
    assert len(fetch_messages(session_factory)) == 2
    assert len(fetch_chat_traces(session_factory)) == 1


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


def test_chat_uses_retrieval_service_results_for_non_broad_queries(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="BeautyVerse is Tumelo's beauty services marketplace.")
    fake_retrieval = FakeRetrievalService(
        [
            build_retrieved_chunk(
                source="projects.md",
                section="BeautyVerse - Beauty Services Marketplace",
                content=(
                    "BeautyVerse is a marketplace and full-stack web application that enables "
                    "providers to manage service listings while customers browse beauty services."
                ),
            )
        ]
    )
    client, session_factory, retrieval_service = build_test_client(tmp_path, fake_llm, fake_retrieval)
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
    assert retrieval_service.calls == [("Tell me about Tumelo's BeautyVerse project", 5)]
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


def test_submit_feedback_stores_thumbs_up_for_assistant_message(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Helpful assistant response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    chat_response = client.post("/chat", json={"message": "Tell me about Tumelo's chatbot work"})
    message_id = chat_response.json()["message_id"]

    response = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        json={"rating": "up"},
    )

    app.dependency_overrides.clear()

    assert chat_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["message_id"] == message_id
    assert response.json()["rating"] == "up"
    assert response.json()["comment"] is None

    feedback_records = fetch_message_feedback(session_factory)
    assert len(feedback_records) == 1
    assert feedback_records[0].message_id == message_id
    assert feedback_records[0].conversation_id == chat_response.json()["conversation_id"]
    assert feedback_records[0].rating == "up"
    assert feedback_records[0].comment is None
    assert feedback_records[0].trace_id is not None
    assert feedback_records[0].feedback_metadata["model_config_id"] == "openai:gpt-4.1-mini"
    assert feedback_records[0].feedback_metadata["prompt_version"] == "v1_professional"
    assert feedback_records[0].feedback_metadata["channel"] == "web_chat"

    traces = fetch_chat_traces(session_factory)
    assert len(traces) == 1
    assert traces[0].trace_metadata["feedback"] == {
        "rating": "up",
        "thumb_rating": "up",
        "feedback_rating": "positive",
        "comment": None,
        "feedback_comment": None,
        "message_feedback_id": feedback_records[0].id,
        "message_id": message_id,
        "conversation_id": chat_response.json()["conversation_id"],
        "trace_id": traces[0].id,
    }


def test_submit_feedback_stores_comment_with_thumbs_down(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Assistant response that needs correction.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    chat_response = client.post("/chat", json={"message": "Tell me about Tumelo's backend work"})
    message_id = chat_response.json()["message_id"]

    response = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        json={
            "rating": "down",
            "comment": "The answer did not use the right source document.",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["rating"] == "down"
    assert response.json()["comment"] == "The answer did not use the right source document."

    feedback_records = fetch_message_feedback(session_factory)
    assert len(feedback_records) == 1
    assert feedback_records[0].rating == "down"
    assert feedback_records[0].comment == "The answer did not use the right source document."


def test_submit_feedback_rejects_invalid_rating(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Helpful assistant response.")
    client, _, _ = build_test_client(tmp_path, fake_llm)

    chat_response = client.post("/chat", json={"message": "Tell me about Tumelo"})
    message_id = chat_response.json()["message_id"]

    response = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        json={"rating": "sideways"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_submit_feedback_returns_not_found_for_missing_message(tmp_path) -> None:
    fake_llm = FakeLLMService()
    client, _, _ = build_test_client(tmp_path, fake_llm)

    response = client.post(
        f"/api/chat/messages/{uuid.uuid4()}/feedback",
        json={"rating": "up"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {"detail": "Message not found."}


def test_submit_feedback_rejects_user_message(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Assistant response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    chat_response = client.post("/chat", json={"message": "Tell me about Tumelo's work"})
    user_message_id = fetch_messages(session_factory)[0].id

    response = client.post(
        f"/api/chat/messages/{user_message_id}/feedback",
        json={"rating": "up"},
    )

    app.dependency_overrides.clear()

    assert chat_response.status_code == 200
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Feedback can only be submitted for assistant messages."
    }
    assert fetch_message_feedback(session_factory) == []


def test_submit_feedback_updates_existing_feedback_for_message(tmp_path) -> None:
    fake_llm = FakeLLMService(reply="Initial assistant response.")
    client, session_factory, _ = build_test_client(tmp_path, fake_llm)

    chat_response = client.post("/chat", json={"message": "Tell me about Tumelo's AI systems"})
    message_id = chat_response.json()["message_id"]

    first_response = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        json={"rating": "up"},
    )
    second_response = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        json={
            "rating": "down",
            "comment": "This answer missed the backend APIs.",
        },
    )

    app.dependency_overrides.clear()

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["id"] == second_response.json()["id"]
    assert second_response.json()["rating"] == "down"
    assert second_response.json()["comment"] == "This answer missed the backend APIs."

    feedback_records = fetch_message_feedback(session_factory)
    assert len(feedback_records) == 1
    assert feedback_records[0].id == first_response.json()["id"]
    assert feedback_records[0].rating == "down"
    assert feedback_records[0].comment == "This answer missed the backend APIs."
