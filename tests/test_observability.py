from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import pytest

from app.config import Settings
from app.infrastructure.observability import NoOpTracer, get_tracer
from app.infrastructure.observability.langfuse_client import LangfuseClient
from app.infrastructure.observability.tracer import CONTENT_PREVIEW_LIMIT, LangfuseTracer
from app.services.retrieval import RetrievedChunk


def build_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite:///observability-tests.db",
        "openai_api_key": "test-key",
        "openai_base_url": "https://api.openai.com/v1",
        "openrouter_api_key": "openrouter-key",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "tavus_api_key": None,
        "tavus_base_url": "https://tavusapi.com",
        "tavus_face_id": None,
        "tavus_pal_id": None,
        "public_backend_url": None,
        "tavus_tool_secret": None,
        "ingestion_api_secret": None,
        "eval_admin_token": None,
        "default_model_config_id": "openai:gpt-4.1-mini",
        "model_configs_json": None,
        "embedding_provider": "hf",
        "knowledge_embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "knowledge_collection_name": "personal_knowledge_base",
        "default_prompt_version": "v1_professional",
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "production-chatbot",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
        "enable_langfuse_observability": True,
        "langfuse_public_key": "pk-lf-test",
        "langfuse_secret_key": "sk-lf-test",
        "langfuse_base_url": "https://cloud.langfuse.com",
        "langfuse_environment": "production",
        "langfuse_release": "modal-v1",
        "langfuse_sample_rate": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@dataclass
class FakeObservation:
    name: str
    as_type: str
    updates: list[dict[str, object]] = field(default_factory=list)
    ended: int = 0
    fail_on_update: bool = False

    def update(self, **kwargs: object) -> None:
        if self.fail_on_update:
            raise RuntimeError("update failed")
        self.updates.append(dict(kwargs))

    def end(self) -> None:
        self.ended += 1


@dataclass
class FakeObservationContext:
    observation: FakeObservation
    exited: bool = False

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class FakeSdkClient:
    def __init__(self, *, fail_on_update: bool = False) -> None:
        self.fail_on_update = fail_on_update
        self.root_contexts: list[FakeObservationContext] = []
        self.child_observations: list[FakeObservation] = []
        self.flush_calls = 0

    def start_as_current_observation(self, **kwargs: object) -> FakeObservationContext:
        observation = FakeObservation(
            name=str(kwargs["name"]),
            as_type=str(kwargs["as_type"]),
            fail_on_update=self.fail_on_update,
        )
        context = FakeObservationContext(observation=observation)
        observation.updates.append(
            {
                "input": kwargs.get("input"),
                "metadata": kwargs.get("metadata"),
                "version": kwargs.get("version"),
            }
        )
        self.root_contexts.append(context)
        return context

    def start_observation(self, **kwargs: object) -> FakeObservation:
        observation = FakeObservation(
            name=str(kwargs["name"]),
            as_type=str(kwargs.get("as_type", "span")),
            fail_on_update=self.fail_on_update,
        )
        observation.updates.append(
            {
                "input": kwargs.get("input"),
                "metadata": kwargs.get("metadata"),
                "version": kwargs.get("version"),
                "model": kwargs.get("model"),
                "model_parameters": kwargs.get("model_parameters"),
            }
        )
        self.child_observations.append(observation)
        return observation

    def flush(self) -> None:
        self.flush_calls += 1


def build_client(*, fail_on_update: bool = False) -> tuple[LangfuseClient, FakeSdkClient, list[dict[str, object]]]:
    sdk_client = FakeSdkClient(fail_on_update=fail_on_update)
    attribute_calls: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> FakeSdkClient:
        return sdk_client

    def attribute_context_factory(**kwargs: object):
        attribute_calls.append(dict(kwargs))
        return nullcontext()

    client = LangfuseClient(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="https://cloud.langfuse.com",
        environment="production",
        release="modal-v1",
        sample_rate=1.0,
        client_factory=client_factory,
        attribute_context_factory=attribute_context_factory,
    )
    return client, sdk_client, attribute_calls


def build_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id="chunk-1",
        source="projects.md",
        section="Portfolio Chatbot",
        content=content,
        similarity=0.91,
        metadata={
            "chunk_id": "chunk-1",
            "source": "projects.md",
            "section": "Portfolio Chatbot",
        },
    )


def test_noop_tracer_methods_are_safe() -> None:
    tracer = NoOpTracer()
    trace = tracer.start_chat_request(
        question="Hello",
        conversation_id="conversation-1",
        session_id="session-1",
        user_id="user-1",
        endpoint="/chat",
        endpoint_name="chat",
        channel="web_chat",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
    )

    retrieval_observation = tracer.start_retrieval(
        trace,
        original_query="Hello",
        rewritten_query=None,
        retriever_type="vector",
        top_k=5,
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        vector_store="pgvector",
    )
    tracer.complete_retrieval(
        trace,
        observation=retrieval_observation,
        retrieved_chunks=[],
        latency_ms=12,
    )
    llm_observation = tracer.start_llm_call(
        trace,
        provider="openai",
        model="gpt-4.1-mini",
        temperature=None,
        max_tokens=None,
    )
    tracer.complete_llm_call(
        trace,
        observation=llm_observation,
        provider="openai",
        model="gpt-4.1-mini",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    tracer.complete_chat_request(
        trace,
        final_answer="Hi",
        conversation_id="conversation-1",
        latency_ms=150,
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    tracer.capture_error(trace, error_message="boom", latency_ms=30)
    tracer.flush()


def test_get_tracer_returns_noop_when_disabled() -> None:
    tracer = get_tracer(build_settings(enable_langfuse_observability=False))

    assert isinstance(tracer, NoOpTracer)


def test_langfuse_tracer_records_expected_payload_shapes() -> None:
    client, sdk_client, attribute_calls = build_client()
    tracer = get_tracer(build_settings(), client=client)
    long_content = "A" * (CONTENT_PREVIEW_LIMIT + 50)

    trace = tracer.start_chat_request(
        question="Tell me about the chatbot",
        conversation_id="conversation-1",
        session_id="session-1",
        user_id="user-1",
        endpoint="/chat",
        endpoint_name="chat",
        channel="web_chat",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
    )
    retrieval_observation = tracer.start_retrieval(
        trace,
        original_query="Tell me about the chatbot",
        rewritten_query=None,
        retriever_type="vector",
        top_k=5,
        embedding_provider="hf",
        embedding_model="all-MiniLM-L6-v2",
        vector_store="pgvector",
    )
    tracer.complete_retrieval(
        trace,
        observation=retrieval_observation,
        retrieved_chunks=[build_chunk(long_content)],
        latency_ms=24,
    )
    llm_observation = tracer.start_llm_call(
        trace,
        provider="openai",
        model="gpt-4.1-mini",
        temperature=0.1,
        max_tokens=256,
    )
    tracer.complete_llm_call(
        trace,
        observation=llm_observation,
        provider="openai",
        model="gpt-4.1-mini",
        latency_ms=81,
        input_tokens=120,
        output_tokens=32,
        total_tokens=152,
    )
    tracer.complete_chat_request(
        trace,
        final_answer="Tumelo built a production-ready chatbot.",
        conversation_id="conversation-1",
        latency_ms=125,
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        input_tokens=120,
        output_tokens=32,
        total_tokens=152,
    )
    tracer.flush()

    assert attribute_calls == [
        {
            "user_id": "user-1",
            "session_id": "session-1",
            "metadata": {
                "endpoint": "/chat",
                "endpoint_name": "chat",
                "channel": "web_chat",
                "llm_provider": "openai",
                "llm_model": "gpt-4.1-mini",
            },
            "version": "modal-v1",
            "trace_name": "chat_request",
            "environment": "production",
        }
    ]
    assert sdk_client.flush_calls == 1
    assert len(sdk_client.root_contexts) == 1
    root_observation = sdk_client.root_contexts[0].observation
    assert root_observation.name == "chat_request"
    assert root_observation.updates[0]["input"] == {
        "question": "Tell me about the chatbot",
        "conversation_id": "conversation-1",
    }
    assert root_observation.updates[0]["metadata"] == {
        "endpoint": "/chat",
        "endpoint_name": "chat",
        "channel": "web_chat",
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
    }
    assert root_observation.updates[0]["version"] == "modal-v1"
    assert root_observation.updates[-1]["output"]["final_answer"] == (
        "Tumelo built a production-ready chatbot."
    )
    assert sdk_client.root_contexts[0].exited is True

    retrieval_observation = sdk_client.child_observations[0]
    assert retrieval_observation.name == "retrieval"
    assert retrieval_observation.as_type == "retriever"
    assert retrieval_observation.updates[0]["metadata"] == {
        "embedding_provider": "hf",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store": "pgvector",
    }
    retrieval_output = retrieval_observation.updates[1]["output"]
    assert retrieval_output["retrieved_sources"] == ["projects.md"]
    assert retrieval_output["chunk_ids"] == ["chunk-1"]
    assert retrieval_output["results"][0]["score"] == 0.91
    assert len(retrieval_output["results"][0]["content_preview"]) == CONTENT_PREVIEW_LIMIT

    llm_observation = sdk_client.child_observations[1]
    assert llm_observation.name == "llm_call"
    assert llm_observation.as_type == "generation"
    assert llm_observation.updates[0]["model"] == "gpt-4.1-mini"
    assert llm_observation.updates[0]["model_parameters"] == {
        "temperature": 0.1,
        "max_tokens": 256,
    }
    assert llm_observation.updates[1]["output"] == {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "latency_ms": 81,
        "error": None,
    }
    assert llm_observation.updates[1]["usage_details"] == {
        "input": 120,
        "output": 32,
        "total": 152,
    }
def test_langfuse_tracer_honors_sample_rate() -> None:
    client, sdk_client, _ = build_client()
    tracer = LangfuseTracer(
        client=client,
        environment="production",
        release="modal-v1",
        sample_rate=0.5,
        random_value_factory=lambda: 0.9,
    )

    trace = tracer.start_chat_request(
        question="Hello",
        conversation_id="conversation-1",
        session_id="session-1",
        user_id="user-1",
        endpoint="/chat",
        endpoint_name="chat",
        channel="web_chat",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
    )

    assert trace.is_active is False
    assert sdk_client.root_contexts == []
