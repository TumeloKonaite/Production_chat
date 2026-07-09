from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.infrastructure.tracking import create_experiment_tracker
from app.infrastructure.tracking.mlflow_client import MLflowClient


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name


def build_test_settings(**overrides: object) -> Settings:
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
        "default_prompt_version": "v1_professional",
        "conversation_history_limit": 10,
        "retriever_type": "vector",
        "retrieval_top_k": 5,
        "retrieval_min_similarity": 0.55,
        "default_retrieval_config": "default",
        "enable_mlflow_tracking": True,
        "mlflow_tracking_uri": "http://localhost:5000",
        "mlflow_experiment_name": "personal-chatbot-model-comparison",
        "enable_dagshub_tracking": False,
        "dagshub_repo_owner": None,
        "dagshub_repo_name": None,
        "dagshub_token": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_mlflow_client_uses_local_tracking_uri_when_dagshub_is_disabled() -> None:
    fake_mlflow = FakeMLflow()
    client = MLflowClient(
        tracking_uri="http://localhost:5000",
        tracking_username=None,
        tracking_password=None,
        enabled=True,
    )
    client._mlflow = fake_mlflow

    configured = client.set_experiment("local-evals")

    assert configured is True
    assert fake_mlflow.tracking_uri == "http://localhost:5000"
    assert fake_mlflow.experiment_name == "local-evals"


def test_mlflow_client_initializes_dagshub_before_setting_experiment(monkeypatch) -> None:
    fake_mlflow = FakeMLflow()
    dagshub_calls: list[dict[str, object]] = []

    def fake_init(**kwargs: object) -> None:
        dagshub_calls.append(kwargs)

    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "dagshub", SimpleNamespace(init=fake_init))

    client = MLflowClient(
        tracking_uri="http://localhost:5000",
        tracking_username=None,
        tracking_password=None,
        enabled=True,
        enable_dagshub_tracking=True,
        dagshub_repo_owner="acme",
        dagshub_repo_name="production-chatbot",
        dagshub_token="dagshub-secret",
    )
    client._mlflow = fake_mlflow

    configured = client.set_experiment("remote-evals")

    assert configured is True
    assert dagshub_calls == [
        {
            "repo_owner": "acme",
            "repo_name": "production-chatbot",
            "mlflow": True,
        }
    ]
    assert fake_mlflow.tracking_uri is None
    assert fake_mlflow.experiment_name == "remote-evals"
    assert os.environ["DAGSHUB_USER_TOKEN"] == "dagshub-secret"


def test_mlflow_client_disables_tracking_when_dagshub_repo_owner_is_missing() -> None:
    fake_mlflow = FakeMLflow()
    client = MLflowClient(
        tracking_uri=None,
        tracking_username=None,
        tracking_password=None,
        enabled=True,
        enable_dagshub_tracking=True,
        dagshub_repo_owner=None,
        dagshub_repo_name="production-chatbot",
    )
    client._mlflow = fake_mlflow

    configured = client.set_experiment("remote-evals")

    assert configured is False
    assert client.enabled is False


def test_create_experiment_tracker_disables_invalid_dagshub_without_mlflow() -> None:
    settings = build_test_settings(
        enable_mlflow_tracking=False,
        enable_dagshub_tracking=True,
        dagshub_repo_owner="acme",
        dagshub_repo_name="production-chatbot",
    )

    tracker = create_experiment_tracker(settings, "remote-evals")

    assert tracker.enabled is False


def test_mlflow_client_disables_tracking_when_dagshub_init_fails(monkeypatch) -> None:
    fake_mlflow = FakeMLflow()

    def fail_init(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "dagshub", SimpleNamespace(init=fail_init))

    client = MLflowClient(
        tracking_uri=None,
        tracking_username=None,
        tracking_password=None,
        enabled=True,
        enable_dagshub_tracking=True,
        dagshub_repo_owner="acme",
        dagshub_repo_name="production-chatbot",
    )
    client._mlflow = fake_mlflow

    configured = client.set_experiment("remote-evals")

    assert configured is False
    assert client.enabled is False
