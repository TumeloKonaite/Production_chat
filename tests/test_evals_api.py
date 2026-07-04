from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies.common_dependencies import get_app_settings
from app.api.evals.routes import (
    get_experiment_tracker_factory,
    get_retrieval_eval_runner,
    get_retrieval_sweep_runner,
)
from app.config import Settings
from app.main import app
from evals.retrieval_eval_runner import (
    RetrievalDatasetValidationSummary,
    RetrievalEvalRunResult,
)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def build_test_settings() -> Settings:
    return Settings(
        database_url="sqlite:///unused-for-tests.db",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        tavus_api_key="tavus-test-key",
        tavus_base_url="https://tavus.example",
        tavus_face_id="face_123",
        tavus_pal_id="pal_123",
        public_backend_url="https://backend.example",
        tavus_tool_secret="tool-secret",
        ingestion_api_secret="ingestion-secret",
        eval_admin_token="eval-secret",
        default_model_config_id="openai:gpt-4.1-mini",
        model_configs_json=None,
        embedding_provider="hf",
        knowledge_embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        knowledge_collection_name="personal_knowledge_base",
        default_prompt_version="v1_professional",
        conversation_history_limit=10,
        retriever_type="vector",
        retrieval_top_k=5,
        retrieval_min_similarity=0.55,
        default_retrieval_config="default",
        enable_mlflow_tracking=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="personal-chatbot-model-comparison",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
        enable_query_rewriting=False,
        query_rewrite_model="openai:gpt-4.1-mini",
        query_rewrite_temperature=0.0,
        query_rewrite_prompt_version="v1",
        query_rewrite_timeout_seconds=10,
        query_rewrite_max_tokens=128,
    )


class FakeTrackerFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[Settings, str]] = []

    def __call__(self, settings: Settings, experiment_name: str) -> object:
        self.calls.append((settings, experiment_name))
        return object()


class FakeRetrievalEvalRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> RetrievalEvalRunResult:
        self.calls.append(kwargs)
        output_dir = Path("evals/results/retrieval/api-vector-k5-baseline")
        return RetrievalEvalRunResult(
            run_name="api-vector-k5-baseline",
            mlflow_run_id="9355f8c9952f4829a7950b59f54e27ad",
            output_dir=output_dir,
            summary={
                "hit_at_k": 0.4166666666666667,
                "mrr": 0.3958333333333333,
                "recall_at_k": 0.3159722222222222,
                "mean_precision_at_k": 0.2638888888888889,
                "num_queries_total": 25,
                "num_queries_evaluated": 24,
                "num_queries_without_expected_source": 1,
                "query_rewrite_avg_latency_ms": 0.0,
                "query_rewrite_total_latency_ms": 0,
                "query_rewrite_success_count": 0,
                "query_rewrite_fallback_count": 0,
                "query_rewrite_failure_count": 0,
                "query_rewrite_total_tokens": 0,
                "query_rewrite_estimated_total_cost": 0.0,
            },
            results=[],
            config={
                "retriever_type": "vector",
                "top_k": 5,
                "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
                "embedding_provider": "hf",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "query_rewriting_enabled": False,
                "query_rewrite_model": "openai:gpt-4.1-mini",
                "query_rewrite_prompt_version": "v1",
                "notes": "Triggered from protected backend API",
            },
            artifact_paths={
                "results_json": output_dir / "results.json",
                "results_csv": output_dir / "results.csv",
                "config_json": output_dir / "config.json",
            },
            validation_summary=RetrievalDatasetValidationSummary(
                total_queries=25,
                queries_with_expected_sources=24,
                queries_without_expected_sources=1,
                missing_expected_source_ids=["q25"],
            ),
        )


class FakeRetrievalSweepRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, Path]]:
        self.calls.append(kwargs)
        rows = [
            {
                "experiment_name": "retrieval-vector-k3",
                "run_name": "retrieval-vector-k3-2026-07-03_170000_run_01_retrieval-vector-k3",
                "retriever_type": "vector",
                "top_k": 3,
                "embedding_provider": "hf",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "query_rewriting_enabled": False,
                "query_rewrite_model": None,
                "query_rewrite_prompt_version": None,
                "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
                "git_commit_sha": "abc123",
                "mrr": 0.62,
                "recall_at_k": 0.71,
                "mean_precision_at_k": 0.45,
                "hit_at_k": 0.88,
                "num_queries_total": 25,
                "num_queries_evaluated": 24,
                "num_queries_without_expected_source": 1,
                "query_rewrite_avg_latency_ms": 0.0,
                "query_rewrite_total_latency_ms": 0,
                "query_rewrite_success_count": 0,
                "query_rewrite_fallback_count": 0,
                "query_rewrite_failure_count": 0,
                "query_rewrite_total_tokens": 0,
                "query_rewrite_estimated_total_cost": 0.0,
                "output_dir": "evals/results/retrieval_sweeps/run_01",
                "results_json": "evals/results/retrieval_sweeps/run_01/results.json",
                "results_csv": "evals/results/retrieval_sweeps/run_01/results.csv",
                "config_json": "evals/results/retrieval_sweeps/run_01/config.json",
            },
            {
                "experiment_name": "retrieval-keyword-k5",
                "run_name": "retrieval-keyword-k5-2026-07-03_170001_run_02_retrieval-keyword-k5",
                "retriever_type": "keyword",
                "top_k": 5,
                "embedding_provider": "hf",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "query_rewriting_enabled": False,
                "query_rewrite_model": None,
                "query_rewrite_prompt_version": None,
                "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
                "git_commit_sha": "abc123",
                "mrr": 0.54,
                "recall_at_k": 0.68,
                "mean_precision_at_k": 0.41,
                "hit_at_k": 0.82,
                "num_queries_total": 25,
                "num_queries_evaluated": 24,
                "num_queries_without_expected_source": 1,
                "query_rewrite_avg_latency_ms": 0.0,
                "query_rewrite_total_latency_ms": 0,
                "query_rewrite_success_count": 0,
                "query_rewrite_fallback_count": 0,
                "query_rewrite_failure_count": 0,
                "query_rewrite_total_tokens": 0,
                "query_rewrite_estimated_total_cost": 0.0,
                "output_dir": "evals/results/retrieval_sweeps/run_02",
                "results_json": "evals/results/retrieval_sweeps/run_02/results.json",
                "results_csv": "evals/results/retrieval_sweeps/run_02/results.csv",
                "config_json": "evals/results/retrieval_sweeps/run_02/config.json",
            },
        ]
        artifact_paths = {
            "comparison_json": Path("evals/results/retrieval_sweeps/comparison.json"),
            "comparison_csv": Path("evals/results/retrieval_sweeps/comparison.csv"),
            "manifest_json": Path("evals/results/retrieval_sweeps/manifest.json"),
        }
        return rows, artifact_paths


def test_retrieval_eval_run_rejects_missing_token() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings

    client = TestClient(app)
    response = client.post("/api/evals/retrieval-runs", json={"retriever_type": "vector", "top_k": 5})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid eval admin token."}


def test_retrieval_eval_run_rejects_invalid_token() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-runs",
        headers={"x-eval-admin-token": "wrong-secret"},
        json={"retriever_type": "vector", "top_k": 5},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid eval admin token."}


def test_retrieval_eval_run_rejects_invalid_top_k() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-runs",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"retriever_type": "vector", "top_k": 0},
    )

    assert response.status_code == 422


def test_retrieval_eval_run_rejects_unsupported_retriever_type() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-runs",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"retriever_type": "semantic", "top_k": 5},
    )

    assert response.status_code == 422


def test_retrieval_eval_run_calls_shared_runner_and_returns_result() -> None:
    fake_runner = FakeRetrievalEvalRunner()
    fake_tracker_factory = FakeTrackerFactory()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_retrieval_eval_runner] = lambda: fake_runner
    app.dependency_overrides[get_experiment_tracker_factory] = lambda: fake_tracker_factory

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-runs",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "retriever_type": "vector",
            "top_k": 5,
            "run_name": "api-vector-k5-baseline",
            "notes": "Triggered from protected backend API",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "run_name": "api-vector-k5-baseline",
        "mlflow_run_id": "9355f8c9952f4829a7950b59f54e27ad",
        "config": {
            "retriever_type": "vector",
            "top_k": 5,
            "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
            "embedding_provider": "hf",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "query_rewriting_enabled": False,
            "query_rewrite_model": "openai:gpt-4.1-mini",
            "query_rewrite_prompt_version": "v1",
            "notes": "Triggered from protected backend API",
        },
        "metrics": {
            "hit_at_k": 0.4166666666666667,
            "mrr": 0.3958333333333333,
            "recall_at_k": 0.3159722222222222,
            "mean_precision_at_k": 0.2638888888888889,
            "num_queries_total": 25,
            "num_queries_evaluated": 24,
            "num_queries_without_expected_source": 1,
            "query_rewrite_avg_latency_ms": 0.0,
            "query_rewrite_total_latency_ms": 0,
            "query_rewrite_success_count": 0,
            "query_rewrite_fallback_count": 0,
            "query_rewrite_failure_count": 0,
            "query_rewrite_total_tokens": 0,
            "query_rewrite_estimated_total_cost": 0.0,
        },
    }
    assert len(fake_tracker_factory.calls) == 1
    tracker_settings, experiment_name = fake_tracker_factory.calls[0]
    assert tracker_settings.retriever_type == "vector"
    assert tracker_settings.retrieval_top_k == 5
    assert experiment_name == "personal-chatbot-model-comparison"
    assert len(fake_runner.calls) == 1
    assert fake_runner.calls[0]["settings"] == tracker_settings
    assert fake_runner.calls[0]["top_k"] == 5
    assert fake_runner.calls[0]["run_name"] == "api-vector-k5-baseline"
    assert fake_runner.calls[0]["notes"] == "Triggered from protected backend API"
    assert fake_runner.calls[0]["argv"] == ["api:/api/evals/retrieval-runs"]


def test_retrieval_eval_run_allows_query_rewriting_override() -> None:
    fake_runner = FakeRetrievalEvalRunner()
    fake_tracker_factory = FakeTrackerFactory()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_retrieval_eval_runner] = lambda: fake_runner
    app.dependency_overrides[get_experiment_tracker_factory] = lambda: fake_tracker_factory

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-runs",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "retriever_type": "vector",
            "top_k": 5,
            "enable_query_rewriting": True,
        },
    )

    assert response.status_code == 200
    tracker_settings, _ = fake_tracker_factory.calls[0]
    assert tracker_settings.enable_query_rewriting is True
    assert fake_runner.calls[0]["settings"].enable_query_rewriting is True


def test_retrieval_eval_sweep_calls_shared_runner_and_returns_result() -> None:
    fake_runner = FakeRetrievalSweepRunner()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_retrieval_sweep_runner] = lambda: fake_runner

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-sweeps",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "experiments": [
                {"name": "retrieval-vector-k3", "retriever_type": "vector", "top_k": 3},
                {"name": "retrieval-keyword-k5", "retriever_type": "keyword", "top_k": 5},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "runs": [
            {
                "experiment_name": "retrieval-vector-k3",
                "run_name": "retrieval-vector-k3-2026-07-03_170000_run_01_retrieval-vector-k3",
                "config": {
                    "retriever_type": "vector",
                    "top_k": 3,
                    "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
                    "embedding_provider": "hf",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "query_rewriting_enabled": False,
                    "query_rewrite_model": None,
                    "query_rewrite_prompt_version": None,
                    "git_commit_sha": "abc123",
                },
                "metrics": {
                    "hit_at_k": 0.88,
                    "mrr": 0.62,
                    "recall_at_k": 0.71,
                    "mean_precision_at_k": 0.45,
                    "num_queries_total": 25,
                    "num_queries_evaluated": 24,
                    "num_queries_without_expected_source": 1,
                    "query_rewrite_avg_latency_ms": 0.0,
                    "query_rewrite_total_latency_ms": 0,
                    "query_rewrite_success_count": 0,
                    "query_rewrite_fallback_count": 0,
                    "query_rewrite_failure_count": 0,
                    "query_rewrite_total_tokens": 0,
                    "query_rewrite_estimated_total_cost": 0.0,
                },
                "artifacts": {
                    "output_dir": "evals/results/retrieval_sweeps/run_01",
                    "results_json": "evals/results/retrieval_sweeps/run_01/results.json",
                    "results_csv": "evals/results/retrieval_sweeps/run_01/results.csv",
                    "config_json": "evals/results/retrieval_sweeps/run_01/config.json",
                },
            },
            {
                "experiment_name": "retrieval-keyword-k5",
                "run_name": "retrieval-keyword-k5-2026-07-03_170001_run_02_retrieval-keyword-k5",
                "config": {
                    "retriever_type": "keyword",
                    "top_k": 5,
                    "dataset_path": "evals/datasets/portfolio_eval_dataset.jsonl",
                    "embedding_provider": "hf",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "query_rewriting_enabled": False,
                    "query_rewrite_model": None,
                    "query_rewrite_prompt_version": None,
                    "git_commit_sha": "abc123",
                },
                "metrics": {
                    "hit_at_k": 0.82,
                    "mrr": 0.54,
                    "recall_at_k": 0.68,
                    "mean_precision_at_k": 0.41,
                    "num_queries_total": 25,
                    "num_queries_evaluated": 24,
                    "num_queries_without_expected_source": 1,
                    "query_rewrite_avg_latency_ms": 0.0,
                    "query_rewrite_total_latency_ms": 0,
                    "query_rewrite_success_count": 0,
                    "query_rewrite_fallback_count": 0,
                    "query_rewrite_failure_count": 0,
                    "query_rewrite_total_tokens": 0,
                    "query_rewrite_estimated_total_cost": 0.0,
                },
                "artifacts": {
                    "output_dir": "evals/results/retrieval_sweeps/run_02",
                    "results_json": "evals/results/retrieval_sweeps/run_02/results.json",
                    "results_csv": "evals/results/retrieval_sweeps/run_02/results.csv",
                    "config_json": "evals/results/retrieval_sweeps/run_02/config.json",
                },
            },
        ],
        "artifacts": {
            "comparison_json": "evals\\results\\retrieval_sweeps\\comparison.json",
            "comparison_csv": "evals\\results\\retrieval_sweeps\\comparison.csv",
            "manifest_json": "evals\\results\\retrieval_sweeps\\manifest.json",
        },
    }
    assert len(fake_runner.calls) == 1
    sweep_config = fake_runner.calls[0]["sweep_config"]
    assert [experiment.name for experiment in sweep_config.experiments] == [
        "retrieval-vector-k3",
        "retrieval-keyword-k5",
    ]
    assert fake_runner.calls[0]["argv"] == ["api:/api/evals/retrieval-sweeps"]
    assert fake_runner.calls[0]["settings"].enable_query_rewriting is False


def test_retrieval_eval_sweep_allows_query_rewriting_override() -> None:
    fake_runner = FakeRetrievalSweepRunner()
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_retrieval_sweep_runner] = lambda: fake_runner

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-sweeps",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "enable_query_rewriting": True,
            "experiments": [
                {"name": "retrieval-vector-k3", "retriever_type": "vector", "top_k": 3},
            ],
        },
    )

    assert response.status_code == 200
    assert fake_runner.calls[0]["settings"].enable_query_rewriting is True


def test_retrieval_eval_sweep_rejects_duplicate_experiment_names() -> None:
    app.dependency_overrides[get_app_settings] = build_test_settings

    client = TestClient(app)
    response = client.post(
        "/api/evals/retrieval-sweeps",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "experiments": [
                {"name": "retrieval-vector-k3", "retriever_type": "vector", "top_k": 3},
                {"name": "retrieval-vector-k3", "retriever_type": "keyword", "top_k": 5},
            ]
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Retrieval sweep config contains duplicate experiment name: retrieval-vector-k3."
    }


def test_eval_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/evals/retrieval-runs" in paths
    assert "/api/evals/retrieval-sweeps" in paths
