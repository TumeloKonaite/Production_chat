from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies.common_dependencies import get_app_settings
from app.api.evals.routes import get_eval_job_runner, get_eval_run_service
from app.config import Settings
from app.main import app
from app.services.evals.eval_run_service import EvalRunService


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
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="test-key",
        knowledge_chunk_size=500,
        knowledge_chunk_overlap=100,
        enable_query_rewriting=False,
        query_rewrite_model="openai:gpt-4.1-mini",
        query_rewrite_temperature=0.0,
        query_rewrite_prompt_version="v1",
        query_rewrite_timeout_seconds=10,
        query_rewrite_max_tokens=128,
        enable_reranking=False,
        reranker_type="none",
        reranker_model="openai:gpt-4.1-mini",
        reranker_initial_top_k=20,
        reranker_final_top_k=5,
    )


class FakeEvalJobRunner:
    def __init__(self, run_service: EvalRunService) -> None:
        self.run_service = run_service
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def resolve_matrix_plan(
        self,
        *,
        suite_name: str,
        dry_run: bool,
        confirm_full_run: bool,
    ) -> SimpleNamespace:
        if suite_name == "rag_full" and not dry_run and not confirm_full_run:
            raise ValueError("Suite rag_full requires confirm_full_run=true.")
        if suite_name == "too_large":
            raise ValueError("Refusing to run because planned combinations exceed max_combinations.")
        return SimpleNamespace(
            suite=SimpleNamespace(
                name=suite_name,
                mode="rag",
                max_combinations=32,
            ),
            retrieval_combinations=[{"top_k": 3}, {"top_k": 5}],
            generation_combinations=[{"prompt_version": "v1"}, {"prompt_version": "v2"}],
            total_planned_runs=4,
            requires_confirmation=suite_name == "rag_full",
        )

    def run_retrieval_job(self, *, run_id: str, payload: dict[str, object], settings: Settings) -> None:
        del settings
        self.calls.append(("retrieval", run_id, payload))
        self.run_service.update_run(
            run_id,
            status="completed",
            summary_payload={
                "run_id": run_id,
                "mode": "retrieval",
                "summary": {"hit_at_k": 0.75, "mrr": 0.5},
            },
            failures_payload={"run_id": run_id, "failures": []},
            artifacts={"results_json": "runner_output/results.json"},
        )

    def run_generation_job(self, *, run_id: str, payload: dict[str, object], settings: Settings) -> None:
        del settings
        self.calls.append(("generation", run_id, payload))
        self.run_service.update_run(
            run_id,
            status="completed",
            summary_payload={
                "run_id": run_id,
                "mode": "generation",
                "summary": {"pass_rate": 1.0, "average_quality_score": 4.5},
            },
            failures_payload={"run_id": run_id, "failures": []},
            artifacts={"results_json": "runner_output/generation.json"},
        )

    def run_rag_job(self, *, run_id: str, payload: dict[str, object], settings: Settings) -> None:
        del settings
        self.calls.append(("rag", run_id, payload))
        self.run_service.update_run(
            run_id,
            status="completed",
            summary_payload={
                "run_id": run_id,
                "mode": "rag",
                "summary": {"avg_answer_relevance": 0.91, "avg_faithfulness": 0.89},
            },
            failures_payload={"run_id": run_id, "failures": []},
            artifacts={"results_json": "runner_output/rag.json"},
        )

    def run_matrix_job(self, *, run_id: str, payload: dict[str, object], settings: Settings) -> None:
        del settings
        self.calls.append(("matrix", run_id, payload))
        self.run_service.update_run(
            run_id,
            status="completed_with_failures",
            total_planned_runs=4,
            successful_runs=3,
            failed_runs=1,
            summary_payload={
                "run_id": run_id,
                "suite": payload["suite"],
                "mode": "rag",
                "top_results": [
                    {
                        "rank": 1,
                        "avg_answer_relevance": 0.91,
                        "avg_faithfulness": 0.89,
                    }
                ],
            },
            failures_payload={
                "run_id": run_id,
                "failures": [
                    {
                        "run_id": "run_003",
                        "mode": "rag",
                        "error_type": "TimeoutError",
                        "error_message": "Request timed out after 60 seconds",
                        "config": {"prompt_version": "v2"},
                    }
                ],
            },
            artifacts={"summary_json": "runner_output/rag_summary.json"},
        )


@pytest.fixture
def run_service(tmp_path: Path) -> EvalRunService:
    return EvalRunService(base_output_dir=tmp_path / "eval-runs")


@pytest.fixture
def fake_job_runner(run_service: EvalRunService) -> FakeEvalJobRunner:
    return FakeEvalJobRunner(run_service)


@pytest.fixture
def client(run_service: EvalRunService, fake_job_runner: FakeEvalJobRunner) -> TestClient:
    app.dependency_overrides[get_app_settings] = build_test_settings
    app.dependency_overrides[get_eval_run_service] = lambda: run_service
    app.dependency_overrides[get_eval_job_runner] = lambda: fake_job_runner
    return TestClient(app)


def test_eval_run_rejects_missing_token(client: TestClient) -> None:
    response = client.post("/api/evals/retrieval", json={"retriever_type": "vector", "top_k": 5})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid eval admin token."}


def test_retrieval_eval_run_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post(
        "/api/evals/retrieval",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"retriever_type": "vector", "top_k": 0},
    )

    assert response.status_code == 422


def test_retrieval_eval_run_queues_job_and_exposes_status(
    client: TestClient,
    fake_job_runner: FakeEvalJobRunner,
) -> None:
    response = client.post(
        "/api/evals/retrieval",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "retriever_type": "vector",
            "top_k": 5,
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "query_rewriting_enabled": False,
            "reranking_enabled": False,
        },
    )

    assert response.status_code == 202
    payload = response.json()
    run_id = payload["run_id"]
    assert payload == {
        "run_id": run_id,
        "status": "queued",
        "mode": "retrieval",
        "suite": None,
        "status_url": f"/api/evals/runs/{run_id}",
    }
    assert fake_job_runner.calls == [
        (
            "retrieval",
            run_id,
            {
                "embedding_provider": None,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": None,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "retriever_type": "vector",
                "top_k": 5,
                "query_rewriting_enabled": False,
                "query_rewrite_model": None,
                "query_rewrite_prompt_version": None,
                "query_rewrite_temperature": None,
                "reranking_enabled": False,
                "reranker_type": None,
                "reranker_model": None,
                "reranker_initial_top_k": None,
                "notes": None,
            },
        )
    ]

    status_response = client.get(
        f"/api/evals/runs/{run_id}",
        headers={"x-eval-admin-token": "eval-secret"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"


def test_generation_eval_run_queues_job(client: TestClient, fake_job_runner: FakeEvalJobRunner) -> None:
    response = client.post(
        "/api/evals/generation",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "provider": "openai",
            "llm_model": "gpt-4.1-mini",
            "prompt_version": "v2_strict_grounding",
            "temperature": 0.0,
            "max_tokens": 800,
        },
    )

    assert response.status_code == 202
    assert fake_job_runner.calls[0][0] == "generation"


def test_rag_eval_run_queues_job(client: TestClient, fake_job_runner: FakeEvalJobRunner) -> None:
    response = client.post(
        "/api/evals/rag",
        headers={"x-eval-admin-token": "eval-secret"},
        json={
            "retrieval": {
                "retriever_type": "vector",
                "top_k": 5,
                "embedding_model": "text-embedding-3-small",
                "chunk_size": 500,
                "chunk_overlap": 100,
            },
            "generation": {
                "provider": "openai",
                "llm_model": "gpt-4.1-mini",
                "prompt_version": "v2_strict_grounding",
                "temperature": 0.0,
            },
        },
    )

    assert response.status_code == 202
    assert fake_job_runner.calls[0][0] == "rag"


def test_matrix_dry_run_returns_plan(client: TestClient) -> None:
    response = client.post(
        "/api/evals/matrix",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"suite": "rag_medium", "dry_run": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "suite": "rag_medium",
        "mode": "rag",
        "dry_run": True,
        "retrieval_combinations": 2,
        "generation_combinations": 2,
        "total_planned_runs": 4,
        "max_combinations": 32,
        "status": "ok",
    }


def test_matrix_run_requires_confirmation_for_full_suite(client: TestClient) -> None:
    response = client.post(
        "/api/evals/matrix",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"suite": "rag_full", "dry_run": False, "confirm_full_run": False},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Suite rag_full requires confirm_full_run=true."}


def test_matrix_run_failure_details_are_available(
    client: TestClient,
    fake_job_runner: FakeEvalJobRunner,
) -> None:
    response = client.post(
        "/api/evals/matrix",
        headers={"x-eval-admin-token": "eval-secret"},
        json={"suite": "rag_medium", "dry_run": False, "confirm_full_run": True},
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert fake_job_runner.calls[0][0] == "matrix"

    status_response = client.get(
        f"/api/evals/runs/{run_id}",
        headers={"x-eval-admin-token": "eval-secret"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed_with_failures"
    assert status_response.json()["failed_runs"] == 1

    summary_response = client.get(
        f"/api/evals/runs/{run_id}/summary",
        headers={"x-eval-admin-token": "eval-secret"},
    )
    assert summary_response.status_code == 200
    assert summary_response.json()["top_results"][0]["rank"] == 1

    failures_response = client.get(
        f"/api/evals/runs/{run_id}/failures",
        headers={"x-eval-admin-token": "eval-secret"},
    )
    assert failures_response.status_code == 200
    assert failures_response.json() == {
        "run_id": run_id,
        "failures": [
            {
                "run_id": "run_003",
                "mode": "rag",
                "error_type": "TimeoutError",
                "error_message": "Request timed out after 60 seconds",
                "config": {"prompt_version": "v2"},
            }
        ],
    }


def test_list_eval_runs_supports_mode_filter(client: TestClient, run_service: EvalRunService) -> None:
    run_service.create_run(
        mode="retrieval",
        config={"top_k": 5},
        run_id="2026-07-04_15-20-11_retrieval",
    )
    run_service.create_run(
        mode="matrix",
        suite="rag_medium",
        config={"suite": "rag_medium"},
        run_id="2026-07-04_15-30-00_rag_medium",
    )

    response = client.get(
        "/api/evals/runs?mode=matrix&limit=20",
        headers={"x-eval-admin-token": "eval-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "2026-07-04_15-30-00_rag_medium",
                "mode": "matrix",
                "status": "queued",
                "suite": "rag_medium",
                "started_at": None,
                "completed_at": None,
                "total_planned_runs": None,
                "successful_runs": None,
                "failed_runs": None,
            }
        ]
    }


def test_eval_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/evals/retrieval" in paths
    assert "/api/evals/generation" in paths
    assert "/api/evals/rag" in paths
    assert "/api/evals/matrix" in paths
    assert "/api/evals/runs" in paths
    assert "/api/evals/runs/{run_id}" in paths
    assert "/api/evals/runs/{run_id}/summary" in paths
    assert "/api/evals/runs/{run_id}/failures" in paths
