from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.services.evals.eval_job_runner import EvalJobRunner
from app.services.evals.eval_run_service import EvalRunService
from evals.runners.retrieval_eval_runner import (
    RetrievalDatasetValidationSummary,
    RetrievalEvalRunResult,
)


def test_eval_job_runner_initializes_tracking_for_retrieval_jobs(tmp_path: Path) -> None:
    run_service = EvalRunService(base_output_dir=tmp_path / "runs")
    run_service.create_run(mode="retrieval", config={"top_k": 5}, run_id="run_001")
    factory_calls: list[tuple[Settings, str]] = []

    def fake_tracker_factory(settings: Settings, experiment_name: str):
        factory_calls.append((settings, experiment_name))
        return SimpleNamespace(enabled=False)

    def fake_retrieval_runner(**kwargs):
        output_dir = kwargs["output_root"] / "retrieval_eval"
        return RetrievalEvalRunResult(
            run_name=str(kwargs["run_name"]),
            mlflow_run_id=None,
            output_dir=output_dir,
            summary={"hit_at_k": 1.0, "mrr": 1.0},
            results=[],
            config={"top_k": kwargs["top_k"]},
            artifact_paths={"results_json": output_dir / "results.json"},
            validation_summary=RetrievalDatasetValidationSummary(
                total_queries=1,
                queries_with_expected_sources=1,
                queries_without_expected_sources=0,
                missing_expected_source_ids=[],
            ),
        )

    runner = EvalJobRunner(
        run_service=run_service,
        experiment_tracker_factory=fake_tracker_factory,
        retrieval_runner=fake_retrieval_runner,
    )
    settings = _build_settings()

    runner.run_retrieval_job(run_id="run_001", payload={"top_k": 5}, settings=settings)

    assert factory_calls == [(settings, settings.mlflow_experiment_name)]
    assert run_service.get_run("run_001").status == "completed"


def test_eval_job_runner_forwards_max_tokens_to_generation_jobs(tmp_path: Path) -> None:
    @dataclass(frozen=True, slots=True)
    class Aggregate:
        pass_rate: float
        average_quality_score: float

    run_service = EvalRunService(base_output_dir=tmp_path / "runs")
    run_service.create_run(mode="generation", config={"max_tokens": 800}, run_id="run_002")
    generation_calls: list[dict[str, object]] = []

    async def fake_generation_runner(**kwargs):
        generation_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        results_path = output_dir / "generation.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            aggregate=Aggregate(pass_rate=1.0, average_quality_score=4.5),
            model_config_id="openai:gpt-4.1-mini",
            judge_model_config_id=None,
            prompt_version="v1_professional",
            temperature=0.2,
            dataset_version="generation_eval_dataset_v1",
            max_tokens=800,
            artifact_paths={"results_json": results_path},
        )

    runner = EvalJobRunner(
        run_service=run_service,
        generation_runner=fake_generation_runner,
    )

    runner.run_generation_job(
        run_id="run_002",
        payload={"model_config_id": "openai:gpt-4.1-mini", "max_tokens": 800},
        settings=_build_settings(),
    )

    assert generation_calls[0]["max_tokens"] == 800
    summary_payload = json.loads(
        run_service.resolve_run_path("run_002", "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["config"]["max_tokens"] == 800


def test_eval_job_runner_forwards_max_tokens_to_rag_jobs(tmp_path: Path) -> None:
    @dataclass(frozen=True, slots=True)
    class Summary:
        avg_answer_relevance: float
        avg_faithfulness: float

    run_service = EvalRunService(base_output_dir=tmp_path / "runs")
    run_service.create_run(
        mode="rag",
        config={"retrieval": {"top_k": 5}, "generation": {"max_tokens": 700}},
        run_id="run_003",
    )
    rag_calls: list[dict[str, object]] = []

    async def fake_rag_runner(**kwargs):
        rag_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        results_path = output_dir / "rag.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            summary=Summary(avg_answer_relevance=0.91, avg_faithfulness=0.89),
            retrieval_config={"top_k": 5},
            model_config_id="openai:gpt-4.1-mini",
            judge_model_config_id=None,
            prompt_version="v1_professional",
            temperature=0.2,
            max_tokens=700,
            artifact_paths={"results_json": results_path},
        )

    runner = EvalJobRunner(
        run_service=run_service,
        rag_runner=fake_rag_runner,
    )

    runner.run_rag_job(
        run_id="run_003",
        payload={
            "retrieval": {"top_k": 5},
            "generation": {"model_config_id": "openai:gpt-4.1-mini", "max_tokens": 700},
        },
        settings=_build_settings(),
    )

    assert rag_calls[0]["max_tokens"] == 700
    summary_payload = json.loads(
        run_service.resolve_run_path("run_003", "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["generation"]["max_tokens"] == 700


def _build_settings() -> Settings:
    return Settings(
        database_url="postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        tavus_api_key=None,
        tavus_base_url="https://tavusapi.com",
        tavus_face_id=None,
        tavus_pal_id=None,
        public_backend_url=None,
        tavus_tool_secret=None,
        ingestion_api_secret=None,
        eval_admin_token=None,
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
        mlflow_experiment_name="test-experiment",
        enable_dagshub_tracking=False,
        dagshub_repo_owner=None,
        dagshub_repo_name=None,
        dagshub_token=None,
    )
