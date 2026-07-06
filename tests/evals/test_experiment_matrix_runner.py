from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.services.evals.generation_eval_service import (
    GenerationEvalAggregate,
    GenerationEvalRecord,
)
from evals.matrix.config_loader import load_experiment_matrix_config
from evals.matrix.models import ExperimentMatrixConfig, ExperimentSuiteConfig
from evals.matrix.runner import run_experiment_matrix
from evals.retrieval_eval_runner import RetrievalDatasetValidationSummary, RetrievalEvalRunResult
from evals.run_generation_eval import GenerationEvalRunResult
import evals.run_experiment_matrix as experiment_matrix_cli


async def _fake_generation_runner(**kwargs) -> GenerationEvalRunResult:
    output_dir = kwargs["output_dir"]
    run_name = kwargs["run_name"]
    prompt_version = kwargs["prompt_version"]
    if prompt_version == "v2_warm_conversational":
        raise TimeoutError("Synthetic failure")

    aggregate = GenerationEvalAggregate(
        model_config_id="openai:gpt-4.1-mini",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        model_base_url="https://api.openai.com/v1",
        total_examples=1,
        passed_examples=1,
        failed_examples=0,
        pass_rate=1.0,
        average_quality_score=5.0,
        average_groundedness_score=4.0,
        average_context_relevance=2.0,
        average_faithfulness=2.0,
        average_answer_relevance=2.0,
        latency_ms_avg=120.0,
        latency_ms_p50=120.0,
        latency_ms_p95=120.0,
        total_prompt_tokens=200,
        total_completion_tokens=100,
        total_tokens=300,
        avg_tokens_per_response=300.0,
        responses_with_usage=1,
        estimated_prompt_cost_usd=0.001,
        estimated_completion_cost_usd=0.002,
        estimated_total_cost_usd=0.003,
        average_cost_per_response_usd=0.003,
        responses_with_cost_estimate=1,
    )
    record = GenerationEvalRecord(
        eval_id="q1",
        model_config_id="openai:gpt-4.1-mini",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        model_base_url="https://api.openai.com/v1",
        question="What does Tumelo do?",
        category="profile",
        expected_facts=["engineer"],
        expected_answer_points=["engineer"],
        expected_behavior=None,
        generated_answer="Tumelo is an engineer.",
        latency_ms=120,
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        estimated_prompt_cost_usd=0.001,
        estimated_completion_cost_usd=0.002,
        estimated_cost_usd=0.003,
        quality_score=5,
        groundedness_score=4.0,
        passed=True,
        used_fallback=False,
        fixed_context_sources=["profile.md"],
        judge_evaluation=None,
    )
    return GenerationEvalRunResult(
        run_name=run_name,
        dataset_path=kwargs["dataset_path"],
        dataset_version=kwargs["dataset_version"],
        prompt_version=prompt_version,
        model_config_id="openai:gpt-4.1-mini",
        judge_model_config_id=kwargs["judge_model_config_id"],
        temperature=float(kwargs["temperature"]),
        model_base_url="https://api.openai.com/v1",
        aggregate=aggregate,
        records=[record],
        artifact_paths={
            "results_json": output_dir / f"{run_name}.json",
            "summary_txt": output_dir / f"{run_name}.txt",
            "config_json": output_dir / f"{run_name}_config.json",
        },
        pricing_lookup_note=None,
    )


def test_run_experiment_matrix_writes_generation_outputs_and_failures(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  generation_smoke:",
                "    mode: generation",
                "    max_combinations: 4",
                "    generation:",
                "      llm_model:",
                "        - gpt-4.1-mini",
                "      prompt_version:",
                "        - v1_professional",
                "        - v2_warm_conversational",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_config = load_experiment_matrix_config(config_path)

    result = run_experiment_matrix(
        matrix_config=matrix_config,
        suite_name="generation_smoke",
        settings=_build_settings(),
        argv=["evals/run_experiment_matrix.py", "--suite", "generation_smoke"],
        output_dir=tmp_path / "outputs",
        generation_dataset_path=tmp_path / "generation.jsonl",
        dry_run=False,
        generation_runner=_fake_generation_runner,
    )

    assert result.status == "completed_with_failures"
    assert len(result.successful_rows) == 1
    assert len(result.failures) == 1
    assert result.summary_paths["summary_json"].exists()
    assert result.summary_paths["summary_csv"].exists()
    assert result.manifest_path.exists()
    assert result.failures_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["successful_runs"] == 1
    assert manifest["failed_runs"] == 1

    failures = json.loads(result.failures_path.read_text(encoding="utf-8"))
    assert failures[0]["run_id"] == "run_002"
    assert failures[0]["error_type"] == "TimeoutError"


def test_run_experiment_matrix_enforces_max_combinations(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  rag_too_large:",
                "    mode: rag",
                "    max_combinations: 3",
                "    retrieval:",
                "      retriever_type:",
                "        - vector",
                "        - hybrid",
                "      top_k:",
                "        - 3",
                "        - 5",
                "    generation:",
                "      llm_model:",
                "        - gpt-4.1-mini",
                "      prompt_version:",
                "        - v1_professional",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_config = load_experiment_matrix_config(config_path)

    try:
        run_experiment_matrix(
            matrix_config=matrix_config,
            suite_name="rag_too_large",
            settings=_build_settings(),
            argv=["evals/run_experiment_matrix.py", "--suite", "rag_too_large"],
            output_dir=tmp_path / "outputs",
        )
    except ValueError as exc:
        assert "Refusing to run because planned combinations exceed max_combinations." in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected max combination safety check to fail.")


def test_run_experiment_matrix_allows_dry_run_for_full_suite(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  rag_full:",
                "    mode: rag",
                "    max_combinations: 8",
                "    retrieval:",
                "      retriever_type:",
                "        - vector",
                "      top_k:",
                "        - 5",
                "    generation:",
                "      llm_model:",
                "        - gpt-4.1-mini",
                "      prompt_version:",
                "        - v1_professional",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_config = load_experiment_matrix_config(config_path)

    plan = run_experiment_matrix(
        matrix_config=matrix_config,
        suite_name="rag_full",
        settings=_build_settings(),
        argv=["evals/run_experiment_matrix.py", "--suite", "rag_full", "--dry-run"],
        output_dir=tmp_path / "outputs",
        dry_run=True,
    )

    assert plan.total_planned_runs == 1


def test_run_experiment_matrix_requires_confirmation_for_full_suite(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_matrix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suites:",
                "  rag_full:",
                "    mode: rag",
                "    max_combinations: 8",
                "    retrieval:",
                "      retriever_type:",
                "        - vector",
                "      top_k:",
                "        - 5",
                "    generation:",
                "      llm_model:",
                "        - gpt-4.1-mini",
                "      prompt_version:",
                "        - v1_professional",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_config = load_experiment_matrix_config(config_path)

    try:
        run_experiment_matrix(
            matrix_config=matrix_config,
            suite_name="rag_full",
            settings=_build_settings(),
            argv=["evals/run_experiment_matrix.py", "--suite", "rag_full"],
            output_dir=tmp_path / "outputs",
        )
    except ValueError as exc:
        assert str(exc) == "Suite rag_full requires --confirm-full-run before execution."
    else:  # pragma: no cover
        raise AssertionError("Expected full suite confirmation check to fail.")


def test_experiment_matrix_cli_prints_plan_before_running(monkeypatch) -> None:
    matrix_config = ExperimentMatrixConfig(
        suites={
            "generation_smoke": ExperimentSuiteConfig(
                name="generation_smoke",
                mode="generation",
                description="Quick generation smoke test.",
                max_combinations=4,
                retrieval={},
                generation={
                    "llm_model": ("gpt-4.1-mini",),
                    "prompt_version": ("v1_professional",),
                },
            )
        },
        source_path=Path("evals/configs/experiment_matrix.yaml"),
    )
    events: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        del kwargs
        events.append(" ".join(str(arg) for arg in args))

    def fake_run_experiment_matrix(**kwargs):
        del kwargs
        events.append("RUNNER_CALLED")
        return SimpleNamespace(
            matrix_run_id="2026-07-04_20-30-00_generation_smoke",
            status="completed",
            summary_paths={
                "summary_json": Path("summary.json"),
                "summary_csv": Path("summary.csv"),
            },
            failures_path=Path("failures.json"),
            manifest_path=Path("manifest.json"),
        )

    monkeypatch.setattr(
        experiment_matrix_cli,
        "parse_args",
        lambda: SimpleNamespace(
            config=Path("evals/configs/experiment_matrix.yaml"),
            suite="generation_smoke",
            output_dir=Path("evals/outputs/experiments"),
            retrieval_dataset=Path("evals/datasets/portfolio_eval_dataset.jsonl"),
            generation_dataset=Path("evals/datasets/generation_eval_dataset.jsonl"),
            rag_dataset=Path("evals/datasets/portfolio_eval_dataset.jsonl"),
            rag_judge_prompt=Path("evals/prompts/judge_prompt_v1.md"),
            generation_judge_model=None,
            rag_judge_model=None,
            dry_run=False,
            confirm_full_run=False,
            min_expected_source_coverage=0.8,
            persist_rag_results=False,
        ),
    )
    monkeypatch.setattr(experiment_matrix_cli, "load_experiment_matrix_config", lambda path: matrix_config)
    monkeypatch.setattr(experiment_matrix_cli, "get_settings", _build_settings)
    monkeypatch.setattr(experiment_matrix_cli, "run_experiment_matrix", fake_run_experiment_matrix)
    monkeypatch.setattr(experiment_matrix_cli, "print", fake_print, raising=False)

    experiment_matrix_cli.main()

    assert events[0].startswith("Suite: generation_smoke")
    assert "Generation combinations: 1" in events[0]
    assert events[2] == "RUNNER_CALLED"
    assert events[0:3] == [events[0], "", "RUNNER_CALLED"]


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
