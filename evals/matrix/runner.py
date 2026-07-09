from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.infrastructure.tracking.conventions import (
    build_common_tracking_params,
    build_generation_tracking_metrics,
    build_retrieval_tracking_metrics,
    get_git_sha,
)
from app.infrastructure.tracking import create_experiment_tracker
from evals.matrix.expander import expand_suite_plan
from evals.matrix.models import (
    ExperimentMatrixConfig,
    ExperimentMatrixRunResult,
    MatrixRunSpec,
    ResolvedSuitePlan,
)
from evals.matrix.ranking import rank_mode_rows
from evals.runners.retrieval_eval_runner import (
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    RetrievalEvalRunResult,
)
from evals.runners.run_generation_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_GENERATION_DATASET_PATH,
    GenerationEvalRunResult,
    run_generation_eval,
)
from evals.runners.run_rag_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RAG_DATASET_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
    RagEvalRunResult,
    run_rag_eval,
)
from evals.runners.run_retrieval_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RETRIEVAL_DATASET_PATH,
)

DEFAULT_EXPERIMENT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "evals" / "outputs" / "experiments"

GenerationRunner = Callable[..., Awaitable[GenerationEvalRunResult]]
RagRunner = Callable[..., Awaitable[RagEvalRunResult]]
RetrievalRunner = Callable[..., RetrievalEvalRunResult]


def run_experiment_matrix(
    *,
    matrix_config: ExperimentMatrixConfig,
    suite_name: str,
    settings: Settings,
    argv: list[str],
    output_dir: Path = DEFAULT_EXPERIMENT_OUTPUT_DIR,
    retrieval_dataset_path: Path = DEFAULT_RETRIEVAL_DATASET_PATH,
    generation_dataset_path: Path = DEFAULT_GENERATION_DATASET_PATH,
    rag_dataset_path: Path = DEFAULT_RAG_DATASET_PATH,
    rag_judge_prompt_path: Path = DEFAULT_JUDGE_PROMPT_PATH,
    generation_judge_model_config_id: str | None = None,
    rag_judge_model_config_id: str | None = None,
    dry_run: bool = False,
    confirm_full_run: bool = False,
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    persist_rag_results: bool = False,
    retrieval_runner: RetrievalRunner | None = None,
    generation_runner: GenerationRunner = run_generation_eval,
    rag_runner: RagRunner = run_rag_eval,
    matrix_run_id: str | None = None,
) -> ExperimentMatrixRunResult | ResolvedSuitePlan:
    suite = matrix_config.suites.get(suite_name)
    if suite is None:
        available = ", ".join(sorted(matrix_config.suites))
        raise ValueError(f"Unknown suite: {suite_name}. Available suites: {available}")

    plan = expand_suite_plan(suite)
    if dry_run:
        _validate_plan(plan=plan, confirm_full_run=True)
        return plan
    _validate_plan(plan=plan, confirm_full_run=confirm_full_run)

    run_started_at = datetime.now().astimezone()
    resolved_matrix_run_id = matrix_run_id or f"{run_started_at.strftime('%Y-%m-%d_%H-%M-%S')}_{suite.name}"
    experiment_output_dir = output_dir.resolve() / resolved_matrix_run_id
    runs_output_dir = experiment_output_dir / "runs"
    runs_output_dir.mkdir(parents=True, exist_ok=False)

    _write_matrix_config_copy(experiment_output_dir, matrix_config.source_path)
    _write_resolved_suite(experiment_output_dir, plan)

    tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)
    successful_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run_spec in plan.runs:
        run_dir = runs_output_dir / run_spec.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_started = datetime.now().astimezone()
        resolved_config = _build_run_config_payload(
            suite_name=suite.name,
            mode=suite.mode,
            run_spec=run_spec,
        )
        _write_json(run_dir / "config.json", resolved_config)

        try:
            if suite.mode == "retrieval":
                retrieval_run = _run_retrieval_suite_item(
                    run_spec=run_spec,
                    run_dir=run_dir,
                    settings=settings,
                    dataset_path=retrieval_dataset_path,
                    tracker=tracker,
                    argv=argv,
                    min_expected_source_coverage=min_expected_source_coverage,
                    retrieval_runner=retrieval_runner,
                )
                metrics_payload = retrieval_run.summary
                results_payloads = retrieval_run.results
                successful_rows.append(
                    _build_retrieval_summary_row(run_spec=run_spec, run_result=retrieval_run)
                )
            elif suite.mode == "generation":
                generation_run = asyncio.run(
                    generation_runner(
                        settings=_apply_generation_settings(settings, run_spec.generation_config),
                        dataset_path=generation_dataset_path.resolve(),
                        output_dir=run_dir,
                        model_config_id=_resolve_generation_model_config_id(
                            run_spec.generation_config
                        ),
                        judge_model_config_id=_resolve_generation_judge_model(
                            run_spec.generation_config,
                            default_judge_model_config_id=generation_judge_model_config_id,
                        ),
                        prompt_version=_resolve_generation_prompt_version(
                            settings=settings,
                            generation_config=run_spec.generation_config,
                        ),
                        temperature=_resolve_generation_temperature(run_spec.generation_config),
                        experiment_name=settings.mlflow_experiment_name,
                        dataset_version=_resolve_generation_dataset_version(
                            run_spec.generation_config
                        ),
                        tracker=tracker,
                        run_name=f"{suite.name}-{run_spec.run_id}",
                        argv=argv,
                    )
                )
                metrics_payload = asdict(generation_run.aggregate)
                results_payloads = [asdict(record) for record in generation_run.records]
                successful_rows.append(
                    _build_generation_summary_row(run_spec=run_spec, run_result=generation_run)
                )
            else:
                rag_run = asyncio.run(
                    rag_runner(
                        settings=_apply_retrieval_settings(settings, run_spec.retrieval_config),
                        dataset_path=rag_dataset_path.resolve(),
                        output_dir=run_dir,
                        model_config_id=_resolve_generation_model_config_id(
                            run_spec.generation_config
                        ),
                        judge_model_config_id=_resolve_generation_judge_model(
                            run_spec.generation_config,
                            default_judge_model_config_id=rag_judge_model_config_id,
                        ),
                        prompt_version=_resolve_generation_prompt_version(
                            settings=settings,
                            generation_config=run_spec.generation_config,
                        ),
                        top_k=_resolve_retrieval_top_k(settings, run_spec.retrieval_config),
                        run_name=f"{suite.name}-{run_spec.run_id}",
                        judge_prompt_path=rag_judge_prompt_path.resolve(),
                        temperature=_resolve_generation_temperature(run_spec.generation_config),
                        experiment_name=settings.mlflow_experiment_name,
                        persist_results=persist_rag_results,
                        tracker=tracker,
                        argv=argv,
                    )
                )
                metrics_payload = asdict(rag_run.summary)
                results_payloads = [asdict(result) for result in rag_run.results]
                successful_rows.append(
                    _build_rag_summary_row(run_spec=run_spec, run_result=rag_run)
                )

            _write_json(run_dir / "metrics.json", metrics_payload)
            _write_jsonl(run_dir / "results.jsonl", results_payloads)
            _write_json(
                run_dir / "status.json",
                {
                    "run_id": run_spec.run_id,
                    "mode": suite.mode,
                    "status": "completed",
                    "started_at": run_started.replace(microsecond=0).isoformat(),
                    "completed_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
                },
            )
        except Exception as exc:
            failure = {
                "run_id": run_spec.run_id,
                "mode": suite.mode,
                "status": "failed",
                "config": {
                    **run_spec.retrieval_config,
                    **run_spec.generation_config,
                },
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
            failures.append(failure)
            _write_json(run_dir / "status.json", failure)

    ranked_rows, ranking_metadata = rank_mode_rows(mode=suite.mode, rows=successful_rows)
    summary_paths = _write_summary_artifacts(
        output_dir=experiment_output_dir,
        mode=suite.mode,
        suite_name=suite.name,
        ranking_metadata=ranking_metadata,
        rows=ranked_rows,
    )
    failures_path = experiment_output_dir / "failures.json"
    _write_json(failures_path, failures)

    completed_at = datetime.now().astimezone()
    if failures and ranked_rows:
        status = "completed_with_failures"
    elif failures:
        status = "failed"
    else:
        status = "completed"
    manifest_path = experiment_output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "matrix_run_id": resolved_matrix_run_id,
            "suite": suite.name,
            "mode": suite.mode,
            "started_at": run_started_at.replace(microsecond=0).isoformat(),
            "completed_at": completed_at.replace(microsecond=0).isoformat(),
            "status": status,
            "total_planned_runs": plan.total_planned_runs,
            "successful_runs": len(ranked_rows),
            "failed_runs": len(failures),
            "config_path": str(matrix_config.source_path),
        },
    )

    if tracker.enabled:
        with tracker.run(f"{suite.name}-summary-{resolved_matrix_run_id}"):
            tracker.log_params(
                build_common_tracking_params(
                    workflow="experiment_matrix",
                    experiment_family="experiment_matrix",
                    run_name=f"{suite.name}-summary-{resolved_matrix_run_id}",
                    git_sha=get_git_sha(),
                    extra={
                        "matrix_config_path": str(matrix_config.source_path),
                        "suite_name": suite.name,
                        "mode": suite.mode,
                        "run_manifest_path": str(manifest_path),
                        "total_planned_runs": plan.total_planned_runs,
                        "successful_runs": len(ranked_rows),
                        "failed_runs": len(failures),
                    },
                )
            )
            if ranked_rows:
                best_row = ranked_rows[0]
                if suite.mode == "generation":
                    tracker.log_metrics(
                        build_generation_tracking_metrics(
                            quality_score=best_row.get("average_quality_score"),
                            groundedness_score=best_row.get("average_groundedness_score"),
                            faithfulness_score=best_row.get("average_faithfulness"),
                            relevance_score=best_row.get("average_answer_relevance"),
                            avg_latency_ms=best_row.get("latency_ms_avg"),
                            p95_latency_ms=best_row.get("latency_ms_p95"),
                            prompt_tokens=None,
                            completion_tokens=None,
                            total_tokens=None,
                            estimated_cost_usd=best_row.get("estimated_total_cost_usd"),
                            extra={"experiment.config_count": len(ranked_rows)},
                        )
                    )
                else:
                    tracker.log_metrics(
                        build_retrieval_tracking_metrics(
                            {
                                "recall_at_k": best_row.get(
                                    "recall_at_k",
                                    best_row.get("avg_recall_at_k"),
                                ),
                                "precision_at_k": best_row.get(
                                    "precision_at_k",
                                    best_row.get("avg_precision_at_k"),
                                ),
                                "mrr": best_row.get("mrr", best_row.get("avg_mrr")),
                                "hit_at_k": best_row.get("hit_at_k"),
                            },
                            extra={"experiment.config_count": len(ranked_rows)},
                        )
                    )
            for artifact_path in summary_paths.values():
                tracker.log_artifact(artifact_path)
            tracker.log_artifact(failures_path)
            tracker.log_artifact(manifest_path)

    return ExperimentMatrixRunResult(
        matrix_run_id=resolved_matrix_run_id,
        suite_name=suite.name,
        mode=suite.mode,
        output_dir=experiment_output_dir,
        manifest_path=manifest_path,
        failures_path=failures_path,
        summary_paths=summary_paths,
        successful_rows=ranked_rows,
        failures=failures,
        status=status,
    )


def _run_retrieval_suite_item(
    *,
    run_spec: MatrixRunSpec,
    run_dir: Path,
    settings: Settings,
    dataset_path: Path,
    tracker,
    argv: list[str],
    min_expected_source_coverage: float,
    retrieval_runner: RetrievalRunner | None,
) -> RetrievalEvalRunResult:
    runner = retrieval_runner
    if runner is None:
        from evals.runners.retrieval_eval_runner import (
            run_retrieval_eval as default_retrieval_runner,
        )

        runner = default_retrieval_runner

    run_settings = _apply_retrieval_settings(settings, run_spec.retrieval_config)
    return runner(
        settings=run_settings,
        dataset_path=dataset_path.resolve(),
        output_root=run_dir,
        top_k=_resolve_retrieval_top_k(settings, run_spec.retrieval_config),
        tracker=tracker,
        argv=argv,
        min_expected_source_coverage=min_expected_source_coverage,
        run_name=f"{run_spec.mode}-{run_spec.run_id}",
        output_label="retrieval_eval",
        timestamp_label=run_spec.run_id,
        chunk_size=_optional_int(run_spec.retrieval_config.get("chunk_size")),
        chunk_overlap=_optional_int(run_spec.retrieval_config.get("chunk_overlap")),
    )


def _apply_retrieval_settings(
    settings: Settings,
    retrieval_config: dict[str, object],
) -> Settings:
    query_rewriting_enabled = _resolve_query_rewriting_enabled(retrieval_config, settings)
    reranker_enabled, reranker_type = _resolve_reranker_config(retrieval_config, settings)
    top_k = _resolve_retrieval_top_k(settings, retrieval_config)
    reranker_initial_top_k = _optional_int(retrieval_config.get("reranker_initial_top_k"))
    chunk_size = int(retrieval_config.get("chunk_size", settings.knowledge_chunk_size))
    chunk_overlap = int(retrieval_config.get("chunk_overlap", settings.knowledge_chunk_overlap))
    if chunk_overlap >= chunk_size:
        raise ValueError("retrieval.chunk_overlap must be smaller than retrieval.chunk_size.")
    return replace(
        settings,
        retriever_type=str(retrieval_config.get("retriever_type", settings.retriever_type)),
        retrieval_top_k=top_k,
        embedding_provider=str(
            retrieval_config.get("embedding_provider", settings.embedding_provider)
        ),
        knowledge_embedding_model=str(
            retrieval_config.get(
                "embedding_model",
                settings.knowledge_embedding_model,
            )
        ),
        embedding_dimension=int(
            retrieval_config.get("embedding_dimension", settings.embedding_dimension)
        ),
        knowledge_chunk_size=chunk_size,
        knowledge_chunk_overlap=chunk_overlap,
        enable_query_rewriting=query_rewriting_enabled,
        query_rewrite_model=str(
            retrieval_config.get("query_rewrite_model", settings.query_rewrite_model)
        ),
        query_rewrite_prompt_version=str(
            retrieval_config.get(
                "query_rewrite_prompt_version",
                settings.query_rewrite_prompt_version,
            )
        ),
        query_rewrite_temperature=float(
            retrieval_config.get(
                "query_rewrite_temperature",
                settings.query_rewrite_temperature,
            )
        ),
        enable_reranking=reranker_enabled,
        reranker_type=reranker_type,
        reranker_model=str(retrieval_config.get("reranker_model", settings.reranker_model)),
        reranker_initial_top_k=(
            reranker_initial_top_k
            if reranker_initial_top_k is not None
            else settings.reranker_initial_top_k
        ),
        reranker_final_top_k=top_k,
    )


def _apply_generation_settings(
    settings: Settings,
    generation_config: dict[str, object],
) -> Settings:
    model_config_id = _resolve_generation_model_config_id(generation_config)
    if model_config_id is None:
        return settings
    return replace(
        settings,
        default_model_config_id=model_config_id,
    )


def _resolve_retrieval_top_k(settings: Settings, retrieval_config: dict[str, object]) -> int:
    top_k = retrieval_config.get("top_k", settings.retrieval_top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("retrieval.top_k must resolve to a positive integer.")
    return top_k


def _resolve_query_rewriting_enabled(
    retrieval_config: dict[str, object],
    settings: Settings,
) -> bool:
    value = retrieval_config.get("query_rewriting_enabled", retrieval_config.get("query_rewriting"))
    if value is None:
        return settings.enable_query_rewriting
    if not isinstance(value, bool):
        raise ValueError("query_rewriting must resolve to a boolean.")
    return value


def _resolve_reranker_config(
    retrieval_config: dict[str, object],
    settings: Settings,
) -> tuple[bool, str]:
    raw_reranker = retrieval_config.get("reranker")
    if isinstance(raw_reranker, str):
        normalized_reranker = raw_reranker.strip().casefold()
        if normalized_reranker == "none":
            return False, "none"
        return True, normalized_reranker

    raw_enabled = retrieval_config.get("reranker_enabled")
    if raw_enabled is None:
        enabled = settings.enable_reranking
    else:
        if not isinstance(raw_enabled, bool):
            raise ValueError("reranker_enabled must resolve to a boolean.")
        enabled = raw_enabled
    raw_type = retrieval_config.get("reranker_type", settings.reranker_type)
    reranker_type = str(raw_type).strip().casefold()
    if not enabled:
        return False, "none"
    return True, reranker_type


def _resolve_generation_model_config_id(generation_config: dict[str, object]) -> str | None:
    value = generation_config.get("model_config_id", generation_config.get("llm_model"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generation.llm_model must resolve to a non-empty string.")
    return value.strip()


def _resolve_generation_prompt_version(
    *,
    settings: Settings,
    generation_config: dict[str, object],
) -> str:
    value = generation_config.get("prompt_version", settings.default_prompt_version)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generation.prompt_version must resolve to a non-empty string.")
    return value.strip()


def _resolve_generation_temperature(generation_config: dict[str, object]) -> float:
    value = generation_config.get("temperature", 0.2)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("generation.temperature must resolve to a number.")
    return float(value)


def _resolve_generation_judge_model(
    generation_config: dict[str, object],
    *,
    default_judge_model_config_id: str | None,
) -> str | None:
    value = generation_config.get(
        "judge_model_config_id",
        generation_config.get("judge_model", default_judge_model_config_id),
    )
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generation.judge_model must resolve to a non-empty string.")
    return value.strip()


def _resolve_generation_dataset_version(generation_config: dict[str, object]) -> str:
    value = generation_config.get("dataset_version", "generation_eval_dataset_v1")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generation.dataset_version must resolve to a non-empty string.")
    return value.strip()


def _build_run_config_payload(
    *,
    suite_name: str,
    mode: str,
    run_spec: MatrixRunSpec,
) -> dict[str, object]:
    return {
        "suite": suite_name,
        "mode": mode,
        "run_id": run_spec.run_id,
        "run_index": run_spec.index,
        "retrieval": run_spec.retrieval_config,
        "generation": run_spec.generation_config,
    }


def _build_retrieval_summary_row(
    *,
    run_spec: MatrixRunSpec,
    run_result: RetrievalEvalRunResult,
) -> dict[str, object]:
    summary = run_result.summary
    return {
        "run_id": run_spec.run_id,
        "run_name": run_result.run_name,
        "retrieval_retriever_type": run_spec.retrieval_config.get("retriever_type"),
        "retrieval_top_k": run_spec.retrieval_config.get("top_k"),
        "retrieval_embedding_provider": run_spec.retrieval_config.get("embedding_provider"),
        "retrieval_embedding_model": run_spec.retrieval_config.get("embedding_model"),
        "retrieval_embedding_dimension": run_spec.retrieval_config.get("embedding_dimension"),
        "retrieval_chunk_size": run_spec.retrieval_config.get("chunk_size"),
        "retrieval_chunk_overlap": run_spec.retrieval_config.get("chunk_overlap"),
        "retrieval_query_rewriting": run_spec.retrieval_config.get(
            "query_rewriting",
            run_spec.retrieval_config.get("query_rewriting_enabled", False),
        ),
        "retrieval_reranker": run_spec.retrieval_config.get(
            "reranker",
            run_spec.retrieval_config.get("reranker_type", "none"),
        ),
        "recall_at_k": summary.get("recall_at_k"),
        "mrr": summary.get("mrr"),
        "precision_at_k": summary.get("precision_at_k", summary.get("mean_precision_at_k")),
        "hit_at_k": summary.get("hit_at_k"),
        "context_relevance": summary.get("context_relevance"),
        "query_rewrite_avg_latency_ms": summary.get("query_rewrite_avg_latency_ms"),
    }


def _build_generation_summary_row(
    *,
    run_spec: MatrixRunSpec,
    run_result: GenerationEvalRunResult,
) -> dict[str, object]:
    aggregate = run_result.aggregate
    return {
        "run_id": run_spec.run_id,
        "run_name": run_result.run_name,
        "generation_llm_model": run_result.model_config_id,
        "generation_prompt_version": run_result.prompt_version,
        "generation_temperature": run_result.temperature,
        "pass_rate": aggregate.pass_rate,
        "average_quality_score": aggregate.average_quality_score,
        "average_groundedness_score": aggregate.average_groundedness_score,
        "average_context_relevance": aggregate.average_context_relevance,
        "average_faithfulness": aggregate.average_faithfulness,
        "average_answer_relevance": aggregate.average_answer_relevance,
        "latency_ms_avg": aggregate.latency_ms_avg,
        "latency_ms_p95": aggregate.latency_ms_p95,
        "estimated_total_cost_usd": aggregate.estimated_total_cost_usd,
        "average_cost_per_response_usd": aggregate.average_cost_per_response_usd,
        "responses_with_cost_estimate": aggregate.responses_with_cost_estimate,
    }


def _build_rag_summary_row(
    *,
    run_spec: MatrixRunSpec,
    run_result: RagEvalRunResult,
) -> dict[str, object]:
    summary = run_result.summary
    return {
        "run_id": run_spec.run_id,
        "run_name": run_result.run_name,
        "retrieval_retriever_type": run_spec.retrieval_config.get("retriever_type"),
        "retrieval_top_k": run_spec.retrieval_config.get("top_k"),
        "retrieval_embedding_provider": run_spec.retrieval_config.get("embedding_provider"),
        "retrieval_embedding_model": run_spec.retrieval_config.get("embedding_model"),
        "retrieval_chunk_size": run_spec.retrieval_config.get("chunk_size"),
        "retrieval_chunk_overlap": run_spec.retrieval_config.get("chunk_overlap"),
        "retrieval_query_rewriting": run_spec.retrieval_config.get(
            "query_rewriting",
            run_spec.retrieval_config.get("query_rewriting_enabled", False),
        ),
        "retrieval_reranker": run_spec.retrieval_config.get(
            "reranker",
            run_spec.retrieval_config.get("reranker_type", "none"),
        ),
        "generation_llm_model": run_result.model_config_id,
        "generation_prompt_version": run_result.prompt_version,
        "generation_temperature": run_result.temperature,
        "avg_precision_at_k": summary.avg_precision_at_k,
        "avg_recall_at_k": summary.avg_recall_at_k,
        "avg_mrr": summary.avg_mrr,
        "avg_ndcg_at_k": summary.avg_ndcg_at_k,
        "avg_context_relevance": summary.avg_context_relevance,
        "avg_faithfulness": summary.avg_faithfulness,
        "avg_answer_relevance": summary.avg_answer_relevance,
        "latency_ms_avg": summary.latency_ms_avg,
        "latency_ms_p95": summary.latency_ms_p95,
        "estimated_total_cost_usd": summary.estimated_total_cost_usd,
        "average_cost_per_question_usd": summary.average_cost_per_question_usd,
    }


def _write_summary_artifacts(
    *,
    output_dir: Path,
    mode: str,
    suite_name: str,
    ranking_metadata: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    summary_json_path = output_dir / f"{mode}_summary.json"
    summary_csv_path = output_dir / f"{mode}_summary.csv"
    ranking_path = output_dir / f"{mode}_ranking.md"
    best_row = next((row for row in rows if row.get("is_best")), None)
    _write_json(
        summary_json_path,
        {
            "suite": suite_name,
            "mode": mode,
            "ranking": ranking_metadata,
            "best_configuration": best_row,
            "runs": rows,
        },
    )
    _write_summary_csv(summary_csv_path, rows)
    ranking_path.write_text(_format_ranking_table(rows), encoding="utf-8")
    return {
        "summary_json": summary_json_path,
        "summary_csv": summary_csv_path,
        "ranking_md": ranking_path,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    lines = [",".join(fieldnames)]
    for row in rows:
        values = []
        for fieldname in fieldnames:
            value = row.get(fieldname)
            if isinstance(value, bool):
                values.append("true" if value else "false")
            elif isinstance(value, int | float):
                values.append(str(value))
            elif value is None:
                values.append("")
            else:
                text = str(value).replace('"', '""')
                if "," in text or "\n" in text:
                    text = f'"{text}"'
                values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_ranking_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = ["rank", "run_id", "score"]
    metric_candidates = [
        "recall_at_k",
        "mrr",
        "precision_at_k",
        "average_quality_score",
        "average_groundedness_score",
        "pass_rate",
        "avg_answer_relevance",
        "avg_faithfulness",
        "avg_recall_at_k",
        "latency_ms_avg",
        "estimated_total_cost_usd",
    ]
    headers.extend([header for header in metric_candidates if header in rows[0]])
    rendered_rows = [
        [str(row.get(header, "")) for header in headers]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(rendered_row[index]) for rendered_row in rendered_rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *[render_row(row) for row in rendered_rows]]) + "\n"


def _validate_plan(*, plan: ResolvedSuitePlan, confirm_full_run: bool) -> None:
    if plan.total_planned_runs > plan.suite.max_combinations:
        raise ValueError(
            "\n".join(
                [
                    f"Suite: {plan.suite.name}",
                    f"Mode: {plan.suite.mode}",
                    "",
                    f"Planned combinations: {plan.total_planned_runs}",
                    f"Configured max_combinations: {plan.suite.max_combinations}",
                    "",
                    "Refusing to run because planned combinations exceed max_combinations.",
                    "Use a smaller suite or intentionally increase max_combinations.",
                ]
            )
        )
    if plan.requires_confirmation and not confirm_full_run:
        raise ValueError(
            f"Suite {plan.suite.name} requires --confirm-full-run before execution."
        )


def _write_matrix_config_copy(output_dir: Path, source_path: Path) -> None:
    copy_path = output_dir / f"matrix_config{source_path.suffix.casefold() or '.yaml'}"
    copy_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_resolved_suite(output_dir: Path, plan: ResolvedSuitePlan) -> None:
    _write_json(
        output_dir / "resolved_suite.json",
        {
            "suite": plan.suite.name,
            "mode": plan.suite.mode,
            "description": plan.suite.description,
            "max_combinations": plan.suite.max_combinations,
            "retrieval_combinations": plan.retrieval_combinations,
            "generation_combinations": plan.generation_combinations,
            "total_planned_runs": plan.total_planned_runs,
            "planned_runs": [
                {
                    "run_id": run.run_id,
                    "index": run.index,
                    "retrieval": run.retrieval_config,
                    "generation": run.generation_config,
                }
                for run in plan.runs
            ],
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer value.")
    return value
