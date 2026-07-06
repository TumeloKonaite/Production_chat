from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.infrastructure.tracking import create_experiment_tracker
from app.services.evals.eval_run_service import EvalRunService
from evals.matrix import expand_suite_plan, load_experiment_matrix_config, run_experiment_matrix
from evals.retrieval_eval_runner import run_retrieval_eval
from evals.run_experiment_matrix import DEFAULT_CONFIG_PATH
from evals.run_generation_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_GENERATION_DATASET_PATH,
    run_generation_eval,
)
from evals.run_rag_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RAG_DATASET_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
    run_rag_eval,
)
from evals.run_retrieval_eval import DEFAULT_DATASET_PATH as DEFAULT_RETRIEVAL_DATASET_PATH


class EvalJobRunner:
    def __init__(
        self,
        *,
        run_service: EvalRunService,
        experiment_tracker_factory: Callable[[Settings, str], Any] = create_experiment_tracker,
        retrieval_runner: Callable[..., Any] = run_retrieval_eval,
        generation_runner: Callable[..., Any] = run_generation_eval,
        rag_runner: Callable[..., Any] = run_rag_eval,
        matrix_runner: Callable[..., Any] = run_experiment_matrix,
        matrix_config_loader: Callable[[Path], Any] = load_experiment_matrix_config,
        matrix_config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self._run_service = run_service
        self._experiment_tracker_factory = experiment_tracker_factory
        self._retrieval_runner = retrieval_runner
        self._generation_runner = generation_runner
        self._rag_runner = rag_runner
        self._matrix_runner = matrix_runner
        self._matrix_config_loader = matrix_config_loader
        self._matrix_config_path = matrix_config_path.resolve()

    def resolve_matrix_plan(
        self,
        *,
        suite_name: str,
        dry_run: bool,
        confirm_full_run: bool,
    ) -> Any:
        matrix_config = self._matrix_config_loader(self._matrix_config_path)
        suite = matrix_config.suites.get(suite_name)
        if suite is None:
            available = ", ".join(sorted(matrix_config.suites))
            raise ValueError(f"Unknown suite: {suite_name}. Available suites: {available}")

        plan = expand_suite_plan(suite)
        if plan.total_planned_runs > plan.suite.max_combinations:
            raise ValueError(
                "Refusing to run because planned combinations exceed max_combinations."
            )
        if not dry_run and plan.requires_confirmation and not confirm_full_run:
            raise ValueError(f"Suite {plan.suite.name} requires confirm_full_run=true.")
        return plan

    def run_retrieval_job(self, *, run_id: str, payload: dict[str, Any], settings: Settings) -> None:
        run_dir = self._run_service.get_run(run_id).run_dir
        self._run_service.update_run(run_id, status="running")
        try:
            resolved_settings = _apply_retrieval_settings(settings, payload)
            tracker = self._experiment_tracker_factory(
                resolved_settings,
                resolved_settings.mlflow_experiment_name,
            )
            result = self._retrieval_runner(
                settings=resolved_settings,
                dataset_path=DEFAULT_RETRIEVAL_DATASET_PATH.resolve(),
                output_root=(run_dir / "runner_output").resolve(),
                top_k=int(payload["top_k"]),
                tracker=tracker,
                argv=["api:/api/evals/retrieval"],
                run_name=run_id,
                notes=payload.get("notes"),
                output_label="retrieval_eval",
                timestamp_label=run_id,
                chunk_size=_optional_int(payload.get("chunk_size")),
                chunk_overlap=_optional_int(payload.get("chunk_overlap")),
            )
            self._run_service.update_run(
                run_id,
                status="completed",
                summary_payload={
                    "run_id": run_id,
                    "mode": "retrieval",
                    "summary": result.summary,
                    "config": result.config,
                },
                failures_payload={"run_id": run_id, "failures": []},
                artifacts=_serialize_artifacts(run_dir, result.artifact_paths | {"output_dir": result.output_dir}),
            )
        except Exception as exc:
            self._record_failure(run_id=run_id, mode="retrieval", payload=payload, exc=exc)

    def run_generation_job(self, *, run_id: str, payload: dict[str, Any], settings: Settings) -> None:
        run_dir = self._run_service.get_run(run_id).run_dir
        self._run_service.update_run(run_id, status="running")
        try:
            result = asyncio.run(
                self._generation_runner(
                    settings=settings,
                    dataset_path=DEFAULT_GENERATION_DATASET_PATH.resolve(),
                    output_dir=(run_dir / "runner_output").resolve(),
                    model_config_id=_resolve_model_config_id(settings, payload),
                    judge_model_config_id=_resolve_judge_model_config_id(payload),
                    prompt_version=_resolve_prompt_version(settings, payload),
                    temperature=float(payload.get("temperature", 0.2)),
                    dataset_version=str(
                        payload.get("dataset_version") or "generation_eval_dataset_v1"
                    ),
                    run_name=run_id,
                    argv=["api:/api/evals/generation"],
                )
            )
            aggregate_payload = asdict(result.aggregate)
            self._run_service.update_run(
                run_id,
                status="completed",
                summary_payload={
                    "run_id": run_id,
                    "mode": "generation",
                    "summary": aggregate_payload,
                    "config": {
                        "model_config_id": result.model_config_id,
                        "judge_model_config_id": result.judge_model_config_id,
                        "prompt_version": result.prompt_version,
                        "temperature": result.temperature,
                        "dataset_version": result.dataset_version,
                        "max_tokens": payload.get("max_tokens"),
                    },
                },
                failures_payload={"run_id": run_id, "failures": []},
                artifacts=_serialize_artifacts(run_dir, result.artifact_paths),
            )
        except Exception as exc:
            self._record_failure(run_id=run_id, mode="generation", payload=payload, exc=exc)

    def run_rag_job(self, *, run_id: str, payload: dict[str, Any], settings: Settings) -> None:
        run_dir = self._run_service.get_run(run_id).run_dir
        self._run_service.update_run(run_id, status="running")
        try:
            retrieval_payload = dict(payload["retrieval"])
            generation_payload = dict(payload["generation"])
            resolved_settings = _apply_retrieval_settings(settings, retrieval_payload)
            result = asyncio.run(
                self._rag_runner(
                    settings=resolved_settings,
                    dataset_path=DEFAULT_RAG_DATASET_PATH.resolve(),
                    output_dir=(run_dir / "runner_output").resolve(),
                    model_config_id=_resolve_model_config_id(settings, generation_payload),
                    judge_model_config_id=_resolve_judge_model_config_id(generation_payload),
                    prompt_version=_resolve_prompt_version(settings, generation_payload),
                    top_k=int(retrieval_payload["top_k"]),
                    run_name=run_id,
                    judge_prompt_path=DEFAULT_JUDGE_PROMPT_PATH.resolve(),
                    temperature=float(generation_payload.get("temperature", 0.2)),
                    persist_results=False,
                    argv=["api:/api/evals/rag"],
                )
            )
            summary_payload = asdict(result.summary)
            self._run_service.update_run(
                run_id,
                status="completed",
                summary_payload={
                    "run_id": run_id,
                    "mode": "rag",
                    "summary": summary_payload,
                    "retrieval": result.retrieval_config,
                    "generation": {
                        "model_config_id": result.model_config_id,
                        "judge_model_config_id": result.judge_model_config_id,
                        "prompt_version": result.prompt_version,
                        "temperature": result.temperature,
                    },
                },
                failures_payload={"run_id": run_id, "failures": []},
                artifacts=_serialize_artifacts(run_dir, result.artifact_paths),
            )
        except Exception as exc:
            self._record_failure(run_id=run_id, mode="rag", payload=payload, exc=exc)

    def run_matrix_job(self, *, run_id: str, payload: dict[str, Any], settings: Settings) -> None:
        run_dir = self._run_service.get_run(run_id).run_dir
        self._run_service.update_run(run_id, status="running")
        try:
            plan = self.resolve_matrix_plan(
                suite_name=str(payload["suite"]),
                dry_run=False,
                confirm_full_run=bool(payload.get("confirm_full_run", False)),
            )
            matrix_config = self._matrix_config_loader(self._matrix_config_path)
            result = self._matrix_runner(
                matrix_config=matrix_config,
                suite_name=str(payload["suite"]),
                settings=settings,
                argv=["api:/api/evals/matrix"],
                output_dir=(run_dir / "runner_output").resolve(),
                dry_run=False,
                confirm_full_run=bool(payload.get("confirm_full_run", False)),
                persist_rag_results=False,
                matrix_run_id=run_id,
            )
            raw_summary = json.loads(
                result.summary_paths["summary_json"].read_text(encoding="utf-8")
            )
            self._run_service.update_run(
                run_id,
                status=result.status,
                total_planned_runs=plan.total_planned_runs,
                successful_runs=len(result.successful_rows),
                failed_runs=len(result.failures),
                summary_payload={
                    "run_id": run_id,
                    "suite": result.suite_name,
                    "mode": result.mode,
                    "top_results": raw_summary.get("runs", [])[:10],
                    "best_configuration": raw_summary.get("best_configuration"),
                    "ranking": raw_summary.get("ranking"),
                    "total_runs": len(raw_summary.get("runs", [])),
                },
                failures_payload={"run_id": run_id, "failures": result.failures},
                artifacts=_serialize_artifacts(
                    run_dir,
                    result.summary_paths
                    | {
                        "output_dir": result.output_dir,
                        "manifest_path": result.manifest_path,
                        "failures_path": result.failures_path,
                    },
                ),
            )
        except Exception as exc:
            self._record_failure(run_id=run_id, mode="matrix", payload=payload, exc=exc)

    def _record_failure(
        self,
        *,
        run_id: str,
        mode: str,
        payload: dict[str, Any],
        exc: Exception,
    ) -> None:
        self._run_service.update_run(
            run_id,
            status="failed",
            failures_payload={
                "run_id": run_id,
                "failures": [
                    {
                        "run_id": run_id,
                        "mode": mode,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                        "config": payload,
                    }
                ],
            },
        )


def _apply_retrieval_settings(settings: Settings, payload: dict[str, Any]) -> Settings:
    chunk_size = _optional_int(payload.get("chunk_size")) or settings.knowledge_chunk_size
    chunk_overlap = (
        _optional_int(payload.get("chunk_overlap"))
        if payload.get("chunk_overlap") is not None
        else settings.knowledge_chunk_overlap
    )
    if chunk_overlap >= chunk_size:
        raise ValueError("retrieval.chunk_overlap must be smaller than retrieval.chunk_size.")

    query_rewriting_enabled = payload.get("query_rewriting_enabled")
    reranking_enabled = payload.get("reranking_enabled")
    resolved_query_rewriting = (
        settings.enable_query_rewriting
        if query_rewriting_enabled is None
        else bool(query_rewriting_enabled)
    )
    resolved_reranking = (
        settings.enable_reranking if reranking_enabled is None else bool(reranking_enabled)
    )
    reranker_type = (
        "none"
        if not resolved_reranking
        else str(payload.get("reranker_type") or settings.reranker_type).casefold()
    )
    top_k = int(payload["top_k"])
    reranker_initial_top_k = _optional_int(payload.get("reranker_initial_top_k"))

    return replace(
        settings,
        retriever_type=str(payload.get("retriever_type") or settings.retriever_type),
        retrieval_top_k=top_k,
        embedding_provider=str(payload.get("embedding_provider") or settings.embedding_provider),
        knowledge_embedding_model=str(
            payload.get("embedding_model") or settings.knowledge_embedding_model
        ),
        embedding_dimension=int(payload.get("embedding_dimension") or settings.embedding_dimension),
        knowledge_chunk_size=chunk_size,
        knowledge_chunk_overlap=chunk_overlap,
        enable_query_rewriting=resolved_query_rewriting,
        query_rewrite_model=str(payload.get("query_rewrite_model") or settings.query_rewrite_model),
        query_rewrite_prompt_version=str(
            payload.get("query_rewrite_prompt_version") or settings.query_rewrite_prompt_version
        ),
        query_rewrite_temperature=float(
            payload.get("query_rewrite_temperature") or settings.query_rewrite_temperature
        ),
        enable_reranking=resolved_reranking,
        reranker_type=reranker_type,
        reranker_model=str(payload.get("reranker_model") or settings.reranker_model),
        reranker_initial_top_k=(
            reranker_initial_top_k
            if reranker_initial_top_k is not None
            else settings.reranker_initial_top_k
        ),
        reranker_final_top_k=top_k,
    )


def _resolve_model_config_id(settings: Settings, payload: dict[str, Any]) -> str:
    explicit_model_config_id = payload.get("model_config_id")
    if isinstance(explicit_model_config_id, str) and explicit_model_config_id.strip():
        return explicit_model_config_id.strip()

    llm_model = payload.get("llm_model")
    if not isinstance(llm_model, str) or not llm_model.strip():
        return settings.default_model_config_id
    normalized_llm_model = llm_model.strip()
    if ":" in normalized_llm_model:
        return normalized_llm_model
    provider = str(payload.get("provider") or settings.llm_provider).strip().casefold()
    return f"{provider}:{normalized_llm_model}"


def _resolve_judge_model_config_id(payload: dict[str, Any]) -> str | None:
    explicit_judge_model = payload.get("judge_model_config_id")
    if isinstance(explicit_judge_model, str) and explicit_judge_model.strip():
        return explicit_judge_model.strip()
    judge_model = payload.get("judge_model")
    if not isinstance(judge_model, str) or not judge_model.strip():
        return None
    return judge_model.strip()


def _resolve_prompt_version(settings: Settings, payload: dict[str, Any]) -> str:
    prompt_version = payload.get("prompt_version")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        return settings.default_prompt_version
    return prompt_version.strip()


def _serialize_artifacts(run_dir: Path, artifacts: dict[str, Path]) -> dict[str, str]:
    resolved_run_dir = run_dir.resolve()
    payload: dict[str, str] = {}
    for name, path in artifacts.items():
        resolved_path = path.resolve()
        try:
            payload[name] = str(resolved_path.relative_to(resolved_run_dir))
        except ValueError:
            payload[name] = str(resolved_path)
    return payload


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected an integer value.")
    return value
