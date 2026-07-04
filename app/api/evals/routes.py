from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.dependencies.common_dependencies import get_app_settings
from app.api.evals.schemas import (
    RetrievalEvalRunRequest,
    RetrievalEvalRunResponse,
    RetrievalEvalSweepRequest,
    RetrievalEvalSweepResponse,
    RetrievalEvalSweepRunResponse,
)
from app.config import Settings
from app.infrastructure.tracking import TrackingSetupError, create_experiment_tracker
from app.services.retrieval import UnsupportedRetrieverError
from evals.retrieval_eval_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
    RetrievalEvalDatasetValidationError,
    RetrievalEvalRunResult,
    run_retrieval_eval,
)
from evals.run_retrieval_sweep import (
    DEFAULT_SWEEP_OUTPUT_DIR,
    RetrievalSweepConfig,
    RetrievalSweepExperiment,
    run_retrieval_sweep,
)

router = APIRouter(prefix="/api/evals", tags=["evals"])


def get_retrieval_eval_runner() -> Callable[..., RetrievalEvalRunResult]:
    return run_retrieval_eval


def get_experiment_tracker_factory() -> Callable[[Settings, str], Any]:
    return create_experiment_tracker


def get_retrieval_sweep_runner() -> Callable[..., tuple[list[dict[str, Any]], dict[str, Path]]]:
    return run_retrieval_sweep


@router.post("/retrieval-runs", response_model=RetrievalEvalRunResponse)
def create_retrieval_eval_run(
    payload: RetrievalEvalRunRequest,
    settings: Settings = Depends(get_app_settings),
    eval_runner: Callable[..., RetrievalEvalRunResult] = Depends(get_retrieval_eval_runner),
    experiment_tracker_factory: Callable[[Settings, str], Any] = Depends(
        get_experiment_tracker_factory
    ),
    eval_admin_token: str | None = Header(default=None, alias="x-eval-admin-token"),
) -> RetrievalEvalRunResponse:
    _validate_eval_admin_token(
        provided_token=eval_admin_token,
        expected_token=settings.eval_admin_token,
    )

    eval_settings = replace(
        settings,
        retriever_type=payload.retriever_type,
        retrieval_top_k=payload.top_k,
        enable_query_rewriting=(
            payload.enable_query_rewriting
            if payload.enable_query_rewriting is not None
            else settings.enable_query_rewriting
        ),
    )

    try:
        tracker = experiment_tracker_factory(eval_settings, eval_settings.mlflow_experiment_name)
        result = eval_runner(
            settings=eval_settings,
            dataset_path=DEFAULT_DATASET_PATH.resolve(),
            output_root=DEFAULT_OUTPUT_DIR.resolve(),
            top_k=payload.top_k,
            tracker=tracker,
            argv=["api:/api/evals/retrieval-runs"],
            run_name=payload.run_name,
            notes=payload.notes,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval eval dataset was not found.",
        ) from exc
    except TrackingSetupError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except RetrievalEvalDatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except UnsupportedRetrieverError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return RetrievalEvalRunResponse(
        status="completed",
        run_name=result.run_name,
        mlflow_run_id=result.mlflow_run_id,
        config=_build_response_config(result),
        metrics=_build_response_metrics(result),
    )


@router.post("/retrieval-sweeps", response_model=RetrievalEvalSweepResponse)
def create_retrieval_eval_sweep(
    payload: RetrievalEvalSweepRequest,
    settings: Settings = Depends(get_app_settings),
    sweep_runner: Callable[..., tuple[list[dict[str, Any]], dict[str, Path]]] = Depends(
        get_retrieval_sweep_runner
    ),
    eval_admin_token: str | None = Header(default=None, alias="x-eval-admin-token"),
) -> RetrievalEvalSweepResponse:
    _validate_eval_admin_token(
        provided_token=eval_admin_token,
        expected_token=settings.eval_admin_token,
    )

    output_dir = (
        DEFAULT_SWEEP_OUTPUT_DIR.resolve()
        / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_api_retrieval_sweep"
    )

    try:
        rows, artifact_paths = sweep_runner(
            sweep_config=_build_sweep_config(payload),
            sweep_config_path=Path("api_retrieval_sweep_request.json"),
            dataset_path=DEFAULT_DATASET_PATH.resolve(),
            output_dir=output_dir,
            settings=replace(
                settings,
                enable_query_rewriting=(
                    payload.enable_query_rewriting
                    if payload.enable_query_rewriting is not None
                    else settings.enable_query_rewriting
                ),
            ),
            argv=["api:/api/evals/retrieval-sweeps"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval eval dataset was not found.",
        ) from exc
    except TrackingSetupError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except RetrievalEvalDatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except UnsupportedRetrieverError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return RetrievalEvalSweepResponse(
        status="completed",
        runs=[_build_sweep_run_response(row) for row in rows],
        artifacts={name: str(path) for name, path in artifact_paths.items()},
    )


def _validate_eval_admin_token(
    *,
    provided_token: str | None,
    expected_token: str | None,
) -> None:
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EVAL_ADMIN_TOKEN is not configured.",
        )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid eval admin token.",
        )


def _build_response_config(result: RetrievalEvalRunResult) -> dict[str, Any]:
    return {
        "retriever_type": result.config.get("retriever_type"),
        "top_k": result.config.get("top_k"),
        "dataset_path": result.config.get("dataset_path"),
        "embedding_provider": result.config.get("embedding_provider"),
        "embedding_model": result.config.get("embedding_model"),
        "embedding_dimension": result.config.get("embedding_dimension"),
        "query_rewriting_enabled": result.config.get("query_rewriting_enabled"),
        "query_rewrite_model": result.config.get("query_rewrite_model"),
        "query_rewrite_prompt_version": result.config.get("query_rewrite_prompt_version"),
        "notes": result.config.get("notes"),
    }


def _build_response_metrics(result: RetrievalEvalRunResult) -> dict[str, Any]:
    return {
        "hit_at_k": result.summary.get("hit_at_k"),
        "mrr": result.summary.get("mrr"),
        "recall_at_k": result.summary.get("recall_at_k"),
        "mean_precision_at_k": result.summary.get("mean_precision_at_k"),
        "num_queries_total": result.summary.get("num_queries_total"),
        "num_queries_evaluated": result.summary.get("num_queries_evaluated"),
        "num_queries_without_expected_source": result.summary.get(
            "num_queries_without_expected_source"
        ),
        "query_rewrite_avg_latency_ms": result.summary.get("query_rewrite_avg_latency_ms"),
        "query_rewrite_total_latency_ms": result.summary.get("query_rewrite_total_latency_ms"),
        "query_rewrite_success_count": result.summary.get("query_rewrite_success_count"),
        "query_rewrite_fallback_count": result.summary.get("query_rewrite_fallback_count"),
        "query_rewrite_failure_count": result.summary.get("query_rewrite_failure_count"),
        "query_rewrite_total_tokens": result.summary.get("query_rewrite_total_tokens"),
        "query_rewrite_estimated_total_cost": result.summary.get(
            "query_rewrite_estimated_total_cost"
        ),
    }


def _build_sweep_config(payload: RetrievalEvalSweepRequest) -> RetrievalSweepConfig:
    experiments: list[RetrievalSweepExperiment] = []
    seen_names: set[str] = set()

    for index, experiment in enumerate(payload.experiments, start=1):
        normalized_name = experiment.name.strip()
        if not normalized_name:
            raise ValueError(f"experiments[{index}].name must be a non-empty string.")
        if normalized_name in seen_names:
            raise ValueError(
                f"Retrieval sweep config contains duplicate experiment name: {normalized_name}."
            )
        seen_names.add(normalized_name)

        if (
            experiment.chunk_size is not None
            and experiment.chunk_overlap is not None
            and experiment.chunk_overlap >= experiment.chunk_size
        ):
            raise ValueError(
                f"experiments[{index}].chunk_overlap must be smaller than chunk_size."
            )

        experiments.append(
            RetrievalSweepExperiment(
                name=normalized_name,
                retriever_type=experiment.retriever_type,
                top_k=experiment.top_k,
                embedding_provider=_normalize_optional_string(
                    experiment.embedding_provider,
                    normalize_case=True,
                ),
                embedding_model=_normalize_optional_string(experiment.embedding_model),
                embedding_dimension=experiment.embedding_dimension,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
            )
        )

    return RetrievalSweepConfig(experiments=experiments)


def _build_sweep_run_response(row: dict[str, Any]) -> RetrievalEvalSweepRunResponse:
    return RetrievalEvalSweepRunResponse(
        experiment_name=str(row.get("experiment_name", "")),
        run_name=str(row.get("run_name", "")),
        config={
            "retriever_type": row.get("retriever_type"),
            "top_k": row.get("top_k"),
            "dataset_path": row.get("dataset_path"),
            "embedding_provider": row.get("embedding_provider"),
            "embedding_model": row.get("embedding_model"),
            "embedding_dimension": row.get("embedding_dimension"),
            "query_rewriting_enabled": row.get("query_rewriting_enabled"),
            "query_rewrite_model": row.get("query_rewrite_model"),
            "query_rewrite_prompt_version": row.get("query_rewrite_prompt_version"),
            "git_commit_sha": row.get("git_commit_sha"),
        },
        metrics={
            "hit_at_k": row.get("hit_at_k"),
            "mrr": row.get("mrr"),
            "recall_at_k": row.get("recall_at_k"),
            "mean_precision_at_k": row.get("mean_precision_at_k"),
            "num_queries_total": row.get("num_queries_total"),
            "num_queries_evaluated": row.get("num_queries_evaluated"),
            "num_queries_without_expected_source": row.get(
                "num_queries_without_expected_source"
            ),
            "query_rewrite_avg_latency_ms": row.get("query_rewrite_avg_latency_ms"),
            "query_rewrite_total_latency_ms": row.get("query_rewrite_total_latency_ms"),
            "query_rewrite_success_count": row.get("query_rewrite_success_count"),
            "query_rewrite_fallback_count": row.get("query_rewrite_fallback_count"),
            "query_rewrite_failure_count": row.get("query_rewrite_failure_count"),
            "query_rewrite_total_tokens": row.get("query_rewrite_total_tokens"),
            "query_rewrite_estimated_total_cost": row.get("query_rewrite_estimated_total_cost"),
        },
        artifacts={
            "output_dir": str(row.get("output_dir", "")),
            "results_json": str(row.get("results_json", "")),
            "results_csv": str(row.get("results_csv", "")),
            "config_json": str(row.get("config_json", "")),
        },
    )


def _normalize_optional_string(
    value: str | None,
    *,
    normalize_case: bool = False,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized.casefold() if normalize_case else normalized
