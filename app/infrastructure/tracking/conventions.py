from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from app.infrastructure.prompts import normalize_prompt_version

ROOT_DIR = Path(__file__).resolve().parents[3]


def get_git_sha(root_dir: Path = ROOT_DIR) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    sha = completed.stdout.strip()
    return sha or None


def percentile(values: list[float], *, percentile_value: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * percentile_value))))
    return values[index]


def resolve_prompt_template_path(*, prompts_dir: Path, prompt_version: str) -> Path | None:
    normalized_version = normalize_prompt_version(prompt_version)
    candidate = prompts_dir / f"{normalized_version}.md"
    if candidate.is_file():
        return candidate
    return None


def extract_model_provider(model_config_id: str | None) -> str | None:
    if model_config_id is None or ":" not in model_config_id:
        return None
    provider, _model = model_config_id.split(":", 1)
    normalized = provider.strip().casefold()
    return normalized or None


def extract_model_name(model_config_id: str | None) -> str | None:
    if model_config_id is None:
        return None
    if ":" not in model_config_id:
        normalized = model_config_id.strip()
        return normalized or None
    _provider, model = model_config_id.split(":", 1)
    normalized = model.strip()
    return normalized or None


def resolve_vector_store_provider(settings: Any) -> str | None:
    value = getattr(settings, "vector_store_provider", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    retriever_type = str(getattr(settings, "retriever_type", "")).casefold()
    if retriever_type in {"vector", "hybrid"}:
        return "pgvector"
    return None


def build_common_tracking_params(
    *,
    workflow: str,
    experiment_family: str,
    run_name: str,
    dataset_path: Path | None = None,
    dataset_version: str | None = None,
    git_sha: str | None = None,
    prompt_version: str | None = None,
    prompt_template_path: Path | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "workflow": workflow,
        "experiment_family": experiment_family,
        "run_name": run_name,
        "git_sha": git_sha,
    }
    if dataset_path is not None:
        params["dataset_name"] = dataset_path.name
        params["dataset_path"] = str(dataset_path)
        params["dataset_version"] = dataset_version or dataset_path.stem
    if prompt_version is not None:
        params["prompt_version"] = prompt_version
    if prompt_template_path is not None:
        params["prompt_template_id"] = prompt_template_path.stem
        params["prompt_template_path"] = str(prompt_template_path)
    if extra:
        params.update(extra)
    return params


def build_retrieval_tracking_params(
    *,
    workflow: str,
    experiment_family: str,
    run_name: str,
    settings: Any,
    dataset_path: Path,
    top_k: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
    git_sha: str | None,
    notes: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    params = build_common_tracking_params(
        workflow=workflow,
        experiment_family=experiment_family,
        run_name=run_name,
        dataset_path=dataset_path,
        git_sha=git_sha,
        extra={
            "notes": notes,
            "retrieval_config": getattr(settings, "default_retrieval_config", None),
            "retriever_type": getattr(settings, "retriever_type", None),
            "top_k": top_k,
            "embedding_provider": getattr(settings, "embedding_provider", None),
            "embedding_model": getattr(settings, "knowledge_embedding_model", None),
            "embedding_dimension": getattr(settings, "embedding_dimension", None),
            "vector_store_provider": resolve_vector_store_provider(settings),
            "knowledge_collection_name": getattr(settings, "knowledge_collection_name", None),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_min_similarity": getattr(settings, "retrieval_min_similarity", None),
            "query_rewriting_enabled": getattr(settings, "enable_query_rewriting", None),
            "query_rewrite_provider": extract_model_provider(
                getattr(settings, "query_rewrite_model", None)
            ),
            "query_rewrite_model": getattr(settings, "query_rewrite_model", None),
            "query_rewrite_prompt_version": getattr(
                settings,
                "query_rewrite_prompt_version",
                None,
            ),
            "query_rewrite_temperature": getattr(settings, "query_rewrite_temperature", None),
            "query_rewrite_timeout_seconds": getattr(
                settings,
                "query_rewrite_timeout_seconds",
                None,
            ),
            "query_rewrite_max_tokens": getattr(settings, "query_rewrite_max_tokens", None),
            "reranker_enabled": getattr(settings, "enable_reranking", None),
            "reranker_type": getattr(settings, "reranker_type", None),
            "reranker_provider": extract_model_provider(
                getattr(settings, "reranker_model", None)
            ),
            "reranker_model": getattr(settings, "reranker_model", None),
            "reranker_initial_top_k": getattr(settings, "reranker_initial_top_k", None),
            "reranker_final_top_k": getattr(settings, "reranker_final_top_k", None),
        },
    )
    if extra:
        params.update(extra)
    return params


def build_generation_tracking_params(
    *,
    workflow: str,
    experiment_family: str,
    run_name: str,
    dataset_path: Path,
    dataset_version: str | None,
    prompt_version: str,
    prompt_template_path: Path | None,
    model_config_id: str,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    temperature: float,
    max_tokens: int | None,
    git_sha: str | None,
    judge_model_config_id: str | None = None,
    retrieval_config: str | None = None,
    context_top_k: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    params = build_common_tracking_params(
        workflow=workflow,
        experiment_family=experiment_family,
        run_name=run_name,
        dataset_path=dataset_path,
        dataset_version=dataset_version,
        git_sha=git_sha,
        prompt_version=prompt_version,
        prompt_template_path=prompt_template_path,
        extra={
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_base_url": llm_base_url,
            "model_config_id": model_config_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "retrieval_config": retrieval_config,
            "context_top_k": context_top_k,
            "judge_model_config_id": judge_model_config_id,
            "judge_provider": extract_model_provider(judge_model_config_id),
            "judge_model": extract_model_name(judge_model_config_id),
        },
    )
    if extra:
        params.update(extra)
    return params


def build_retrieval_tracking_metrics(
    summary: dict[str, Any],
    *,
    extra: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "retrieval.recall_at_k": float(summary.get("recall_at_k") or 0.0),
        "retrieval.precision_at_k": float(
            summary.get("precision_at_k", summary.get("mean_precision_at_k")) or 0.0
        ),
        "retrieval.mrr": float(summary.get("mrr") or 0.0),
        "retrieval.hit_rate": float(summary.get("hit_at_k") or 0.0),
        "retrieval.avg_latency_ms": float(summary.get("retrieval_avg_latency_ms") or 0.0),
        "retrieval.p95_latency_ms": float(summary.get("retrieval_p95_latency_ms") or 0.0),
        "retrieval.context_relevance": float(summary.get("context_relevance") or 0.0),
        "retrieval.total_queries": int(summary.get("num_queries_total") or 0),
        "retrieval.evaluated_queries": int(summary.get("num_queries_evaluated") or 0),
        "retrieval.unevaluated_queries": int(
            summary.get("num_queries_without_expected_sources")
            or summary.get("num_queries_without_expected_source")
            or 0
        ),
        "query_rewrite.avg_latency_ms": float(
            summary.get("query_rewrite_avg_latency_ms") or 0.0
        ),
        "query_rewrite.total_latency_ms": float(
            summary.get("query_rewrite_total_latency_ms") or 0.0
        ),
        "query_rewrite.success_count": int(summary.get("query_rewrite_success_count") or 0),
        "query_rewrite.fallback_count": int(
            summary.get("query_rewrite_fallback_count") or 0
        ),
        "query_rewrite.failure_count": int(summary.get("query_rewrite_failure_count") or 0),
        "query_rewrite.prompt_tokens": int(
            summary.get("query_rewrite_total_prompt_tokens") or 0
        ),
        "query_rewrite.completion_tokens": int(
            summary.get("query_rewrite_total_completion_tokens") or 0
        ),
        "query_rewrite.total_tokens": int(summary.get("query_rewrite_total_tokens") or 0),
        "query_rewrite.estimated_cost_usd": float(
            summary.get("query_rewrite_estimated_total_cost") or 0.0
        ),
    }
    if extra:
        metrics.update(extra)
    return metrics


def build_generation_tracking_metrics(
    *,
    quality_score: float | None,
    groundedness_score: float | None,
    faithfulness_score: float | None,
    relevance_score: float | None,
    avg_latency_ms: float | None,
    p95_latency_ms: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    estimated_cost_usd: float | None,
    extra: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "generation.quality_score": float(quality_score or 0.0),
        "generation.groundedness_score": float(groundedness_score or 0.0),
        "generation.faithfulness_score": float(faithfulness_score or 0.0),
        "generation.relevance_score": float(relevance_score or 0.0),
        "generation.avg_latency_ms": float(avg_latency_ms or 0.0),
        "generation.p95_latency_ms": float(p95_latency_ms or 0.0),
        "generation.prompt_tokens": int(prompt_tokens or 0),
        "generation.completion_tokens": int(completion_tokens or 0),
        "generation.total_tokens": int(total_tokens or 0),
        "generation.estimated_cost_usd": float(estimated_cost_usd or 0.0),
    }
    if extra:
        metrics.update(extra)
    return metrics


def build_rag_tracking_metrics(summary: Any) -> dict[str, float | int]:
    return {
        "retrieval.recall_at_k": float(summary.avg_recall_at_k),
        "retrieval.precision_at_k": float(summary.avg_precision_at_k),
        "retrieval.mrr": float(summary.avg_mrr),
        "generation.faithfulness_score": float(summary.avg_faithfulness),
        "generation.relevance_score": float(summary.avg_answer_relevance),
        "generation.avg_latency_ms": float(summary.latency_ms_avg),
        "generation.p95_latency_ms": float(summary.latency_ms_p95),
        "generation.estimated_cost_usd": float(summary.estimated_total_cost_usd or 0.0),
        "rag.answer_quality": float(summary.avg_answer_relevance),
        "rag.retrieval_relevance": float(summary.avg_context_relevance),
        "rag.groundedness": float(summary.avg_faithfulness),
        "rag.end_to_end_latency_ms": float(summary.latency_ms_avg),
        "rag.estimated_cost_usd": float(summary.estimated_total_cost_usd or 0.0),
        "rag.total_questions": int(summary.total_questions),
    }
