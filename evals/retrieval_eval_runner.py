from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import Settings
from app.services.retrieval import RetrievalService
from evals.metrics.retrieval_metrics import (
    first_relevant_rank,
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    unique_ranked_sources,
)

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "portfolio_eval_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results" / "retrieval"
DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE = 0.95
SAFE_LABEL_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrievalEvalExample:
    id: str
    question: str
    expected_source_documents: list[str]


@dataclass(frozen=True, slots=True)
class RetrievalDatasetValidationSummary:
    total_queries: int
    queries_with_expected_sources: int
    queries_without_expected_sources: int
    missing_expected_source_ids: list[str]

    @property
    def expected_source_coverage(self) -> float:
        if self.total_queries == 0:
            return 1.0
        return self.queries_with_expected_sources / self.total_queries


@dataclass(frozen=True, slots=True)
class RetrievalEvalRunResult:
    run_name: str
    mlflow_run_id: str | None
    output_dir: Path
    summary: dict[str, Any]
    results: list[dict[str, Any]]
    config: dict[str, Any]
    artifact_paths: dict[str, Path]
    validation_summary: RetrievalDatasetValidationSummary


class RetrievalEvalDatasetValidationError(ValueError):
    """Raised when the retrieval evaluation dataset is missing too many expected sources."""


def load_dataset(path: Path) -> list[RetrievalEvalExample]:
    examples: list[RetrievalEvalExample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        payload = json.loads(line)
        examples.append(
            RetrievalEvalExample(
                id=str(payload["id"]),
                question=str(payload["question"]),
                expected_source_documents=[
                    str(item) for item in payload.get("expected_source_documents", [])
                ],
            )
        )
    return examples


def load_and_validate_dataset(
    path: Path,
    *,
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
) -> tuple[list[RetrievalEvalExample], RetrievalDatasetValidationSummary]:
    examples = load_dataset(path)
    validation_summary = validate_dataset_examples(
        examples,
        min_expected_source_coverage=min_expected_source_coverage,
    )
    return examples, validation_summary


def validate_dataset_examples(
    examples: list[RetrievalEvalExample],
    *,
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
) -> RetrievalDatasetValidationSummary:
    if not 0.0 <= min_expected_source_coverage <= 1.0:
        raise ValueError("--min-expected-source-coverage must be between 0.0 and 1.0.")

    missing_expected_source_ids = [
        example.id for example in examples if not example.expected_source_documents
    ]
    summary = RetrievalDatasetValidationSummary(
        total_queries=len(examples),
        queries_with_expected_sources=len(examples) - len(missing_expected_source_ids),
        queries_without_expected_sources=len(missing_expected_source_ids),
        missing_expected_source_ids=missing_expected_source_ids,
    )

    if summary.expected_source_coverage < min_expected_source_coverage:
        raise RetrievalEvalDatasetValidationError(
            "Retrieval eval dataset is not valid: "
            f"{summary.queries_without_expected_sources} of {summary.total_queries} queries "
            "are missing expected_source_documents."
        )

    return summary


def format_dataset_validation_summary(summary: RetrievalDatasetValidationSummary) -> str:
    lines = [
        "Retrieval eval dataset validation",
        "---------------------------------",
        f"total_queries: {summary.total_queries}",
        f"queries_with_expected_sources: {summary.queries_with_expected_sources}",
        f"queries_without_expected_sources: {summary.queries_without_expected_sources}",
    ]
    if summary.missing_expected_source_ids:
        lines.append(
            "queries_missing_expected_sources: "
            + ", ".join(summary.missing_expected_source_ids)
        )
    return "\n".join(lines)


def evaluate_examples(
    examples: list[RetrievalEvalExample],
    retrieval_service: RetrievalService,
    *,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, per_query_results = evaluate_examples_for_k_values(
        examples,
        retrieval_service,
        k_values=[k],
    )
    return {
        "num_queries_total": summary["num_queries_total"],
        "num_queries_evaluated": summary["num_queries_evaluated"],
        "num_queries_without_expected_source": summary["num_queries_without_expected_source"],
        "num_queries_without_expected_sources": summary["num_queries_without_expected_sources"],
        "k": k,
        "hit_at_k": summary["hit_at_k"],
        "recall_at_k": summary["recall_at_k"],
        "mean_precision_at_k": summary["mean_precision_at_k"],
        "mrr": summary["mrr"],
    }, per_query_results


def evaluate_examples_for_k_values(
    examples: list[RetrievalEvalExample],
    retrieval_service: RetrievalService,
    *,
    k_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_k_values = _normalize_k_values(k_values)
    max_k = max(normalized_k_values)
    per_query_results: list[dict[str, Any]] = []
    aggregate_hits = {k: [] for k in normalized_k_values}
    aggregate_recalls = {k: [] for k in normalized_k_values}
    aggregate_precisions = {k: [] for k in normalized_k_values}
    aggregate_mrrs: list[float] = []

    for example in examples:
        retrieved_chunks = retrieval_service.retrieve(example.question, top_k=max_k)
        retrieved_sources = unique_ranked_sources([chunk.source for chunk in retrieved_chunks])
        retrieved_chunk_ids = [chunk.id for chunk in retrieved_chunks]
        has_expected_sources = bool(example.expected_source_documents)
        metrics_by_k: dict[str, dict[str, float]] = {}

        if has_expected_sources:
            mrr_score = mrr(retrieved_sources, example.expected_source_documents)
            first_rank = first_relevant_rank(retrieved_sources, example.expected_source_documents)

            aggregate_mrrs.append(mrr_score)
            evaluation_group = "retrieval_evaluated"
            for current_k in normalized_k_values:
                hit_score = hit_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    current_k,
                )
                recall_score = recall_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    current_k,
                )
                precision_score = precision_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    current_k,
                )
                aggregate_hits[current_k].append(hit_score)
                aggregate_recalls[current_k].append(recall_score)
                aggregate_precisions[current_k].append(precision_score)
                metrics_by_k[str(current_k)] = {
                    "hit_at_k": hit_score,
                    "recall_at_k": recall_score,
                    "precision_at_k": precision_score,
                }
        else:
            mrr_score = None
            first_rank = None
            evaluation_group = "no_expected_source"

        primary_metrics = metrics_by_k.get(str(max_k), {})
        per_query_results.append(
            {
                "id": example.id,
                "question": example.question,
                "expected_source_documents": list(example.expected_source_documents),
                "retrieved_sources": retrieved_sources,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "has_expected_sources": has_expected_sources,
                "evaluation_group": evaluation_group,
                "evaluated_k_values": normalized_k_values,
                "metrics_by_k": metrics_by_k,
                "hit_at_k": primary_metrics.get("hit_at_k"),
                "recall_at_k": primary_metrics.get("recall_at_k"),
                "precision_at_k": primary_metrics.get("precision_at_k"),
                "mrr": mrr_score,
                "first_relevant_rank": first_rank,
            }
        )

    num_queries_total = len(examples)
    num_queries_evaluated = len(aggregate_mrrs)
    num_queries_without_expected_sources = num_queries_total - num_queries_evaluated

    summary = {
        "num_queries_total": num_queries_total,
        "num_queries_evaluated": num_queries_evaluated,
        "num_queries_without_expected_source": num_queries_without_expected_sources,
        "num_queries_without_expected_sources": num_queries_without_expected_sources,
        "k": max_k,
        "k_values": normalized_k_values,
        "hit_at_k": _mean_or_none(aggregate_hits[max_k]),
        "recall_at_k": _mean_or_none(aggregate_recalls[max_k]),
        "mean_precision_at_k": _mean_or_none(aggregate_precisions[max_k]),
        "mrr": _mean_or_none(aggregate_mrrs),
        "metrics_by_k": {
            str(current_k): {
                "hit_at_k": _mean_or_none(aggregate_hits[current_k]),
                "recall_at_k": _mean_or_none(aggregate_recalls[current_k]),
                "mean_precision_at_k": _mean_or_none(aggregate_precisions[current_k]),
            }
            for current_k in normalized_k_values
        },
    }
    return summary, per_query_results


def build_run_config(
    *,
    settings: Settings,
    dataset_path: Path,
    top_k: int,
    timestamp: str,
    argv: list[str],
    run_name: str,
    notes: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    resolved_chunk_size = chunk_size
    if resolved_chunk_size is None:
        resolved_chunk_size = getattr(settings, "knowledge_chunk_size", None)

    resolved_chunk_overlap = chunk_overlap
    if resolved_chunk_overlap is None:
        resolved_chunk_overlap = getattr(settings, "knowledge_chunk_overlap", None)

    return {
        "timestamp": timestamp,
        "run_name": run_name,
        "notes": notes,
        "dataset_path": str(dataset_path),
        "retriever_type": settings.retriever_type,
        "top_k": top_k,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.knowledge_embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "vector_store_type": "pgvector" if settings.retriever_type in {"vector", "hybrid"} else None,
        "retrieval_strategy": settings.retriever_type,
        "chunk_size": resolved_chunk_size,
        "chunk_overlap": resolved_chunk_overlap,
        "settings_used_by_retriever": {
            "retriever_type": settings.retriever_type,
            "default_retrieval_config": settings.default_retrieval_config,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_min_similarity": settings.retrieval_min_similarity,
            "knowledge_collection_name": settings.knowledge_collection_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_dimension": settings.embedding_dimension,
            "vector_store_connection_scheme": settings.database_url.split(":", 1)[0],
        },
        "git_commit_sha": _git_commit_sha(),
        "python_command_used": subprocess.list2cmdline([sys.executable, *argv]),
    }


def create_output_directory(
    base_output_dir: Path,
    *,
    timestamp_label: str,
    run_label: str | None = None,
) -> Path:
    directory_name = f"{timestamp_label}_retrieval_baseline"
    if run_label:
        directory_name = f"{timestamp_label}_{_slugify_label(run_label)}"
    output_dir = base_output_dir / directory_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_artifacts(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json_path = output_dir / "results.json"
    results_csv_path = output_dir / "results.csv"
    config_path = output_dir / "config.json"

    results_json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "chunking": {
                    "chunk_size": config.get("chunk_size"),
                    "chunk_overlap": config.get("chunk_overlap"),
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_results_csv(results_csv_path, results)

    return {
        "results_json": results_json_path,
        "results_csv": results_csv_path,
        "config_json": config_path,
    }


def build_tracking_run_name(
    *,
    retriever_type: str,
    top_k: int,
    timestamp_label: str,
    experiment_name: str | None = None,
) -> str:
    if experiment_name and experiment_name.strip():
        return f"{experiment_name.strip()}-{timestamp_label}"
    return f"retrieval-{retriever_type}-k{top_k}-{timestamp_label}"


def log_run_to_tracker(
    *,
    tracker,
    settings: Settings,
    dataset_path: Path,
    top_k: int,
    summary: dict[str, Any],
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
    run_name: str,
) -> str | None:
    if not tracker.enabled:
        return None

    with tracker.run(run_name) as run:
        tracker.log_params(
            {
                "run_name": run_name,
                "notes": config.get("notes"),
                "dataset_name": dataset_path.name,
                "dataset_path": str(dataset_path),
                "retriever_type": settings.retriever_type,
                "retrieval_config": settings.default_retrieval_config,
                "top_k": top_k,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.knowledge_embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "knowledge_collection_name": settings.knowledge_collection_name,
                "chunk_size": config.get("chunk_size"),
                "chunk_overlap": config.get("chunk_overlap"),
                "retrieval_min_similarity": settings.retrieval_min_similarity,
                "git_commit_sha": config.get("git_commit_sha"),
            }
        )
        tracker.log_metrics(
            {
                "num_queries_total": summary["num_queries_total"],
                "num_queries_evaluated": summary["num_queries_evaluated"],
                "num_queries_without_expected_source": summary[
                    "num_queries_without_expected_source"
                ],
                "num_queries_without_expected_sources": summary[
                    "num_queries_without_expected_sources"
                ],
                "hit_at_k": summary["hit_at_k"],
                "recall_at_k": summary["recall_at_k"],
                "mean_precision_at_k": summary["mean_precision_at_k"],
                "mrr": summary["mrr"],
            }
        )
        for artifact_path in artifact_paths.values():
            tracker.log_artifact(artifact_path)
    return _extract_mlflow_run_id(run)


def run_retrieval_eval(
    *,
    settings: Settings,
    dataset_path: Path,
    output_root: Path,
    top_k: int,
    tracker,
    argv: list[str],
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    run_name: str | None = None,
    notes: str | None = None,
    output_label: str | None = None,
    timestamp: str | None = None,
    timestamp_label: str | None = None,
    examples: list[RetrievalEvalExample] | None = None,
    validation_summary: RetrievalDatasetValidationSummary | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RetrievalEvalRunResult:
    if top_k <= 0:
        raise ValueError("--k must be greater than 0")

    resolved_dataset_path = dataset_path.resolve()
    resolved_output_root = output_root.resolve()

    if examples is None:
        examples, validation_summary = load_and_validate_dataset(
            resolved_dataset_path,
            min_expected_source_coverage=min_expected_source_coverage,
        )
    elif validation_summary is None:
        validation_summary = validate_dataset_examples(
            examples,
            min_expected_source_coverage=min_expected_source_coverage,
        )

    assert validation_summary is not None

    run_started_at = datetime.now().astimezone()
    resolved_timestamp = timestamp or run_started_at.replace(microsecond=0).isoformat()
    resolved_timestamp_label = timestamp_label or run_started_at.strftime("%Y-%m-%d_%H%M%S")
    resolved_run_name = run_name or build_tracking_run_name(
        retriever_type=settings.retriever_type,
        top_k=top_k,
        timestamp_label=resolved_timestamp_label,
    )

    retrieval_service = RetrievalService(settings=settings)
    summary, results = evaluate_examples(examples, retrieval_service, k=top_k)
    config = build_run_config(
        settings=settings,
        dataset_path=resolved_dataset_path,
        top_k=top_k,
        timestamp=resolved_timestamp,
        argv=argv,
        run_name=resolved_run_name,
        notes=notes,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    run_output_dir = create_output_directory(
        resolved_output_root,
        timestamp_label=resolved_timestamp_label,
        run_label=output_label,
    )
    artifact_paths = write_artifacts(
        run_output_dir,
        summary=summary,
        results=results,
        config=config,
    )
    mlflow_run_id = log_run_to_tracker(
        tracker=tracker,
        settings=settings,
        dataset_path=resolved_dataset_path,
        top_k=top_k,
        summary=summary,
        config=config,
        artifact_paths=artifact_paths,
        run_name=resolved_run_name,
    )

    return RetrievalEvalRunResult(
        run_name=resolved_run_name,
        mlflow_run_id=mlflow_run_id,
        output_dir=run_output_dir,
        summary=summary,
        results=results,
        config=config,
        artifact_paths=artifact_paths,
        validation_summary=validation_summary,
    )


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "expected_source_documents",
        "retrieved_sources",
        "retrieved_chunk_ids",
        "has_expected_sources",
        "evaluation_group",
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "first_relevant_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {name: result.get(name) for name in fieldnames}
            row["expected_source_documents"] = json.dumps(
                result["expected_source_documents"],
                ensure_ascii=True,
            )
            row["retrieved_sources"] = json.dumps(result["retrieved_sources"], ensure_ascii=True)
            row["retrieved_chunk_ids"] = json.dumps(result["retrieved_chunk_ids"], ensure_ascii=True)
            writer.writerow(row)


def _git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    sha = completed.stdout.strip()
    return sha or None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _extract_mlflow_run_id(run: object | None) -> str | None:
    if run is None:
        return None

    info = getattr(run, "info", None)
    run_id = getattr(info, "run_id", None)
    return run_id if isinstance(run_id, str) and run_id else None


def _normalize_k_values(k_values: list[int]) -> list[int]:
    normalized = sorted(set(k_values))
    if not normalized:
        raise ValueError("At least one k value must be provided.")
    if any(k <= 0 for k in normalized):
        raise ValueError("All k values must be greater than 0.")
    return normalized


def _slugify_label(value: str) -> str:
    normalized = SAFE_LABEL_PATTERN.sub("-", value.strip().casefold()).strip("-")
    return normalized or "run"
