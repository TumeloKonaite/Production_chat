from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import Settings, get_settings
from app.infrastructure.tracking import create_experiment_tracker
from app.knowledge.ingestion import ingest_knowledge, prepare_knowledge_ingestion_storage
from app.repositories.db.session import get_engine, get_session_factory
from app.services.retrieval import RetrievalService
from evals.runners.run_retrieval_eval import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    build_run_config,
    evaluate_examples_for_k_values,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    log_run_to_tracker,
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results" / "embedding_experiments"
SAFE_LABEL_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class EmbeddingRunConfig:
    provider: str
    model: str
    dimension: int


@dataclass(frozen=True, slots=True)
class EmbeddingExperimentConfig:
    embedding_runs: list[EmbeddingRunConfig]
    k_values: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only evaluation across multiple embedding setups.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the embedding experiment config JSON file.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the retrieval evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where embedding experiment artifacts will be written.",
    )
    parser.add_argument(
        "--min-expected-source-coverage",
        type=float,
        default=DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
        help=(
            "Minimum fraction of dataset rows that must include expected_source_documents "
            "before evaluation runs."
        ),
    )
    return parser.parse_args()


def load_embedding_experiment_config(path: Path) -> EmbeddingExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Embedding experiment config must be a JSON object.")

    raw_embedding_runs = payload.get("embedding_runs")
    if not isinstance(raw_embedding_runs, list) or not raw_embedding_runs:
        raise ValueError("embedding_runs must be a non-empty list.")

    embedding_runs: list[EmbeddingRunConfig] = []
    seen_run_keys: set[tuple[str, str, int]] = set()
    for index, raw_run in enumerate(raw_embedding_runs, start=1):
        if not isinstance(raw_run, dict):
            raise ValueError(f"embedding_runs[{index}] must be an object.")

        provider = raw_run.get("provider")
        model = raw_run.get("model")
        dimension = raw_run.get("dimension")

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"embedding_runs[{index}].provider must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"embedding_runs[{index}].model must be a non-empty string.")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"embedding_runs[{index}].dimension must be a positive integer.")

        run_config = EmbeddingRunConfig(
            provider=provider.strip().casefold(),
            model=model.strip(),
            dimension=dimension,
        )
        run_key = (run_config.provider, run_config.model, run_config.dimension)
        if run_key in seen_run_keys:
            raise ValueError(
                "Embedding experiment config contains duplicate runs for "
                f"{run_config.provider}/{run_config.model}/{run_config.dimension}."
            )
        seen_run_keys.add(run_key)
        embedding_runs.append(run_config)

    raw_k_values = payload.get("k_values")
    if not isinstance(raw_k_values, list):
        raise ValueError("k_values must be a non-empty list of positive integers.")

    k_values = _normalize_k_values(raw_k_values)
    return EmbeddingExperimentConfig(embedding_runs=embedding_runs, k_values=k_values)


def run_embedding_experiment_matrix(
    *,
    experiment_config: EmbeddingExperimentConfig,
    experiment_config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    settings: Settings,
    argv: list[str],
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)
    database_vector_dimension = _get_database_vector_store_dimension()
    examples, validation_summary = load_and_validate_dataset(
        dataset_path,
        min_expected_source_coverage=min_expected_source_coverage,
    )
    print(format_dataset_validation_summary(validation_summary))
    session_factory = get_session_factory()
    runs_output_dir = output_dir / "runs"
    runs_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []

    for index, embedding_run in enumerate(experiment_config.embedding_runs, start=1):
        run_settings = replace(
            settings,
            embedding_provider=embedding_run.provider,
            knowledge_embedding_model=embedding_run.model,
            embedding_dimension=embedding_run.dimension,
        )
        _validate_vector_store_dimension(
            embedding_run=embedding_run,
            database_vector_dimension=database_vector_dimension,
        )
        retrieval_service = RetrievalService(settings=run_settings)
        timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
        timestamp_label = _build_run_timestamp_label(index=index, embedding_run=embedding_run)

        with session_factory() as session:
            documents, ingestion_results = ingest_knowledge(
                session,
                retrieval_service,
                chunk_size=run_settings.knowledge_chunk_size,
                chunk_overlap=run_settings.knowledge_chunk_overlap,
            )

        summary, results = evaluate_examples_for_k_values(
            examples,
            retrieval_service,
            k_values=experiment_config.k_values,
        )
        chunks_indexed = sum(item.chunk_count for item in ingestion_results)
        config = build_run_config(
            settings=run_settings,
            dataset_path=dataset_path,
            top_k=max(experiment_config.k_values),
            timestamp=timestamp,
            argv=argv,
            run_name=timestamp_label,
            chunk_size=run_settings.knowledge_chunk_size,
            chunk_overlap=run_settings.knowledge_chunk_overlap,
        )
        config["k_values"] = list(experiment_config.k_values)
        config["embedding_run_index"] = index
        config["experiment_config_path"] = str(experiment_config_path)
        config["documents_loaded"] = len(documents)
        config["chunks_indexed"] = chunks_indexed

        run_output_dir = runs_output_dir / timestamp_label
        artifact_paths = write_embedding_run_artifacts(
            run_output_dir,
            summary=summary,
            results=results,
            config=config,
            k_values=experiment_config.k_values,
        )
        mlflow_run_id = log_run_to_tracker(
            tracker=tracker,
            settings=run_settings,
            dataset_path=dataset_path,
            top_k=max(experiment_config.k_values),
            summary=summary,
            config=config,
            artifact_paths=artifact_paths,
            run_name=timestamp_label,
        )

        comparison_rows.append(
            build_embedding_result_row(
                embedding_run=embedding_run,
                summary=summary,
                config=config,
                documents_loaded=len(documents),
                chunks_indexed=chunks_indexed,
                run_output_dir=run_output_dir,
                artifact_paths=artifact_paths,
                mlflow_run_id=mlflow_run_id,
            )
        )

    ranked_rows = build_embedding_comparison_rows(
        comparison_rows,
        k_values=experiment_config.k_values,
    )
    manifest_path = write_experiment_manifest(
        output_dir,
        experiment_config=experiment_config,
        experiment_config_path=experiment_config_path,
        dataset_path=dataset_path,
        argv=argv,
        settings=settings,
    )
    comparison_paths = write_embedding_comparison_artifacts(
        output_dir,
        rows=ranked_rows,
        k_values=experiment_config.k_values,
        experiment_manifest_path=manifest_path,
    )
    _log_summary_to_tracker(
        tracker=tracker,
        run_name=f"embedding-experiment-summary-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        rows=ranked_rows,
        artifact_paths=comparison_paths,
        settings=settings,
        top_k=max(experiment_config.k_values),
    )
    return ranked_rows, comparison_paths


def build_embedding_result_row(
    *,
    embedding_run: EmbeddingRunConfig,
    summary: dict[str, Any],
    config: dict[str, Any],
    documents_loaded: int,
    chunks_indexed: int,
    run_output_dir: Path,
    artifact_paths: dict[str, Path],
    mlflow_run_id: str | None,
) -> dict[str, Any]:
    top_k = int(summary.get("k", max(summary.get("k_values", [1]))))
    row = {
        "embedding_provider": embedding_run.provider,
        "embedding_model": embedding_run.model,
        "embedding_dimension": embedding_run.dimension,
        "chunk_size": config.get("chunk_size"),
        "chunk_overlap": config.get("chunk_overlap"),
        "retriever_type": config.get("retriever_type"),
        "top_k": top_k,
        "query_rewriting": config.get("query_rewriting"),
        "reranker": config.get("reranker"),
        "documents_loaded": documents_loaded,
        "chunks_indexed": chunks_indexed,
        "k_values": list(summary.get("k_values", [])),
        "recall_at_k": summary.get("recall_at_k"),
        "precision_at_k": summary.get("precision_at_k"),
        "mrr": summary.get("mrr"),
        "mlflow_run_id": mlflow_run_id,
        "run_output_dir": str(run_output_dir),
        "results_json": str(artifact_paths["results_json"]),
        "results_csv": str(artifact_paths["results_csv"]),
        "config_json": str(artifact_paths["config_json"]),
    }
    metrics_by_k = summary.get("metrics_by_k", {})
    if isinstance(metrics_by_k, dict):
        for raw_k, metrics in metrics_by_k.items():
            if not isinstance(metrics, dict):
                continue
            row[f"recall_at_{raw_k}"] = metrics.get("recall_at_k")
    return row


def build_embedding_comparison_rows(
    runs: list[dict[str, Any]],
    *,
    k_values: list[int],
) -> list[dict[str, Any]]:
    ranking_k_values = _build_ranking_k_values(k_values)
    ranked_runs = sorted(
        runs,
        key=lambda item: _comparison_sort_key(item, ranking_k_values),
    )

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(ranked_runs, start=1):
        row = dict(run)
        row["rank"] = index
        row["ranking_k_values"] = ranking_k_values
        row["is_best"] = index == 1
        rows.append(row)
    return rows


def write_embedding_run_artifacts(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    config: dict[str, Any],
    k_values: list[int],
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
    _write_embedding_results_csv(results_csv_path, results, k_values=k_values)
    return {
        "results_json": results_json_path,
        "results_csv": results_csv_path,
        "config_json": config_path,
    }


def write_embedding_comparison_artifacts(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    k_values: list[int],
    experiment_manifest_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "embedding_experiment_summary.json"
    csv_path = output_dir / "embedding_experiment_summary.csv"
    table_path = output_dir / "embedding_experiment_ranking.md"

    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "best_embedding_setup": next((row for row in rows if row["is_best"]), None),
        "ranking": {
            "primary_metric": "recall_at_k",
            "tiebreak_metrics": ["mrr", "precision_at_k", *[f"recall_at_{k}" for k in _build_ranking_k_values(k_values)]],
        },
        "experiment_manifest_path": str(experiment_manifest_path),
        "runs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    fieldnames = [
        "rank",
        "is_best",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "chunk_size",
        "chunk_overlap",
        "retriever_type",
        "top_k",
        "query_rewriting",
        "reranker",
        "documents_loaded",
        "chunks_indexed",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        *[f"recall_at_{k}" for k in k_values],
        "mlflow_run_id",
        "run_output_dir",
        "results_json",
        "results_csv",
        "config_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    table_path.write_text(format_embedding_summary_table(rows), encoding="utf-8")

    return {
        "summary_json": json_path,
        "summary_csv": csv_path,
        "ranking_md": table_path,
    }


def write_experiment_manifest(
    output_dir: Path,
    *,
    experiment_config: EmbeddingExperimentConfig,
    experiment_config_path: Path,
    dataset_path: Path,
    argv: list[str],
    settings: Settings,
) -> Path:
    manifest_path = output_dir / "experiment_manifest.json"
    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "experiment_config_path": str(experiment_config_path),
        "dataset_path": str(dataset_path),
        "embedding_runs": [
            {
                "provider": run.provider,
                "model": run.model,
                "dimension": run.dimension,
            }
            for run in experiment_config.embedding_runs
        ],
        "k_values": list(experiment_config.k_values),
        "chunk_size": settings.knowledge_chunk_size,
        "chunk_overlap": settings.knowledge_chunk_overlap,
        "python_command_used": str(sys.executable) + " " + " ".join(argv),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    config_path = args.config.resolve()
    base_output_dir = args.output_dir.resolve()
    timestamp_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_output_dir = base_output_dir / f"{timestamp_label}_embedding_matrix"

    try:
        experiment_config = load_embedding_experiment_config(config_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    settings = get_settings()
    prepare_knowledge_ingestion_storage(get_engine())
    try:
        rows, artifact_paths = run_embedding_experiment_matrix(
            experiment_config=experiment_config,
            experiment_config_path=config_path,
            dataset_path=dataset_path,
            output_dir=experiment_output_dir,
            settings=settings,
            argv=sys.argv,
            min_expected_source_coverage=args.min_expected_source_coverage,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    best_row = rows[0]
    best_recall_k = 5 if 5 in experiment_config.k_values else max(experiment_config.k_values)
    print(
        "Best embedding setup: "
        f"{best_row['embedding_provider']}/{best_row['embedding_model']}"
    )
    print(f"Recall@{best_recall_k}: {best_row.get(f'recall_at_{best_recall_k}')}")
    print(f"MRR: {best_row['mrr']}")
    print(f"Precision@{best_row['top_k']}: {best_row.get('precision_at_k')}")
    print()
    print(format_embedding_summary_table(rows), end="")
    print(f"Summary JSON written to: {artifact_paths['summary_json']}")
    print(f"Summary CSV written to: {artifact_paths['summary_csv']}")
    print(f"Ranking table written to: {artifact_paths['ranking_md']}")


def _write_embedding_results_csv(
    path: Path,
    results: list[dict[str, Any]],
    *,
    k_values: list[int],
) -> None:
    metric_fieldnames: list[str] = []
    for current_k in k_values:
        metric_fieldnames.extend(
            [
                f"hit_at_{current_k}",
                f"recall_at_{current_k}",
                f"precision_at_{current_k}",
            ]
        )

    fieldnames = [
        "id",
        "question",
        "expected_source_documents",
        "retrieved_sources",
        "retrieved_chunk_ids",
        "has_expected_sources",
        "evaluation_group",
        *metric_fieldnames,
        "mrr",
        "first_relevant_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "id": result.get("id"),
                "question": result.get("question"),
                "expected_source_documents": json.dumps(
                    result.get("expected_source_documents", []),
                    ensure_ascii=True,
                ),
                "retrieved_sources": json.dumps(result.get("retrieved_sources", []), ensure_ascii=True),
                "retrieved_chunk_ids": json.dumps(
                    result.get("retrieved_chunk_ids", []),
                    ensure_ascii=True,
                ),
                "has_expected_sources": result.get("has_expected_sources"),
                "evaluation_group": result.get("evaluation_group"),
                "mrr": result.get("mrr"),
                "first_relevant_rank": result.get("first_relevant_rank"),
            }
            metrics_by_k = result.get("metrics_by_k", {})
            if isinstance(metrics_by_k, dict):
                for current_k in k_values:
                    metrics = metrics_by_k.get(str(current_k), {})
                    row[f"hit_at_{current_k}"] = metrics.get("hit_at_k")
                    row[f"recall_at_{current_k}"] = metrics.get("recall_at_k")
                    row[f"precision_at_{current_k}"] = metrics.get("precision_at_k")
            writer.writerow(row)


def _build_run_timestamp_label(*, index: int, embedding_run: EmbeddingRunConfig) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    provider_label = _slugify_label(embedding_run.provider)
    model_label = _slugify_label(embedding_run.model)
    return f"{timestamp}_run_{index:02d}_{provider_label}_{model_label}_{embedding_run.dimension}"


def _build_ranking_k_values(k_values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for preferred_k in (5, 3, 1):
        if preferred_k in k_values and preferred_k not in seen:
            ordered.append(preferred_k)
            seen.add(preferred_k)
    for current_k in sorted(k_values, reverse=True):
        if current_k in seen:
            continue
        ordered.append(current_k)
        seen.add(current_k)
    return ordered


def _comparison_sort_key(run: dict[str, Any], ranking_k_values: list[int]) -> tuple[Any, ...]:
    metric_keys = ["recall_at_k", "mrr", "precision_at_k", *[f"recall_at_{k}" for k in ranking_k_values]]
    return (
        *[_descending_metric_key(run.get(metric_key)) for metric_key in metric_keys],
        str(run.get("embedding_provider", "")),
        str(run.get("embedding_model", "")),
        int(run.get("embedding_dimension", 0)),
    )


def _descending_metric_key(value: Any) -> tuple[int, float]:
    if isinstance(value, (int, float)):
        return (0, -float(value))
    return (1, 0.0)


def _normalize_k_values(raw_k_values: list[Any]) -> list[int]:
    if not raw_k_values:
        raise ValueError("k_values must be a non-empty list of positive integers.")

    normalized: set[int] = set()
    for index, value in enumerate(raw_k_values, start=1):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"k_values[{index}] must be a positive integer.")
        normalized.add(value)

    return sorted(normalized)


def _validate_vector_store_dimension(
    *,
    embedding_run: EmbeddingRunConfig,
    database_vector_dimension: int | None,
) -> None:
    if (
        database_vector_dimension is None
        or database_vector_dimension == embedding_run.dimension
    ):
        return

    raise ValueError(
        "Database vector dimension mismatch for embedding experiment run "
        f"{embedding_run.provider}/{embedding_run.model}. "
        f"Configured dimension: {embedding_run.dimension}. "
        f"Database vector store dimension: {database_vector_dimension}. "
        "Update the embedding experiment config to match the pgvector schema, or run the "
        "required database migration and rebuild the knowledge index before comparing this model."
    )


def _get_database_vector_store_dimension() -> int | None:
    query = text(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
        FROM pg_attribute AS a
        JOIN pg_class AS c
          ON a.attrelid = c.oid
        JOIN pg_namespace AS n
          ON c.relnamespace = n.oid
        WHERE c.relname = 'langchain_pg_embedding'
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END, n.nspname
        LIMIT 1
        """
    )
    try:
        with get_engine().connect() as connection:
            embedding_type = connection.execute(query).scalar_one_or_none()
    except Exception:
        return None

    if not isinstance(embedding_type, str):
        return None

    match = re.fullmatch(r"vector\((\d+)\)", embedding_type.strip())
    if match is None:
        return None
    return int(match.group(1))


def _slugify_label(value: str) -> str:
    normalized = SAFE_LABEL_PATTERN.sub("_", value.strip().casefold()).strip("_")
    return normalized or "value"


def format_embedding_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "rank",
        "embedding_provider",
        "embedding_model",
        "top_k",
        "query_rewriting",
        "reranker",
        "recall_at_k",
        "mrr",
        "precision_at_k",
    ]
    rendered_rows = [
        [
            str(row.get("rank", "")),
            str(row.get("embedding_provider", "")),
            str(row.get("embedding_model", "")),
            str(row.get("top_k", "")),
            str(row.get("query_rewriting", "")),
            str(row.get("reranker", "")),
            _format_metric(row.get("recall_at_k")),
            _format_metric(row.get("mrr")),
            _format_metric(row.get("precision_at_k")),
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(rendered_row[index]) for rendered_row in rendered_rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(rendered_row) for rendered_row in rendered_rows)
    return "\n".join(lines) + "\n"


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return ""


def _log_summary_to_tracker(
    *,
    tracker,
    run_name: str,
    rows: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
    settings: Settings,
    top_k: int,
) -> None:
    if (
        not tracker.enabled
        or not rows
        or not hasattr(tracker, "run")
        or not hasattr(tracker, "log_params")
        or not hasattr(tracker, "log_metrics")
        or not hasattr(tracker, "log_artifact")
    ):
        return

    best_row = rows[0]
    with tracker.run(run_name):
        tracker.log_params(
            {
                "workflow": "embedding_experiment",
                "config_count": len(rows),
                "chunk_size": settings.knowledge_chunk_size,
                "chunk_overlap": settings.knowledge_chunk_overlap,
                "retriever_type": settings.retriever_type,
                "top_k": top_k,
                "query_rewriting": settings.enable_query_rewriting,
                "reranker": (
                    settings.reranker_type if getattr(settings, "enable_reranking", False) else "none"
                ),
            }
        )
        tracker.log_metrics(
            {
                "best_recall_at_k": float(best_row.get("recall_at_k") or 0.0),
                "best_mrr": float(best_row.get("mrr") or 0.0),
                "best_precision_at_k": float(best_row.get("precision_at_k") or 0.0),
            }
        )
        for artifact_path in artifact_paths.values():
            tracker.log_artifact(artifact_path)


if __name__ == "__main__":
    main()
