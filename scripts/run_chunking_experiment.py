from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.tracking import create_experiment_tracker
from app.knowledge.ingestion import ingest_knowledge, prepare_knowledge_ingestion_storage
from app.repositories.db.session import get_engine, get_session_factory
from app.services.retrieval import RetrievalService
from evals.runners.run_retrieval_eval import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    run_retrieval_eval,
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results" / "chunking_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestion and retrieval eval across multiple chunking configs.",
    )
    parser.add_argument(
        "--configs",
        required=True,
        help='Comma-separated chunk configs in the form "300:50,500:100".',
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the retrieval evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval top-k used during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where chunking experiment artifacts will be written.",
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


def parse_chunk_configs(raw_configs: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw_item in raw_configs.split(","):
        item = raw_item.strip()
        if not item:
            continue

        chunk_size_text, separator, chunk_overlap_text = item.partition(":")
        if separator != ":":
            raise ValueError(f"Invalid chunk config '{item}'. Expected format '<size>:<overlap>'.")

        try:
            chunk_size = int(chunk_size_text)
            chunk_overlap = int(chunk_overlap_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid chunk config '{item}'. Chunk values must be integers."
            ) from exc

        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk config '{item}'. chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError(
                f"Invalid chunk config '{item}'. chunk_overlap must be zero or positive."
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Invalid chunk config '{item}'. chunk_overlap must be smaller than chunk_size."
            )

        configs.append((chunk_size, chunk_overlap))

    if not configs:
        raise ValueError("At least one chunk config must be provided.")

    return configs


def build_comparison_rows(
    runs: list[dict[str, Any]],
    *,
    primary_metric: str = "recall_at_k",
) -> list[dict[str, Any]]:
    ranked_runs = sorted(
        runs,
        key=lambda item: (
            _sort_metric(item.get(primary_metric)),
            _sort_metric(item.get("mrr")),
            _sort_metric(item.get("precision_at_k")),
            _sort_metric(item.get("hit_at_k")),
        ),
        reverse=True,
    )
    best_value = max(
        (
            run.get(primary_metric)
            for run in ranked_runs
            if isinstance(run.get(primary_metric), (int, float))
        ),
        default=None,
    )

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(ranked_runs, start=1):
        row = dict(run)
        row["rank"] = index
        row["primary_metric"] = primary_metric
        row["is_best"] = best_value is not None and run.get(primary_metric) == best_value
        rows.append(row)
    return rows


def write_comparison_artifacts(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "chunking_experiment_summary.json"
    csv_path = output_dir / "chunking_experiment_summary.csv"
    table_path = output_dir / "chunking_experiment_ranking.md"

    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "ranking": {
            "primary_metric": "recall_at_k",
            "tiebreak_metrics": ["mrr", "precision_at_k", "hit_at_k"],
        },
        "best_configuration": next((row for row in rows if row["is_best"]), None),
        "runs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    fieldnames = [
        "rank",
        "is_best",
        "primary_metric",
        "chunk_size",
        "chunk_overlap",
        "documents_loaded",
        "chunks_indexed",
        "embedding_model",
        "retriever_type",
        "top_k",
        "query_rewriting",
        "reranker",
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr",
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

    table_path.write_text(format_ranked_summary_table(rows), encoding="utf-8")

    return {
        "summary_json": json_path,
        "summary_csv": csv_path,
        "ranking_md": table_path,
    }


def format_ranked_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "rank",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "retriever_type",
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
            str(row.get("chunk_size", "")),
            str(row.get("chunk_overlap", "")),
            str(row.get("embedding_model", "")),
            str(row.get("retriever_type", "")),
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


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be greater than 0")

    try:
        chunk_configs = parse_chunk_configs(args.configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset_path = args.dataset.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)
    prepare_knowledge_ingestion_storage(get_engine(use_direct=True))
    session_factory = get_session_factory(use_direct=True)
    examples, validation_summary = load_and_validate_dataset(
        dataset_path,
        min_expected_source_coverage=args.min_expected_source_coverage,
    )
    print(format_dataset_validation_summary(validation_summary))
    runs: list[dict[str, Any]] = []

    for chunk_size, chunk_overlap in chunk_configs:
        retrieval_service = RetrievalService(settings=settings)
        timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
        timestamp_label = datetime.now().strftime(
            f"%Y-%m-%d_%H%M%S_chunk_{chunk_size}_{chunk_overlap}"
        )

        with session_factory() as session:
            documents, ingestion_results = ingest_knowledge(
                session,
                retrieval_service,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        run_result = run_retrieval_eval(
            settings=settings,
            dataset_path=dataset_path,
            output_root=output_root,
            top_k=args.k,
            tracker=tracker,
            argv=sys.argv,
            run_name=f"chunking-{chunk_size}-{chunk_overlap}-{timestamp_label}",
            output_label=f"chunk_{chunk_size}_{chunk_overlap}",
            timestamp=timestamp,
            timestamp_label=timestamp_label,
            examples=examples,
            validation_summary=validation_summary,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks_indexed = sum(item.chunk_count for item in ingestion_results)

        runs.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "documents_loaded": len(documents),
                "chunks_indexed": chunks_indexed,
                "embedding_model": run_result.config.get("embedding_model"),
                "retriever_type": run_result.config.get("retriever_type"),
                "top_k": run_result.summary["k"],
                "query_rewriting": run_result.config.get("query_rewriting"),
                "reranker": run_result.config.get("reranker"),
                "hit_at_k": run_result.summary["hit_at_k"],
                "recall_at_k": run_result.summary["recall_at_k"],
                "precision_at_k": run_result.summary["precision_at_k"],
                "mrr": run_result.summary["mrr"],
                "mlflow_run_id": run_result.mlflow_run_id,
                "run_output_dir": str(run_result.output_dir),
                "results_json": str(run_result.artifact_paths["results_json"]),
                "results_csv": str(run_result.artifact_paths["results_csv"]),
                "config_json": str(run_result.artifact_paths["config_json"]),
            }
        )

    comparison_rows = build_comparison_rows(runs)
    comparison_paths = write_comparison_artifacts(output_root, rows=comparison_rows)
    _log_summary_to_tracker(
        tracker=tracker,
        run_name=f"chunking-experiment-summary-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        rows=comparison_rows,
        artifact_paths=comparison_paths,
        settings=settings,
        top_k=args.k,
    )
    best_row = comparison_rows[0]

    print()
    print("Best chunking configuration")
    print(
        f"rank={best_row['rank']} chunk_size={best_row['chunk_size']} "
        f"chunk_overlap={best_row['chunk_overlap']} "
        f"recall_at_k={_format_metric(best_row.get('recall_at_k'))} "
        f"mrr={_format_metric(best_row.get('mrr'))} "
        f"precision_at_k={_format_metric(best_row.get('precision_at_k'))}"
    )
    print()
    print(format_ranked_summary_table(comparison_rows), end="")
    print(f"Summary JSON written to: {comparison_paths['summary_json']}")
    print(f"Summary CSV written to: {comparison_paths['summary_csv']}")
    print(f"Ranking table written to: {comparison_paths['ranking_md']}")


def _sort_metric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


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
    settings,
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
                "workflow": "chunking_experiment",
                "config_count": len(rows),
                "embedding_model": settings.knowledge_embedding_model,
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
