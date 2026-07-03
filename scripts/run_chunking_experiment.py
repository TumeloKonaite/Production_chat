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
from app.knowledge.ingestion import ingest_knowledge, prepare_knowledge_ingestion_storage
from app.repositories.db.session import get_engine, get_session_factory
from app.services.retrieval import RetrievalService
from evals.run_retrieval_eval import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    build_run_config,
    create_output_directory,
    evaluate_examples,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    write_artifacts,
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
    primary_metric: str = "mrr",
) -> list[dict[str, Any]]:
    ranked_runs = sorted(
        runs,
        key=lambda item: (
            _sort_metric(item.get(primary_metric)),
            _sort_metric(item.get("hit_at_k")),
            _sort_metric(item.get("recall_at_k")),
            _sort_metric(item.get("mean_precision_at_k")),
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
    json_path = output_dir / "chunking_comparison.json"
    csv_path = output_dir / "chunking_comparison.csv"

    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "primary_metric": "mrr",
        "best_config": next((row for row in rows if row["is_best"]), None),
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
        "k",
        "hit_at_k",
        "recall_at_k",
        "mean_precision_at_k",
        "mrr",
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

    return {"comparison_json": json_path, "comparison_csv": csv_path}


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
    prepare_knowledge_ingestion_storage(get_engine())
    session_factory = get_session_factory()
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

        summary, results = evaluate_examples(examples, retrieval_service, k=args.k)
        config = build_run_config(
            settings=settings,
            dataset_path=dataset_path,
            top_k=args.k,
            timestamp=timestamp,
            argv=sys.argv,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        config["documents_loaded"] = len(documents)
        config["chunks_indexed"] = sum(item.chunk_count for item in ingestion_results)

        run_output_dir = create_output_directory(output_root, timestamp_label=timestamp_label)
        artifact_paths = write_artifacts(
            run_output_dir,
            summary=summary,
            results=results,
            config=config,
        )

        runs.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "documents_loaded": len(documents),
                "chunks_indexed": sum(item.chunk_count for item in ingestion_results),
                "k": summary["k"],
                "hit_at_k": summary["hit_at_k"],
                "recall_at_k": summary["recall_at_k"],
                "mean_precision_at_k": summary["mean_precision_at_k"],
                "mrr": summary["mrr"],
                "run_output_dir": str(run_output_dir),
                "results_json": str(artifact_paths["results_json"]),
                "results_csv": str(artifact_paths["results_csv"]),
                "config_json": str(artifact_paths["config_json"]),
            }
        )

    comparison_rows = build_comparison_rows(runs)
    comparison_paths = write_comparison_artifacts(output_root, rows=comparison_rows)

    print(json.dumps(comparison_rows, indent=2, ensure_ascii=True))
    print(f"Comparison JSON written to: {comparison_paths['comparison_json']}")
    print(f"Comparison CSV written to: {comparison_paths['comparison_csv']}")


def _sort_metric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


if __name__ == "__main__":
    main()
