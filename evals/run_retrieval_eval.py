from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import Settings, get_settings
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


@dataclass(frozen=True, slots=True)
class RetrievalEvalExample:
    id: str
    question: str
    expected_source_documents: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only baseline evaluation for the current vector retriever.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval top-k used during evaluation.",
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
        help="Directory where retrieval evaluation artifacts will be written.",
    )
    return parser.parse_args()


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


def evaluate_examples(
    examples: list[RetrievalEvalExample],
    retrieval_service: RetrievalService,
    *,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query_results: list[dict[str, Any]] = []
    aggregate_hits: list[float] = []
    aggregate_recalls: list[float] = []
    aggregate_precisions: list[float] = []
    aggregate_mrrs: list[float] = []

    for example in examples:
        retrieved_chunks = retrieval_service.retrieve(example.question, top_k=k)
        retrieved_sources = unique_ranked_sources([chunk.source for chunk in retrieved_chunks])
        retrieved_chunk_ids = [chunk.id for chunk in retrieved_chunks]
        has_expected_sources = bool(example.expected_source_documents)

        if has_expected_sources:
            hit_score = hit_at_k(retrieved_sources, example.expected_source_documents, k)
            recall_score = recall_at_k(retrieved_sources, example.expected_source_documents, k)
            precision_score = precision_at_k(retrieved_sources, example.expected_source_documents, k)
            mrr_score = mrr(retrieved_sources, example.expected_source_documents)
            first_rank = first_relevant_rank(retrieved_sources, example.expected_source_documents)

            aggregate_hits.append(hit_score)
            aggregate_recalls.append(recall_score)
            aggregate_precisions.append(precision_score)
            aggregate_mrrs.append(mrr_score)
            evaluation_group = "retrieval_evaluated"
        else:
            hit_score = None
            recall_score = None
            precision_score = None
            mrr_score = None
            first_rank = None
            evaluation_group = "no_expected_source"

        per_query_results.append(
            {
                "id": example.id,
                "question": example.question,
                "expected_source_documents": list(example.expected_source_documents),
                "retrieved_sources": retrieved_sources,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "has_expected_sources": has_expected_sources,
                "evaluation_group": evaluation_group,
                "hit_at_k": hit_score,
                "recall_at_k": recall_score,
                "precision_at_k": precision_score,
                "mrr": mrr_score,
                "first_relevant_rank": first_rank,
            }
        )

    num_queries_total = len(examples)
    num_queries_evaluated = len(aggregate_hits)
    num_queries_without_expected_sources = num_queries_total - num_queries_evaluated

    summary = {
        "num_queries_total": num_queries_total,
        "num_queries_evaluated": num_queries_evaluated,
        "num_queries_without_expected_sources": num_queries_without_expected_sources,
        "k": k,
        "hit_at_k": _mean_or_none(aggregate_hits),
        "recall_at_k": _mean_or_none(aggregate_recalls),
        "mean_precision_at_k": _mean_or_none(aggregate_precisions),
        "mrr": _mean_or_none(aggregate_mrrs),
    }
    return summary, per_query_results


def build_run_config(
    *,
    settings: Settings,
    dataset_path: Path,
    top_k: int,
    timestamp: str,
    argv: list[str],
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
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "embedding_model": settings.knowledge_embedding_model,
        "vector_store_type": "pgvector",
        "retrieval_strategy": "similarity_search_with_relevance_scores",
        "chunk_size": resolved_chunk_size,
        "chunk_overlap": resolved_chunk_overlap,
        "settings_used_by_retriever": {
            "default_retrieval_config": settings.default_retrieval_config,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_min_similarity": settings.retrieval_min_similarity,
            "knowledge_collection_name": settings.knowledge_collection_name,
            "vector_store_connection_scheme": settings.database_url.split(":", 1)[0],
        },
        "git_commit_sha": _git_commit_sha(),
        "python_command_used": subprocess.list2cmdline([sys.executable, *argv]),
    }


def create_output_directory(base_output_dir: Path, *, timestamp_label: str) -> Path:
    output_dir = base_output_dir / f"{timestamp_label}_retrieval_baseline"
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


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be greater than 0")

    dataset_path = args.dataset.resolve()
    output_root = args.output_dir.resolve()
    timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
    timestamp_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    settings = get_settings()
    retrieval_service = RetrievalService(settings=settings)
    examples = load_dataset(dataset_path)
    summary, results = evaluate_examples(examples, retrieval_service, k=args.k)
    config = build_run_config(
        settings=settings,
        dataset_path=dataset_path,
        top_k=args.k,
        timestamp=timestamp,
        argv=sys.argv,
    )

    run_output_dir = create_output_directory(output_root, timestamp_label=timestamp_label)
    artifact_paths = write_artifacts(
        run_output_dir,
        summary=summary,
        results=results,
        config=config,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Results JSON written to: {artifact_paths['results_json']}")
    print(f"Results CSV written to: {artifact_paths['results_csv']}")
    print(f"Config JSON written to: {artifact_paths['config_json']}")


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
            row = dict(result)
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


if __name__ == "__main__":
    main()
