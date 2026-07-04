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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import SUPPORTED_RETRIEVER_TYPES, Settings, get_settings
from app.infrastructure.tracking import create_experiment_tracker
from evals.retrieval_eval_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    RetrievalEvalRunResult,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    run_retrieval_eval,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run retrieval sweeps.") from exc

DEFAULT_SWEEP_OUTPUT_DIR = ROOT_DIR / "evals" / "results" / "retrieval_sweeps"
SAFE_LABEL_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrievalSweepExperiment:
    name: str
    retriever_type: str
    top_k: int
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    reranker_enabled: bool = False
    reranker_type: str = "none"
    reranker_model: str | None = None
    reranker_initial_top_k: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalSweepConfig:
    experiments: list[RetrievalSweepExperiment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation sweeps across multiple retriever configurations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the retrieval sweep YAML config file.",
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
        default=DEFAULT_SWEEP_OUTPUT_DIR,
        help="Directory where retrieval sweep artifacts will be written.",
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


def load_retrieval_sweep_config(path: Path) -> RetrievalSweepConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Retrieval sweep config must be a YAML object.")

    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("experiments must be a non-empty list.")

    experiments: list[RetrievalSweepExperiment] = []
    seen_names: set[str] = set()
    for index, raw_experiment in enumerate(raw_experiments, start=1):
        if not isinstance(raw_experiment, dict):
            raise ValueError(f"experiments[{index}] must be an object.")

        name = raw_experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"experiments[{index}].name must be a non-empty string.")
        normalized_name = name.strip()
        if normalized_name in seen_names:
            raise ValueError(
                f"Retrieval sweep config contains duplicate experiment name: {normalized_name}."
            )
        seen_names.add(normalized_name)

        retriever_type = raw_experiment.get("retriever_type")
        if not isinstance(retriever_type, str) or not retriever_type.strip():
            raise ValueError(f"experiments[{index}].retriever_type must be a non-empty string.")
        normalized_retriever_type = retriever_type.strip().casefold()
        if normalized_retriever_type not in SUPPORTED_RETRIEVER_TYPES:
            supported_values = ", ".join(sorted(SUPPORTED_RETRIEVER_TYPES))
            raise ValueError(
                f"experiments[{index}].retriever_type must be one of: {supported_values}."
            )

        top_k = raw_experiment.get("top_k")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"experiments[{index}].top_k must be a positive integer.")

        embedding_provider = _optional_string(
            raw_experiment.get("embedding_provider"),
            field_name=f"experiments[{index}].embedding_provider",
            normalize_case=True,
        )
        embedding_model = _optional_string(
            raw_experiment.get("embedding_model"),
            field_name=f"experiments[{index}].embedding_model",
        )
        embedding_dimension = _optional_positive_int(
            raw_experiment.get("embedding_dimension"),
            field_name=f"experiments[{index}].embedding_dimension",
        )
        chunk_size = _optional_positive_int(
            raw_experiment.get("chunk_size"),
            field_name=f"experiments[{index}].chunk_size",
        )
        chunk_overlap = _optional_non_negative_int(
            raw_experiment.get("chunk_overlap"),
            field_name=f"experiments[{index}].chunk_overlap",
        )
        reranker_enabled = _optional_bool(
            raw_experiment.get("reranker_enabled"),
            field_name=f"experiments[{index}].reranker_enabled",
            default=False,
        )
        reranker_type = _optional_string(
            raw_experiment.get("reranker_type"),
            field_name=f"experiments[{index}].reranker_type",
            normalize_case=True,
        ) or ("llm" if reranker_enabled else "none")
        if reranker_type not in {"none", "llm"}:
            raise ValueError(
                f"experiments[{index}].reranker_type must be one of: llm, none."
            )
        reranker_model = _optional_string(
            raw_experiment.get("reranker_model"),
            field_name=f"experiments[{index}].reranker_model",
        )
        reranker_initial_top_k = _optional_positive_int(
            raw_experiment.get("reranker_initial_top_k"),
            field_name=f"experiments[{index}].reranker_initial_top_k",
        )
        if reranker_enabled and reranker_type == "none":
            raise ValueError(
                f"experiments[{index}].reranker_type must not be none when reranker_enabled is true."
            )
        if chunk_size is not None and chunk_overlap is not None and chunk_overlap >= chunk_size:
            raise ValueError(
                f"experiments[{index}].chunk_overlap must be smaller than chunk_size."
            )

        experiments.append(
            RetrievalSweepExperiment(
                name=normalized_name,
                retriever_type=normalized_retriever_type,
                top_k=top_k,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reranker_enabled=reranker_enabled,
                reranker_type=reranker_type,
                reranker_model=reranker_model,
                reranker_initial_top_k=reranker_initial_top_k,
            )
        )

    return RetrievalSweepConfig(experiments=experiments)


def run_retrieval_sweep(
    *,
    sweep_config: RetrievalSweepConfig,
    sweep_config_path: Path,
    dataset_path: Path,
    output_dir: Path,
    settings: Settings,
    argv: list[str],
    min_expected_source_coverage: float = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    resolved_dataset_path = dataset_path.resolve()
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    runs_output_dir = resolved_output_dir / "runs"
    runs_output_dir.mkdir(parents=True, exist_ok=True)

    examples, validation_summary = load_and_validate_dataset(
        resolved_dataset_path,
        min_expected_source_coverage=min_expected_source_coverage,
    )
    print(format_dataset_validation_summary(validation_summary))

    tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)
    comparison_rows: list[dict[str, Any]] = []
    run_results: list[RetrievalEvalRunResult] = []

    for index, experiment in enumerate(sweep_config.experiments, start=1):
        experiment_settings = build_experiment_settings(settings=settings, experiment=experiment)
        timestamp_label = _build_sweep_run_timestamp_label(index=index, experiment_name=experiment.name)
        run_result = run_retrieval_eval(
            settings=experiment_settings,
            dataset_path=resolved_dataset_path,
            output_root=runs_output_dir,
            top_k=experiment.top_k,
            tracker=tracker,
            argv=argv,
            min_expected_source_coverage=min_expected_source_coverage,
            run_name=build_tracking_run_name_for_experiment(
                experiment=experiment,
                timestamp_label=timestamp_label,
            ),
            output_label=f"run_{index:02d}_{experiment.name}",
            timestamp_label=timestamp_label,
            examples=examples,
            validation_summary=validation_summary,
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
        )
        run_results.append(run_result)
        comparison_rows.append(build_sweep_result_row(experiment=experiment, run_result=run_result))

    ranked_rows = rank_sweep_rows(comparison_rows)
    manifest_path = write_retrieval_sweep_manifest(
        output_dir=resolved_output_dir,
        sweep_config=sweep_config,
        sweep_config_path=sweep_config_path.resolve(),
        dataset_path=resolved_dataset_path,
        settings=settings,
        argv=argv,
    )
    artifact_paths = write_retrieval_sweep_comparison_artifacts(
        output_dir=resolved_output_dir,
        rows=ranked_rows,
        manifest_path=manifest_path,
    )
    _log_summary_to_tracker(
        tracker=tracker,
        run_name=f"retrieval-sweep-summary-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        rows=ranked_rows,
        artifact_paths=artifact_paths,
        settings=settings,
    )
    return ranked_rows, artifact_paths


def build_experiment_settings(
    *,
    settings: Settings,
    experiment: RetrievalSweepExperiment,
) -> Settings:
    chunk_size = experiment.chunk_size or settings.knowledge_chunk_size
    chunk_overlap = (
        experiment.chunk_overlap
        if experiment.chunk_overlap is not None
        else settings.knowledge_chunk_overlap
    )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Experiment {experiment.name} has chunk_overlap >= chunk_size, which is invalid."
        )

    return replace(
        settings,
        retriever_type=experiment.retriever_type,
        retrieval_top_k=experiment.top_k,
        embedding_provider=experiment.embedding_provider or settings.embedding_provider,
        knowledge_embedding_model=experiment.embedding_model or settings.knowledge_embedding_model,
        embedding_dimension=experiment.embedding_dimension or settings.embedding_dimension,
        knowledge_chunk_size=chunk_size,
        knowledge_chunk_overlap=chunk_overlap,
        enable_reranking=experiment.reranker_enabled,
        reranker_type=experiment.reranker_type,
        reranker_model=experiment.reranker_model or settings.reranker_model,
        reranker_initial_top_k=experiment.reranker_initial_top_k or settings.reranker_initial_top_k,
        reranker_final_top_k=experiment.top_k,
    )


def build_tracking_run_name_for_experiment(
    *,
    experiment: RetrievalSweepExperiment,
    timestamp_label: str,
) -> str:
    return f"{experiment.name}-{timestamp_label}"


def build_sweep_result_row(
    *,
    experiment: RetrievalSweepExperiment,
    run_result: RetrievalEvalRunResult,
) -> dict[str, Any]:
    summary = run_result.summary
    return {
        "run_name": run_result.run_name,
        "experiment_name": experiment.name,
        "chunk_size": run_result.config.get("chunk_size"),
        "chunk_overlap": run_result.config.get("chunk_overlap"),
        "retriever_type": experiment.retriever_type,
        "top_k": experiment.top_k,
        "embedding_provider": run_result.config.get("embedding_provider"),
        "embedding_model": run_result.config.get("embedding_model"),
        "embedding_dimension": run_result.config.get("embedding_dimension"),
        "query_rewriting": run_result.config.get("query_rewriting"),
        "query_rewriting_enabled": run_result.config.get("query_rewriting_enabled"),
        "query_rewrite_model": run_result.config.get("query_rewrite_model"),
        "query_rewrite_prompt_version": run_result.config.get("query_rewrite_prompt_version"),
        "reranker": run_result.config.get("reranker"),
        "reranker_enabled": run_result.config.get("reranker_enabled"),
        "reranker_type": run_result.config.get("reranker_type"),
        "reranker_model": run_result.config.get("reranker_model"),
        "reranker_initial_top_k": run_result.config.get("reranker_initial_top_k"),
        "reranker_final_top_k": run_result.config.get("reranker_final_top_k"),
        "dataset_path": run_result.config.get("dataset_path"),
        "git_commit_sha": run_result.config.get("git_commit_sha"),
        "mrr": summary.get("mrr"),
        "recall_at_k": summary.get("recall_at_k"),
        "precision_at_k": summary.get("precision_at_k", summary.get("mean_precision_at_k")),
        "mean_precision_at_k": summary.get("mean_precision_at_k"),
        "hit_at_k": summary.get("hit_at_k"),
        "context_relevance": summary.get("context_relevance"),
        "num_queries_total": summary.get("num_queries_total"),
        "num_queries_evaluated": summary.get("num_queries_evaluated"),
        "num_queries_without_expected_source": summary.get("num_queries_without_expected_source"),
        "query_rewrite_avg_latency_ms": summary.get("query_rewrite_avg_latency_ms"),
        "query_rewrite_total_latency_ms": summary.get("query_rewrite_total_latency_ms"),
        "query_rewrite_success_count": summary.get("query_rewrite_success_count"),
        "query_rewrite_fallback_count": summary.get("query_rewrite_fallback_count"),
        "query_rewrite_failure_count": summary.get("query_rewrite_failure_count"),
        "query_rewrite_total_tokens": summary.get("query_rewrite_total_tokens"),
        "query_rewrite_estimated_total_cost": summary.get("query_rewrite_estimated_total_cost"),
        "mlflow_run_id": run_result.mlflow_run_id,
        "output_dir": str(run_result.output_dir),
        "results_json": str(run_result.artifact_paths["results_json"]),
        "results_csv": str(run_result.artifact_paths["results_csv"]),
        "config_json": str(run_result.artifact_paths["config_json"]),
    }


def rank_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_runs = sorted(
        rows,
        key=lambda item: (
            _sort_metric(item.get("recall_at_k")),
            _sort_metric(item.get("mrr")),
            _sort_metric(item.get("precision_at_k")),
            _sort_metric(item.get("hit_at_k")),
        ),
        reverse=True,
    )
    best_row = ranked_runs[0] if ranked_runs else None
    ranked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ranked_runs, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = index
        ranked_row["is_best"] = best_row is not None and row is best_row
        ranked_rows.append(ranked_row)
    return ranked_rows


def write_retrieval_sweep_manifest(
    output_dir: Path,
    *,
    sweep_config: RetrievalSweepConfig,
    sweep_config_path: Path,
    dataset_path: Path,
    settings: Settings,
    argv: list[str],
) -> Path:
    manifest_path = output_dir / "sweep_manifest.json"
    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "sweep_config_path": str(sweep_config_path),
        "dataset_path": str(dataset_path),
        "experiments": [
            {
                "name": experiment.name,
                "retriever_type": experiment.retriever_type,
                "top_k": experiment.top_k,
                "embedding_provider": experiment.embedding_provider,
                "embedding_model": experiment.embedding_model,
                "embedding_dimension": experiment.embedding_dimension,
                "chunk_size": experiment.chunk_size,
                "chunk_overlap": experiment.chunk_overlap,
                "reranker_enabled": experiment.reranker_enabled,
                "reranker_type": experiment.reranker_type,
                "reranker_model": experiment.reranker_model,
                "reranker_initial_top_k": experiment.reranker_initial_top_k,
                "reranker_final_top_k": experiment.top_k,
            }
            for experiment in sweep_config.experiments
        ],
        "query_rewriting_enabled": settings.enable_query_rewriting,
        "query_rewrite_model": settings.query_rewrite_model if settings.enable_query_rewriting else None,
        "query_rewrite_prompt_version": (
            settings.query_rewrite_prompt_version if settings.enable_query_rewriting else None
        ),
        "reranker_enabled": settings.enable_reranking,
        "reranker_type": settings.reranker_type,
        "reranker_model": settings.reranker_model if settings.enable_reranking else None,
        "reranker_initial_top_k": settings.reranker_initial_top_k if settings.enable_reranking else None,
        "reranker_final_top_k": settings.reranker_final_top_k if settings.enable_reranking else None,
        "python_command_used": str(sys.executable) + " " + " ".join(argv),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path


def write_retrieval_sweep_comparison_artifacts(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Path]:
    comparison_json_path = output_dir / "retrieval_sweep_summary.json"
    comparison_csv_path = output_dir / "retrieval_sweep_summary.csv"
    comparison_table_path = output_dir / "retrieval_sweep_ranking.md"

    payload = {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "sweep_manifest_path": str(manifest_path),
        "ranking": {
            "primary_metric": "recall_at_k",
            "tiebreak_metrics": ["mrr", "precision_at_k", "hit_at_k"],
        },
        "best_configuration": next((row for row in rows if row.get("is_best")), None),
        "runs": rows,
    }
    comparison_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "rank",
        "is_best",
        "run_name",
        "experiment_name",
        "chunk_size",
        "chunk_overlap",
        "retriever_type",
        "top_k",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "query_rewriting",
        "query_rewriting_enabled",
        "query_rewrite_model",
        "query_rewrite_prompt_version",
        "reranker",
        "reranker_enabled",
        "reranker_type",
        "reranker_model",
        "reranker_initial_top_k",
        "reranker_final_top_k",
        "dataset_path",
        "git_commit_sha",
        "mrr",
        "recall_at_k",
        "precision_at_k",
        "mean_precision_at_k",
        "hit_at_k",
        "context_relevance",
        "num_queries_total",
        "num_queries_evaluated",
        "num_queries_without_expected_source",
        "query_rewrite_avg_latency_ms",
        "query_rewrite_total_latency_ms",
        "query_rewrite_success_count",
        "query_rewrite_fallback_count",
        "query_rewrite_failure_count",
        "query_rewrite_total_tokens",
        "query_rewrite_estimated_total_cost",
        "mlflow_run_id",
        "output_dir",
        "results_json",
        "results_csv",
        "config_json",
    ]
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    comparison_table_path.write_text(format_retrieval_sweep_summary(rows), encoding="utf-8")

    return {
        "summary_json": comparison_json_path,
        "summary_csv": comparison_csv_path,
        "ranking_md": comparison_table_path,
        "manifest_json": manifest_path,
    }


def format_retrieval_sweep_summary(rows: list[dict[str, Any]]) -> str:
    headers = [
        "rank",
        "run_name",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "retriever_type",
        "top_k",
        "recall_at_k",
        "mrr",
        "precision_at_k",
        "query_rewriting",
        "reranker",
    ]
    rendered_rows = [
        [
            str(row.get("rank", "")),
            str(row.get("run_name", "")),
            str(row.get("embedding_model", "")),
            str(row.get("chunk_size", "")),
            str(row.get("chunk_overlap", "")),
            str(row.get("retriever_type", "")),
            str(row.get("top_k", "")),
            _format_metric(row.get("recall_at_k")),
            _format_metric(row.get("mrr")),
            _format_metric(row.get("precision_at_k")),
            str(row.get("query_rewriting", "")),
            str(row.get("reranker", "")),
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
    try:
        sweep_config = load_retrieval_sweep_config(args.config.resolve())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp_label = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_output_dir = args.output_dir.resolve() / f"{timestamp_label}_retrieval_sweep"
    settings = get_settings()

    try:
        rows, artifact_paths = run_retrieval_sweep(
            sweep_config=sweep_config,
            sweep_config_path=args.config.resolve(),
            dataset_path=args.dataset.resolve(),
            output_dir=experiment_output_dir,
            settings=settings,
            argv=sys.argv,
            min_expected_source_coverage=args.min_expected_source_coverage,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print()
    print("Retrieval sweep completed")
    print()
    if rows:
        best_row = rows[0]
        print(
            "Best configuration: "
            f"{best_row['experiment_name']} "
            f"(Recall@{best_row['top_k']}={_format_metric(best_row.get('recall_at_k'))}, "
            f"MRR={_format_metric(best_row.get('mrr'))}, "
            f"Precision@{best_row['top_k']}={_format_metric(best_row.get('precision_at_k'))})"
        )
        print()
    print(format_retrieval_sweep_summary(rows))
    print()
    print(f"Summary JSON written to: {artifact_paths['summary_json']}")
    print(f"Summary CSV written to: {artifact_paths['summary_csv']}")
    print(f"Ranking table written to: {artifact_paths['ranking_md']}")
    print(f"Manifest JSON written to: {artifact_paths['manifest_json']}")


def _build_sweep_run_timestamp_label(*, index: int, experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_name = _slugify_label(experiment_name)
    return f"{timestamp}_run_{index:02d}_{safe_name}"


def _optional_string(
    value: object,
    *,
    field_name: str,
    normalize_case: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    normalized = value.strip()
    return normalized.casefold() if normalize_case else normalized


def _optional_positive_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided.")
    return value


def _optional_non_negative_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer when provided.")
    return value


def _optional_bool(value: object, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean when provided.")
    return value


def _format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return ""


def _sort_metric(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def _slugify_label(value: str) -> str:
    normalized = SAFE_LABEL_PATTERN.sub("-", value.strip().casefold()).strip("-")
    return normalized or "run"


def _log_summary_to_tracker(
    *,
    tracker,
    run_name: str,
    rows: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
    settings: Settings,
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
                "workflow": "retrieval_sweep",
                "experiment_count": len(rows),
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
