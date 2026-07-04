from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.tracking import create_experiment_tracker
from evals.retrieval_eval_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    DEFAULT_OUTPUT_DIR,
    RetrievalEvalExample,
    RetrievalEvalRunResult,
    RetrievalEvalDatasetValidationError,
    RetrievalDatasetValidationSummary,
    build_run_config,
    build_tracking_run_name,
    create_output_directory,
    evaluate_examples,
    evaluate_examples_for_k_values,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    log_run_to_tracker,
    run_retrieval_eval,
    validate_dataset_examples,
    write_artifacts,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE",
    "DEFAULT_OUTPUT_DIR",
    "RetrievalDatasetValidationSummary",
    "RetrievalEvalDatasetValidationError",
    "RetrievalEvalExample",
    "RetrievalEvalRunResult",
    "build_run_config",
    "build_tracking_run_name",
    "create_output_directory",
    "evaluate_examples",
    "evaluate_examples_for_k_values",
    "format_dataset_validation_summary",
    "load_and_validate_dataset",
    "log_run_to_tracker",
    "parse_args",
    "run_retrieval_eval",
    "validate_dataset_examples",
    "write_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only baseline evaluation for the current vector retriever.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Retrieval top-k used during evaluation. Defaults to RETRIEVAL_TOP_K.",
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
    parser.add_argument(
        "--min-expected-source-coverage",
        type=float,
        default=DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
        help=(
            "Minimum fraction of dataset rows that must include expected_source_documents "
            "before evaluation runs."
        ),
    )
    parser.add_argument(
        "--enable-query-rewriting",
        action="store_true",
        help="Enable eval-only query rewriting before retrieval.",
    )
    parser.add_argument(
        "--disable-query-rewriting",
        action="store_true",
        help="Disable eval-only query rewriting even if enabled in the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.enable_query_rewriting and args.disable_query_rewriting:
        raise SystemExit(
            "Choose either --enable-query-rewriting or --disable-query-rewriting, not both."
        )
    if args.enable_query_rewriting:
        settings = replace(settings, enable_query_rewriting=True)
    elif args.disable_query_rewriting:
        settings = replace(settings, enable_query_rewriting=False)
    top_k = args.k if args.k is not None else settings.retrieval_top_k
    tracker = create_experiment_tracker(settings, settings.mlflow_experiment_name)

    try:
        result = run_retrieval_eval(
            settings=settings,
            dataset_path=args.dataset.resolve(),
            output_root=args.output_dir.resolve(),
            top_k=top_k,
            tracker=tracker,
            argv=sys.argv,
            min_expected_source_coverage=args.min_expected_source_coverage,
        )
    except (RetrievalEvalDatasetValidationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(format_dataset_validation_summary(result.validation_summary))
    print(json.dumps(result.summary, indent=2, ensure_ascii=True))
    print(f"Results JSON written to: {result.artifact_paths['results_json']}")
    print(f"Results CSV written to: {result.artifact_paths['results_csv']}")
    print(f"Config JSON written to: {result.artifact_paths['config_json']}")
    prompt_artifact = result.artifact_paths.get("query_rewrite_prompt_txt")
    if prompt_artifact is not None:
        print(f"Query rewrite prompt written to: {prompt_artifact}")


if __name__ == "__main__":
    main()
