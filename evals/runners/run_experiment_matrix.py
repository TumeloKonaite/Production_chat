from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from evals.matrix import (
    DEFAULT_EXPERIMENT_OUTPUT_DIR,
    expand_suite_plan,
    format_suite_plan,
    load_experiment_matrix_config,
    run_experiment_matrix,
)
from evals.runners.retrieval_eval_runner import DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE
from evals.runners.run_generation_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_GENERATION_DATASET_PATH,
)
from evals.runners.run_rag_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RAG_DATASET_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
)
from evals.runners.run_retrieval_eval import (
    DEFAULT_DATASET_PATH as DEFAULT_RETRIEVAL_DATASET_PATH,
)

DEFAULT_CONFIG_PATH = ROOT_DIR / "evals" / "configs" / "experiment_matrix.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled retrieval, generation, or RAG experiment suites.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment matrix YAML or JSON config file.",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Named suite to run from the experiment matrix config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT_DIR,
        help="Directory where experiment matrix outputs will be written.",
    )
    parser.add_argument(
        "--retrieval-dataset",
        type=Path,
        default=DEFAULT_RETRIEVAL_DATASET_PATH,
        help="Dataset path for retrieval suites.",
    )
    parser.add_argument(
        "--generation-dataset",
        type=Path,
        default=DEFAULT_GENERATION_DATASET_PATH,
        help="Dataset path for generation suites.",
    )
    parser.add_argument(
        "--rag-dataset",
        type=Path,
        default=DEFAULT_RAG_DATASET_PATH,
        help="Dataset path for RAG suites.",
    )
    parser.add_argument(
        "--rag-judge-prompt",
        type=Path,
        default=DEFAULT_JUDGE_PROMPT_PATH,
        help="Judge prompt template used for RAG suites.",
    )
    parser.add_argument(
        "--generation-judge-model",
        default=None,
        help="Optional judge model config ID for generation suites.",
    )
    parser.add_argument(
        "--rag-judge-model",
        default=None,
        help="Optional judge model config ID for RAG suites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the resolved suite plan without executing any evals.",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required for full suites or suite names containing 'full'.",
    )
    parser.add_argument(
        "--min-expected-source-coverage",
        type=float,
        default=DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
        help="Minimum retrieval dataset coverage required before retrieval evals run.",
    )
    parser.add_argument(
        "--persist-rag-results",
        action="store_true",
        help="Persist RAG eval results to the database during matrix runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        matrix_config = load_experiment_matrix_config(args.config.resolve())
        suite = matrix_config.suites.get(args.suite)
        if suite is None:
            available = ", ".join(sorted(matrix_config.suites))
            raise ValueError(f"Unknown suite: {args.suite}. Available suites: {available}")
        plan = expand_suite_plan(suite)
        print(format_suite_plan(plan))
        print()
        if args.dry_run:
            print("No evals were executed because --dry-run was provided.")
            return
        result = run_experiment_matrix(
            matrix_config=matrix_config,
            suite_name=args.suite,
            settings=get_settings(),
            argv=sys.argv,
            output_dir=args.output_dir.resolve(),
            retrieval_dataset_path=args.retrieval_dataset.resolve(),
            generation_dataset_path=args.generation_dataset.resolve(),
            rag_dataset_path=args.rag_dataset.resolve(),
            rag_judge_prompt_path=args.rag_judge_prompt.resolve(),
            generation_judge_model_config_id=args.generation_judge_model,
            rag_judge_model_config_id=args.rag_judge_model,
            dry_run=args.dry_run,
            confirm_full_run=args.confirm_full_run,
            min_expected_source_coverage=args.min_expected_source_coverage,
            persist_rag_results=args.persist_rag_results,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Matrix run ID: {result.matrix_run_id}")
    print(f"Status: {result.status}")
    print(f"Summary JSON written to: {result.summary_paths['summary_json']}")
    print(f"Summary CSV written to: {result.summary_paths['summary_csv']}")
    print(f"Failures JSON written to: {result.failures_path}")
    print(f"Manifest JSON written to: {result.manifest_path}")


if __name__ == "__main__":
    main()
