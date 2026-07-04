from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import (
    SUPPORTED_RERANKER_TYPES,
    SUPPORTED_RETRIEVER_TYPES,
    get_settings,
)
from app.infrastructure.tracking import create_experiment_tracker
from evals.retrieval_eval_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE,
    DEFAULT_OUTPUT_DIR,
    RetrievalEvalDatasetValidationError,
    RetrievalEvalExample,
    RetrievalEvalRunResult,
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
    "load_eval_config",
    "log_run_to_tracker",
    "parse_args",
    "run_retrieval_eval",
    "validate_dataset_examples",
    "write_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation experiments with optional reranking.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file for a retrieval evaluation run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Final top-k used during evaluation. Overrides config when provided.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the retrieval evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where retrieval evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--min-expected-source-coverage",
        type=float,
        default=None,
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
        help="Disable eval-only query rewriting even if enabled in config or environment.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Enable reranking during retrieval evaluation.",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranking even if enabled in config or environment.",
    )
    parser.add_argument(
        "--reranker-type",
        choices=sorted(SUPPORTED_RERANKER_TYPES),
        default=None,
        help="Override the reranker type used for this run.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Override the reranker model config ID used for this run.",
    )
    parser.add_argument(
        "--reranker-initial-top-k",
        type=int,
        default=None,
        help="Number of candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit tracking run name.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes to attach to the run config and tracker params.",
    )
    return parser.parse_args()


def load_eval_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Retrieval eval config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Retrieval eval config must be valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Retrieval eval config must be a JSON object.")
    return payload


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve() if args.config is not None else None

    try:
        config = load_eval_config(config_path) if config_path is not None else {}
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    settings = get_settings()
    config_base_dir = config_path.parent if config_path is not None else ROOT_DIR

    if args.enable_query_rewriting and args.disable_query_rewriting:
        raise SystemExit(
            "Choose either --enable-query-rewriting or --disable-query-rewriting, not both."
        )
    if args.enable_reranker and args.disable_reranker:
        raise SystemExit("Choose either --enable-reranker or --disable-reranker, not both.")

    settings = _apply_retriever_config(settings, config)
    settings = _apply_query_rewrite_config(settings, args=args, config=config)

    top_k = _resolve_top_k(args=args, config=config, settings=settings)
    settings = _apply_reranker_config(settings, args=args, config=config, top_k=top_k)

    experiment_name = (
        _get_optional_string(config, "experiment_name", "mlflow_experiment_name")
        or settings.mlflow_experiment_name
    )
    tracker = create_experiment_tracker(settings, experiment_name)

    dataset_path = _resolve_path_option(
        cli_value=args.dataset,
        config_value=_get_optional_string(config, "dataset", "dataset_path"),
        config_base_dir=config_base_dir,
        default=DEFAULT_DATASET_PATH,
    )
    output_dir = _resolve_path_option(
        cli_value=args.output_dir,
        config_value=_get_optional_string(config, "output_dir"),
        config_base_dir=config_base_dir,
        default=DEFAULT_OUTPUT_DIR,
    )
    min_expected_source_coverage = (
        args.min_expected_source_coverage
        if args.min_expected_source_coverage is not None
        else _get_optional_float(config, "min_expected_source_coverage")
    )
    if min_expected_source_coverage is None:
        min_expected_source_coverage = DEFAULT_MIN_EXPECTED_SOURCE_COVERAGE

    run_name = args.run_name or _get_optional_string(config, "run_name")
    notes = args.notes or _get_optional_string(config, "notes")

    try:
        result = run_retrieval_eval(
            settings=settings,
            dataset_path=dataset_path.resolve(),
            output_root=output_dir.resolve(),
            top_k=top_k,
            tracker=tracker,
            argv=sys.argv,
            min_expected_source_coverage=min_expected_source_coverage,
            run_name=run_name,
            notes=notes,
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


def _apply_retriever_config(settings, config: dict[str, Any]):
    config_retriever_type = _get_optional_string(config, "retriever_type")
    if config_retriever_type is None:
        return settings

    normalized_retriever_type = config_retriever_type.casefold()
    if normalized_retriever_type not in SUPPORTED_RETRIEVER_TYPES:
        supported_values = ", ".join(sorted(SUPPORTED_RETRIEVER_TYPES))
        raise SystemExit(f"retriever_type must be one of: {supported_values}.")
    return replace(settings, retriever_type=normalized_retriever_type)


def _apply_query_rewrite_config(settings, *, args: argparse.Namespace, config: dict[str, Any]):
    enabled = settings.enable_query_rewriting
    config_enabled = _get_optional_bool(
        config,
        "query_rewriting_enabled",
        "enable_query_rewriting",
    )
    if config_enabled is not None:
        enabled = config_enabled
    if args.enable_query_rewriting:
        enabled = True
    elif args.disable_query_rewriting:
        enabled = False
    return replace(settings, enable_query_rewriting=enabled)


def _apply_reranker_config(
    settings,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    top_k: int,
):
    enabled = settings.enable_reranking
    config_enabled = _get_optional_bool(config, "reranker_enabled", "enable_reranking")
    if config_enabled is not None:
        enabled = config_enabled
    if args.enable_reranker:
        enabled = True
    elif args.disable_reranker:
        enabled = False

    reranker_type = (
        args.reranker_type
        or _get_optional_string(config, "reranker_type")
        or settings.reranker_type
    )
    if reranker_type not in SUPPORTED_RERANKER_TYPES:
        supported_values = ", ".join(sorted(SUPPORTED_RERANKER_TYPES))
        raise SystemExit(f"reranker_type must be one of: {supported_values}.")
    reranker_model = (
        args.reranker_model
        or _get_optional_string(config, "reranker_model")
        or settings.reranker_model
    )
    reranker_initial_top_k = (
        args.reranker_initial_top_k
        or _get_optional_int(config, "reranker_initial_top_k")
        or _get_optional_int(config, "retriever_top_k")
        or settings.reranker_initial_top_k
    )
    return replace(
        settings,
        enable_reranking=enabled,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
        reranker_initial_top_k=reranker_initial_top_k,
        reranker_final_top_k=top_k,
    )


def _resolve_top_k(*, args: argparse.Namespace, config: dict[str, Any], settings) -> int:
    if args.k is not None:
        return args.k

    for key in ("final_top_k", "top_k", "k", "retriever_top_k"):
        value = _get_optional_int(config, key)
        if value is not None:
            return value

    return settings.retrieval_top_k


def _resolve_path_option(
    *,
    cli_value: Path | None,
    config_value: str | None,
    config_base_dir: Path,
    default: Path,
) -> Path:
    if cli_value is not None:
        return cli_value
    if config_value is None:
        return default

    candidate = Path(config_value)
    if candidate.is_absolute():
        return candidate
    return config_base_dir / candidate


def _get_optional_string(config: dict[str, Any], *keys: str) -> str | None:
    value = _get_config_value(config, *keys)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{keys[0]} must be a non-empty string when provided.")
    return value.strip()


def _get_optional_bool(config: dict[str, Any], *keys: str) -> bool | None:
    value = _get_config_value(config, *keys)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise SystemExit(f"{keys[0]} must be a boolean when provided.")
    return value


def _get_optional_int(config: dict[str, Any], *keys: str) -> int | None:
    value = _get_config_value(config, *keys)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{keys[0]} must be an integer when provided.")
    return value


def _get_optional_float(config: dict[str, Any], *keys: str) -> float | None:
    value = _get_config_value(config, *keys)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(f"{keys[0]} must be a number when provided.")
    return float(value)


def _get_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


if __name__ == "__main__":
    main()
