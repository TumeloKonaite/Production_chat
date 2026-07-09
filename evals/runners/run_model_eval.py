from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.prompts import PromptLoader
from app.infrastructure.tracking.conventions import (
    build_common_tracking_params,
    build_generation_tracking_metrics,
    build_generation_tracking_params,
    get_git_sha,
    resolve_prompt_template_path,
)
from app.infrastructure.tracking import create_experiment_tracker
from app.services.evals.eval_service import (
    load_eval_dataset,
    records_as_json,
    render_comparison_summary,
    write_json,
)
from app.services.evals.model_experiment_service import ModelExperimentService
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "model_eval_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results"
DEFAULT_PROMPTS_DIR = ROOT_DIR / "app" / "infrastructure" / "prompts" / "templates"
DEFAULT_TEMPERATURE = 0.2


def _safe_file_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared chatbot eval dataset across multiple configured models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model config IDs to evaluate, for example openai:gpt-4.1-mini openai:gpt-4.1",
    )
    parser.add_argument(
        "--prompt-version",
        required=True,
        help="Prompt version to use for every evaluated model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the eval dataset JSONL file.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the MLflow experiment name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for every model run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override passed to every evaluated model.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Override the retrieval top-k used during evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON results and summary artifacts will be written.",
    )
    parser.add_argument(
        "--dataset-version",
        default="model_eval_dataset_v1",
        help="Version label logged with each experiment run.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    prompt_loader = PromptLoader(prompts_dir=DEFAULT_PROMPTS_DIR)
    retrieval_service = RetrievalService(settings=settings)
    llm_service = LLMService(settings=settings)
    experiment_service = ModelExperimentService(
        prompt_loader=prompt_loader,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )

    examples = load_eval_dataset(args.dataset)
    retrieval_top_k = args.retrieval_top_k or settings.retrieval_top_k
    retrieval_config = settings.default_retrieval_config
    experiment_name = args.experiment_name or settings.mlflow_experiment_name
    tracker = create_experiment_tracker(settings, experiment_name)
    prompt_template_path = resolve_prompt_template_path(
        prompts_dir=DEFAULT_PROMPTS_DIR,
        prompt_version=args.prompt_version,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = datetime.now(UTC).replace(microsecond=0).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_artifact_path: Path | None = None
    if prompt_template_path is not None:
        prompt_artifact_path = args.output_dir / f"model_comparison_prompt_{timestamp}.md"
        prompt_artifact_path.write_text(
            prompt_template_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    config_path = args.output_dir / f"model_comparison_config_{timestamp}.json"
    write_json(
        config_path,
        {
            "run_id": run_id,
            "dataset_path": str(args.dataset),
            "dataset_version": args.dataset_version,
            "prompt_version": args.prompt_version,
            "prompt_template_path": (
                str(prompt_template_path) if prompt_template_path is not None else None
            ),
            "models": args.models,
            "retrieval_config": retrieval_config,
            "retrieval_top_k": retrieval_top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "git_sha": get_git_sha(),
        },
    )

    aggregates = []
    result_payloads: list[dict[str, object]] = []

    for model_config_id in args.models:
        run = await experiment_service.evaluate_model(
            model_config_id=model_config_id,
            examples=examples,
            prompt_version=args.prompt_version,
            retrieval_config=retrieval_config,
            retrieval_top_k=retrieval_top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        aggregate = run.aggregate
        aggregates.append(aggregate)

        result_payload = {
            "run_id": run_id,
            "mlflow_experiment_name": experiment_name,
            "prompt_version": args.prompt_version,
            "models": args.models,
            "results": records_as_json(run.records),
        }
        result_payloads.append(result_payload)

        result_path = args.output_dir / f"{_safe_file_stem(model_config_id)}_model_eval_{timestamp}.json"
        write_json(result_path, result_payload)

        with tracker.run(model_config_id):
            tracker.log_params(
                build_generation_tracking_params(
                    workflow="model_eval",
                    experiment_family="model_eval",
                    run_name=model_config_id,
                    dataset_path=args.dataset,
                    dataset_version=args.dataset_version,
                    prompt_version=args.prompt_version,
                    prompt_template_path=prompt_template_path,
                    model_config_id=aggregate.model_config_id,
                    llm_provider=aggregate.model_provider,
                    llm_model=aggregate.model_name,
                    llm_base_url=llm_service.get_model_base_url(aggregate.model_config_id),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    git_sha=get_git_sha(),
                    retrieval_config=retrieval_config,
                    context_top_k=retrieval_top_k,
                    extra={
                        "retriever_type": settings.retriever_type,
                        "embedding_provider": settings.embedding_provider,
                        "embedding_model": settings.knowledge_embedding_model,
                        "embedding_dimension": settings.embedding_dimension,
                    },
                )
            )
            tracker.log_metrics(
                build_generation_tracking_metrics(
                    quality_score=aggregate.average_quality_score,
                    groundedness_score=aggregate.average_groundedness_score,
                    faithfulness_score=None,
                    relevance_score=None,
                    avg_latency_ms=aggregate.average_latency_ms,
                    p95_latency_ms=aggregate.p95_latency_ms,
                    prompt_tokens=aggregate.total_input_tokens,
                    completion_tokens=aggregate.total_output_tokens,
                    total_tokens=aggregate.total_tokens,
                    estimated_cost_usd=aggregate.total_cost_usd,
                    extra={
                        "generation.total_examples": aggregate.total_examples,
                        "generation.passed_examples": aggregate.passed_examples,
                        "generation.failed_examples": aggregate.failed_examples,
                        "generation.pass_rate": aggregate.pass_rate,
                        "generation.average_cost_per_response_usd": (
                            aggregate.average_cost_per_response_usd
                        ),
                    },
                )
            )
            tracker.log_artifact(result_path)
            if prompt_artifact_path is not None:
                tracker.log_artifact(prompt_artifact_path)

    summary_text = render_comparison_summary(aggregates)
    summary_path = args.output_dir / f"model_comparison_summary_{timestamp}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    aggregate_path = args.output_dir / f"model_comparison_results_{timestamp}.json"
    write_json(
        aggregate_path,
        {
            "run_id": run_id,
            "mlflow_experiment_name": experiment_name,
            "prompt_version": args.prompt_version,
            "models": args.models,
            "aggregates": [asdict(aggregate) for aggregate in aggregates],
            "results": result_payloads,
        },
    )

    if tracker.enabled:
        best_aggregate = max(
            aggregates,
            key=lambda aggregate: (
                aggregate.pass_rate,
                aggregate.average_quality_score,
                aggregate.average_groundedness_score,
            ),
        )
        with tracker.run(f"{args.prompt_version}-summary"):
            tracker.log_params(
                build_common_tracking_params(
                    workflow="model_eval_summary",
                    experiment_family="model_eval",
                    run_name=f"{args.prompt_version}-summary",
                    dataset_path=args.dataset,
                    dataset_version=args.dataset_version,
                    git_sha=get_git_sha(),
                    prompt_version=args.prompt_version,
                    prompt_template_path=prompt_template_path,
                    extra={
                        "models": ",".join(args.models),
                        "model_count": len(args.models),
                        "retrieval_config": retrieval_config,
                        "retriever_type": settings.retriever_type,
                        "retrieval_top_k": retrieval_top_k,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                    },
                )
            )
            tracker.log_metrics(
                build_generation_tracking_metrics(
                    quality_score=best_aggregate.average_quality_score,
                    groundedness_score=best_aggregate.average_groundedness_score,
                    faithfulness_score=None,
                    relevance_score=None,
                    avg_latency_ms=best_aggregate.average_latency_ms,
                    p95_latency_ms=best_aggregate.p95_latency_ms,
                    prompt_tokens=best_aggregate.total_input_tokens,
                    completion_tokens=best_aggregate.total_output_tokens,
                    total_tokens=best_aggregate.total_tokens,
                    estimated_cost_usd=best_aggregate.total_cost_usd,
                    extra={
                        "generation.pass_rate": best_aggregate.pass_rate,
                        "generation.passed_examples": best_aggregate.passed_examples,
                        "generation.failed_examples": best_aggregate.failed_examples,
                        "experiment.model_count": len(args.models),
                    },
                )
            )
            tracker.log_artifact(summary_path)
            tracker.log_artifact(aggregate_path)
            tracker.log_artifact(config_path)
            if prompt_artifact_path is not None:
                tracker.log_artifact(prompt_artifact_path)

    print(summary_text, end="")


if __name__ == "__main__":
    asyncio.run(main())
