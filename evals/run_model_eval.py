from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.prompts import PromptLoader
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

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = datetime.now(UTC).replace(microsecond=0).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

        result_path = args.output_dir / f"{model_config_id.replace(':', '_')}_model_eval_{timestamp}.json"
        write_json(result_path, result_payload)

        with tracker.run(model_config_id):
            tracker.log_params(
                {
                    "model_config_id": aggregate.model_config_id,
                    "model_provider": aggregate.model_provider,
                    "model_name": aggregate.model_name,
                    "prompt_version": args.prompt_version,
                    "retrieval_config": retrieval_config,
                    "retriever_type": settings.retriever_type,
                    "eval_dataset_version": args.dataset_version,
                    "temperature": args.temperature,
                    "top_k": retrieval_top_k,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.knowledge_embedding_model,
                    "embedding_dimension": settings.embedding_dimension,
                    "judge_model": None,
                }
            )
            tracker.log_metrics(
                {
                    "total_examples": aggregate.total_examples,
                    "passed_examples": aggregate.passed_examples,
                    "failed_examples": aggregate.failed_examples,
                    "pass_rate": aggregate.pass_rate,
                    "average_quality_score": aggregate.average_quality_score,
                    "average_groundedness_score": aggregate.average_groundedness_score,
                    "average_latency_ms": aggregate.average_latency_ms,
                    "p95_latency_ms": aggregate.p95_latency_ms,
                    "total_input_tokens": aggregate.total_input_tokens,
                    "total_output_tokens": aggregate.total_output_tokens,
                    "total_tokens": aggregate.total_tokens,
                    "total_cost_usd": aggregate.total_cost_usd,
                    "average_cost_per_response_usd": aggregate.average_cost_per_response_usd,
                }
            )
            tracker.log_artifact(result_path)

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
        with tracker.run(f"{args.prompt_version}-summary"):
            tracker.log_params(
                {
                    "prompt_version": args.prompt_version,
                    "retrieval_config": retrieval_config,
                    "retriever_type": settings.retriever_type,
                    "eval_dataset_version": args.dataset_version,
                    "models": ",".join(args.models),
                }
            )
            tracker.log_artifact(summary_path)
            tracker.log_artifact(aggregate_path)

    print(summary_text, end="")


if __name__ == "__main__":
    asyncio.run(main())
