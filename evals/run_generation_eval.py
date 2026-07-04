from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.llm import (
    JudgeClient,
    OpenRouterPricingLookupError,
    fetch_openrouter_model_pricing,
)
from app.infrastructure.prompts import PromptLoader
from app.infrastructure.tracking import create_experiment_tracker
from app.services.evals.generation_eval_service import GenerationEvalService
from app.services.llm import LLMConfigurationError, LLMServiceError
from app.services.llm import LLMService

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "generation_eval_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "results"
DEFAULT_PROMPTS_DIR = ROOT_DIR / "app" / "infrastructure" / "prompts" / "templates"


def _safe_file_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-context generation evaluation for the configured chat model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the fixed-context generation eval dataset JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model config ID. Defaults to the active runtime model from config.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model config ID used for judge scoring.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Optional prompt version. Defaults to DEFAULT_PROMPT_VERSION.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature used for answer generation.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the MLflow experiment name for this run.",
    )
    parser.add_argument(
        "--dataset-version",
        default="generation_eval_dataset_v1",
        help="Version label logged with the run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON and summary artifacts will be written.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    prompt_version = args.prompt_version or settings.default_prompt_version
    prompt_loader = PromptLoader(prompts_dir=DEFAULT_PROMPTS_DIR)
    llm_service = LLMService(settings=settings)
    model_config = llm_service.get_model_config(args.model)
    pricing_lookup_note: str | None = None
    settings, llm_service, pricing_lookup_note = await _apply_openrouter_pricing_if_missing(
        settings=settings,
        llm_service=llm_service,
        model_config_id=model_config.config_id,
    )
    model_config = llm_service.get_model_config(args.model)
    judge_client = JudgeClient(settings=settings) if args.judge_model else None
    eval_service = GenerationEvalService(
        prompt_loader=prompt_loader,
        llm_service=llm_service,
        judge_client=judge_client,
    )
    examples = eval_service.load_dataset(args.dataset)
    model_base_url = llm_service.get_model_base_url(model_config.config_id)
    experiment_name = args.experiment_name or settings.mlflow_experiment_name
    tracker = create_experiment_tracker(settings, experiment_name)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_model_config_id = _safe_file_stem(model_config.config_id)
    run_name = f"generation-eval-{safe_model_config_id}-{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run = await eval_service.evaluate_dataset(
            examples=examples,
            prompt_version=prompt_version,
            model_config_id=model_config.config_id,
            judge_model_config_id=args.judge_model,
            temperature=args.temperature,
        )
    except (LLMConfigurationError, LLMServiceError) as exc:
        raise SystemExit(
            "\n".join(
                [
                    "Generation evaluation failed before results could be written.",
                    f"Model config: {model_config.config_id}",
                    f"Provider: {model_config.provider}",
                    f"Model: {model_config.model}",
                    f"Base URL: {model_base_url}",
                    f"Reason: {exc}",
                ]
            )
        ) from exc
    aggregate = run.aggregate

    result_payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "dataset_path": str(args.dataset),
        "dataset_version": args.dataset_version,
        "prompt_version": prompt_version,
        "aggregate": asdict(aggregate),
        "records": eval_service.records_as_json(run.records),
    }
    result_path = args.output_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    summary_text = eval_service.render_summary(aggregate)
    summary_path = args.output_dir / f"{run_name}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    if tracker.enabled:
        with tracker.run(run_name):
            tracker.log_params(
                {
                    "dataset_path": str(args.dataset),
                    "dataset_version": args.dataset_version,
                    "prompt_version": prompt_version,
                    "llm_provider": aggregate.model_provider,
                    "llm_model": aggregate.model_name,
                    "llm_base_url": aggregate.model_base_url,
                    "model_config_id": aggregate.model_config_id,
                    "judge_model_config_id": args.judge_model,
                    "temperature": args.temperature,
                    "retrieval_mode": "fixed_context",
                    "retriever_type": "fixed_context",
                }
            )
            metrics: dict[str, float | int] = {
                "total_examples": aggregate.total_examples,
                "passed_examples": aggregate.passed_examples,
                "failed_examples": aggregate.failed_examples,
                "pass_rate": aggregate.pass_rate,
                "average_quality_score": aggregate.average_quality_score,
                "average_groundedness_score": aggregate.average_groundedness_score,
                "latency_ms_avg": aggregate.latency_ms_avg,
                "latency_ms_p50": aggregate.latency_ms_p50,
                "latency_ms_p95": aggregate.latency_ms_p95,
                "total_prompt_tokens": aggregate.total_prompt_tokens,
                "total_completion_tokens": aggregate.total_completion_tokens,
                "total_tokens": aggregate.total_tokens,
                "avg_tokens_per_response": aggregate.avg_tokens_per_response,
                "responses_with_usage": aggregate.responses_with_usage,
                "responses_with_cost_estimate": aggregate.responses_with_cost_estimate,
            }
            if aggregate.average_context_relevance is not None:
                metrics["average_context_relevance"] = aggregate.average_context_relevance
                metrics["average_faithfulness"] = aggregate.average_faithfulness or 0.0
                metrics["average_answer_relevance"] = aggregate.average_answer_relevance or 0.0
            if aggregate.estimated_prompt_cost_usd is not None:
                metrics["estimated_prompt_cost_usd"] = aggregate.estimated_prompt_cost_usd
                metrics["estimated_completion_cost_usd"] = (
                    aggregate.estimated_completion_cost_usd or 0.0
                )
                metrics["estimated_total_cost_usd"] = aggregate.estimated_total_cost_usd or 0.0
                metrics["average_cost_per_response_usd"] = (
                    aggregate.average_cost_per_response_usd or 0.0
                )
            tracker.log_metrics(metrics)
            tracker.log_artifact(result_path)
            tracker.log_artifact(summary_path)

    print(summary_text, end="")
    print(f"Detailed results written to: {result_path}")
    print(f"Active base URL: {model_base_url}")
    if pricing_lookup_note:
        print(pricing_lookup_note)


async def _apply_openrouter_pricing_if_missing(
    *,
    settings,
    llm_service: LLMService,
    model_config_id: str,
) -> tuple[object, LLMService, str | None]:
    model_config = llm_service.get_model_config(model_config_id)
    if model_config.provider != "openrouter":
        return settings, llm_service, None
    if (
        model_config.input_cost_per_1m_tokens is not None
        and model_config.output_cost_per_1m_tokens is not None
    ):
        return settings, llm_service, None

    try:
        pricing = await fetch_openrouter_model_pricing(
            api_key=settings.openrouter_api_key or settings.llm_api_key,
            base_url=llm_service.get_model_base_url(model_config.config_id),
            model=model_config.model,
        )
    except OpenRouterPricingLookupError as exc:
        return (
            settings,
            llm_service,
            f"OpenRouter pricing lookup unavailable; continuing without cost estimates. Reason: {exc}",
        )

    if (
        pricing.prompt_cost_per_1m_tokens is None
        or pricing.completion_cost_per_1m_tokens is None
    ):
        return (
            settings,
            llm_service,
            "OpenRouter pricing lookup returned incomplete token pricing; continuing without cost estimates.",
        )

    updated_settings = replace(
        settings,
        model_configs_json=_merge_model_cost_override(
            settings.model_configs_json,
            config_id=model_config.config_id,
            provider=model_config.provider,
            model=model_config.model,
            prompt_cost_per_1m_tokens=pricing.prompt_cost_per_1m_tokens,
            completion_cost_per_1m_tokens=pricing.completion_cost_per_1m_tokens,
        ),
    )
    return (
        updated_settings,
        LLMService(settings=updated_settings),
        (
            "Loaded OpenRouter pricing automatically for "
            f"{model_config.model}: prompt=${pricing.prompt_cost_per_1m_tokens:.6f}/1M, "
            f"completion=${pricing.completion_cost_per_1m_tokens:.6f}/1M"
        ),
    )


def _merge_model_cost_override(
    model_configs_json: str | None,
    *,
    config_id: str,
    provider: str,
    model: str,
    prompt_cost_per_1m_tokens: float,
    completion_cost_per_1m_tokens: float,
) -> str:
    if model_configs_json is None:
        payload: list[dict[str, object]] = []
    else:
        parsed = json.loads(model_configs_json)
        if not isinstance(parsed, list):
            raise ValueError("MODEL_CONFIGS_JSON must be a JSON array when updating model costs.")
        payload = [dict(item) for item in parsed if isinstance(item, dict)]

    override_entry = {
        "config_id": config_id,
        "provider": provider,
        "model": model,
        "input_cost_per_1m_tokens": prompt_cost_per_1m_tokens,
        "output_cost_per_1m_tokens": completion_cost_per_1m_tokens,
    }
    for entry in payload:
        if entry.get("config_id") == config_id:
            entry.update(override_entry)
            break
    else:
        payload.append(override_entry)

    return json.dumps(payload)


if __name__ == "__main__":
    asyncio.run(main())
