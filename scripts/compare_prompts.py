from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import mean
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.infrastructure.prompts import PromptLoader
from app.infrastructure.tracking import create_experiment_tracker
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    should_use_direct_fallback,
)
from app.services.llm import LLMChatMessage, LLMService
from app.services.retrieval import RetrievedChunk, RetrievalService

DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "prompt_eval_questions.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "evals" / "prompt_eval_results"
DEFAULT_PROMPTS_DIR = ROOT_DIR / "app" / "infrastructure" / "prompts" / "templates"
DEFAULT_TEMPERATURE = 0.2
REFUSAL_MARKERS = (
    "do not have enough",
    "don't have enough",
    "do not know",
    "don't know",
    "not enough information",
    "not have that information",
    "cannot confirm",
    "can't confirm",
    "not supported by the context",
)
STOP_WORDS = {
    "about",
    "after",
    "also",
    "been",
    "from",
    "have",
    "into",
    "that",
    "their",
    "them",
    "they",
    "this",
    "what",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    category: str
    expected_behavior: str


@dataclass(frozen=True, slots=True)
class EvalResult:
    question_id: str
    question: str
    category: str
    expected_behavior: str
    prompt_version: str
    response: str
    model: str
    used_fallback: bool
    latency_ms: int
    retrieved_sources: list[str]
    groundedness_score: float
    tone_score: float
    refusal_score: float
    unsupported_claim_score: float
    answer_quality_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt variants against a shared evaluation set and log results to MLflow.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated evaluation artifacts.",
    )
    parser.add_argument(
        "--prompt-version",
        action="append",
        dest="prompt_versions",
        help="Prompt version to evaluate. Repeat to evaluate multiple versions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature to send to the model during evaluation.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Override the retrieval top-k used during evaluation.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the MLflow experiment name used for prompt comparisons.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[EvalQuestion]:
    rows: list[EvalQuestion] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(
            EvalQuestion(
                id=str(payload["id"]),
                question=str(payload["question"]),
                category=str(payload["category"]),
                expected_behavior=str(payload["expected_behavior"]),
            )
        )
    return rows


def clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def is_refusal(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def extract_keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.casefold())
        if len(token) >= 4 and token not in STOP_WORDS
    }


def score_groundedness(response: str, retrieved_chunks: list[RetrievedChunk]) -> float:
    if not retrieved_chunks:
        return 1.0 if is_refusal(response) else 0.2

    if is_refusal(response):
        return 0.35

    context_terms = extract_keywords(" ".join(chunk.content for chunk in retrieved_chunks))
    response_terms = extract_keywords(response)
    overlap_count = len(context_terms & response_terms)
    return clamp(0.35 + min(overlap_count, 5) * 0.13)


def score_tone(response: str, prompt_version: str) -> float:
    words = re.findall(r"[A-Za-z']+", response)
    word_count = len(words)
    contractions = len(re.findall(r"\b\w+'\w+\b", response))
    normalized = response.casefold()

    if prompt_version == "v2_warm_conversational":
        warm_markers = sum(
            1
            for marker in ("you", "your", "help", "practical", "visitors", "can")
            if marker in normalized
        )
        return clamp(0.35 + min(warm_markers, 3) * 0.15 + (0.2 if contractions else 0.0) + (0.1 if word_count >= 18 else 0.0))

    professional_markers = sum(
        1
        for marker in ("experience", "projects", "engineering", "systems", "production")
        if marker in normalized
    )
    return clamp(
        0.4
        + min(professional_markers, 3) * 0.15
        + (0.15 if contractions == 0 else 0.0)
        + (0.1 if "!" not in response else 0.0)
    )


def score_refusal(response: str, expected_behavior: str) -> float:
    refusal = is_refusal(response)
    if expected_behavior == "refuse_or_clarify":
        return 1.0 if refusal else 0.0
    return 1.0 if not refusal else 0.4


def score_unsupported_claim(response: str, category: str) -> float:
    if category != "unsupported_claim":
        return 1.0
    return 1.0 if is_refusal(response) else 0.0


def score_answer_quality(response: str, expected_behavior: str) -> float:
    refusal = is_refusal(response)
    word_count = len(re.findall(r"[A-Za-z']+", response))

    if refusal:
        return 0.9 if expected_behavior == "refuse_or_clarify" else 0.25
    if word_count >= 30:
        return 1.0
    if word_count >= 18:
        return 0.85
    if word_count >= 10:
        return 0.65
    return 0.4


async def evaluate_prompt_version(
    *,
    prompt_version: str,
    questions: list[EvalQuestion],
    prompt_loader: PromptLoader,
    llm_service: LLMService,
    retrieval_service: RetrievalService,
    retrieval_top_k: int,
    temperature: float,
) -> list[EvalResult]:
    prompt_text = prompt_loader.load(prompt_version)
    results: list[EvalResult] = []

    for question in questions:
        retrieved_chunks = retrieval_service.retrieve(question.question, top_k=retrieval_top_k)
        used_fallback = should_use_direct_fallback(question.question, retrieved_chunks)

        if used_fallback:
            response_text = build_direct_fallback_text(question.question)
            model = llm_service.model
            latency_ms = 0
        else:
            system_prompt = build_chat_system_prompt(
                base_prompt=prompt_text,
                message=question.question,
                retrieved_chunks=retrieved_chunks,
            )
            response = await llm_service.generate_response(
                [LLMChatMessage(role="user", content=question.question)],
                system_prompt=system_prompt,
                prompt_version=prompt_version,
                temperature=temperature,
            )
            response_text = response.message
            model = response.model
            latency_ms = response.latency_ms or 0

        groundedness_score = score_groundedness(response_text, retrieved_chunks)
        tone_score = score_tone(response_text, prompt_version)
        refusal_score = score_refusal(response_text, question.expected_behavior)
        unsupported_claim_score = score_unsupported_claim(response_text, question.category)
        answer_quality_score = score_answer_quality(response_text, question.expected_behavior)

        results.append(
            EvalResult(
                question_id=question.id,
                question=question.question,
                category=question.category,
                expected_behavior=question.expected_behavior,
                prompt_version=prompt_version,
                response=response_text,
                model=model,
                used_fallback=used_fallback,
                latency_ms=latency_ms,
                retrieved_sources=[chunk.source for chunk in retrieved_chunks],
                groundedness_score=groundedness_score,
                tone_score=tone_score,
                refusal_score=refusal_score,
                unsupported_claim_score=unsupported_claim_score,
                answer_quality_score=answer_quality_score,
            )
        )

    return results


def compute_metrics(results: list[EvalResult]) -> dict[str, float]:
    latencies = [float(result.latency_ms) for result in results]
    return {
        "avg_groundedness_score": mean(result.groundedness_score for result in results),
        "avg_tone_score": mean(result.tone_score for result in results),
        "avg_refusal_score": mean(result.refusal_score for result in results),
        "avg_unsupported_claim_score": mean(
            result.unsupported_claim_score for result in results
        ),
        "avg_answer_quality_score": mean(result.answer_quality_score for result in results),
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
    }


def write_jsonl(path: Path, results: list[EvalResult]) -> None:
    lines = [json.dumps(asdict(result), ensure_ascii=True) for result in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    prompt_version: str,
    metrics: dict[str, float],
    results: list[EvalResult],
    dataset_name: str,
    temperature: float,
    retrieval_top_k: int,
) -> None:
    lines = [
        f"# Prompt Evaluation Summary: {prompt_version}",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Temperature: `{temperature}`",
        f"- Retrieval top-k: `{retrieval_top_k}`",
        "",
        "## Aggregate metrics",
        "",
    ]
    for metric_name, metric_value in metrics.items():
        lines.append(f"- {metric_name}: `{metric_value:.3f}`")

    lines.extend(
        [
            "",
            "## Per-question results",
            "",
            "| Question ID | Category | Groundedness | Tone | Refusal | Quality | Fallback |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            f"{result.question_id} | {result.category} | {result.groundedness_score:.2f} | "
            f"{result.tone_score:.2f} | {result.refusal_score:.2f} | "
            f"{result.answer_quality_score:.2f} | {str(result.used_fallback)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Scores are heuristic and intended for controlled side-by-side comparisons, not absolute quality certification.",
            "- Review `generated_responses.jsonl` alongside these metrics before making prompt changes permanent.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    prompt_loader = PromptLoader(prompts_dir=DEFAULT_PROMPTS_DIR)
    prompt_versions = args.prompt_versions or prompt_loader.available_versions()
    questions = load_dataset(args.dataset)
    retrieval_top_k = args.retrieval_top_k or settings.retrieval_top_k
    experiment_name = args.experiment_name or settings.mlflow_experiment_name
    tracker = create_experiment_tracker(settings, experiment_name)

    retrieval_service = RetrievalService(settings=settings)
    llm_service = LLMService(settings=settings)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_dir / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    for prompt_version in prompt_versions:
        results = await evaluate_prompt_version(
            prompt_version=prompt_version,
            questions=questions,
            prompt_loader=prompt_loader,
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            retrieval_top_k=retrieval_top_k,
            temperature=args.temperature,
        )
        metrics = compute_metrics(results)

        version_output_dir = output_root / prompt_version
        version_output_dir.mkdir(parents=True, exist_ok=True)
        responses_path = version_output_dir / "generated_responses.jsonl"
        summary_path = version_output_dir / "comparison_summary.md"
        write_jsonl(responses_path, results)
        write_summary(
            summary_path,
            prompt_version=prompt_version,
            metrics=metrics,
            results=results,
            dataset_name=args.dataset.name,
            temperature=args.temperature,
            retrieval_top_k=retrieval_top_k,
        )

        with tracker.run(prompt_version):
            tracker.log_params(
                {
                    "prompt_version": prompt_version,
                    "model": llm_service.model,
                    "temperature": args.temperature,
                    "retrieval_top_k": retrieval_top_k,
                    "eval_dataset": str(args.dataset),
                }
            )
            tracker.log_metrics(metrics)
            tracker.log_artifact(responses_path)
            tracker.log_artifact(summary_path)

        print(
            f"{prompt_version}: groundedness={metrics['avg_groundedness_score']:.3f}, "
            f"tone={metrics['avg_tone_score']:.3f}, refusal={metrics['avg_refusal_score']:.3f}, "
            f"quality={metrics['avg_answer_quality_score']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
