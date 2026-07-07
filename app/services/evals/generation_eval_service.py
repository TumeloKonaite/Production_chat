from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

from app.domain.evals import JudgeEvaluation
from app.infrastructure.llm import JudgeClient
from app.infrastructure.prompts import PromptLoader
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    format_retrieved_context,
    should_use_direct_fallback,
)
from app.services.evals.eval_service import score_groundedness, score_quality
from app.services.llm import LLMChatMessage, LLMService
from app.services.retrieval import RetrievedChunk
from evals.feedback.feedback_dataset import FEEDBACK_DATASET_SOURCE


@dataclass(frozen=True, slots=True)
class FixedContextItem:
    source: str
    section: str
    content: str
    similarity: float = 1.0


@dataclass(frozen=True, slots=True)
class GenerationEvalExample:
    id: str
    question: str
    category: str
    context: list[FixedContextItem]
    expected_facts: list[str]
    expected_answer_points: list[str]
    expected_behavior: str | None = None
    expected_answer: str | None = None
    actual_answer: str | None = None
    dataset_source: str = "generation_eval"
    feedback_rating: str | None = None
    feedback_reason: str | None = None
    feedback_comment: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GenerationEvalRecord:
    eval_id: str
    model_config_id: str
    model_provider: str
    model_name: str
    model_base_url: str
    dataset_source: str
    trace_id: str | None
    question: str
    category: str
    expected_facts: list[str]
    expected_answer_points: list[str]
    expected_answer: str | None
    expected_behavior: str | None
    actual_answer: str | None
    feedback_rating: str | None
    feedback_reason: str | None
    feedback_comment: str | None
    generated_answer: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    estimated_prompt_cost_usd: float | None
    estimated_completion_cost_usd: float | None
    estimated_cost_usd: float | None
    quality_score: int
    scored: bool
    groundedness_score: float
    passed: bool
    requires_human_label: bool
    skipped_missing_labels: bool
    expected_facts_coverage: float | None
    judge_score: float | None
    used_fallback: bool
    fixed_context_sources: list[str]
    judge_evaluation: dict[str, dict[str, int | str]] | None = None


@dataclass(frozen=True, slots=True)
class GenerationEvalAggregate:
    model_config_id: str
    model_provider: str
    model_name: str
    model_base_url: str
    total_examples: int
    scored_examples: int
    skipped_examples: int
    passed_examples: int
    failed_examples: int
    pass_rate: float
    average_quality_score: float
    average_groundedness_score: float
    average_context_relevance: float | None
    average_faithfulness: float | None
    average_answer_relevance: float | None
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_tokens_per_response: float
    responses_with_usage: int
    estimated_prompt_cost_usd: float | None
    estimated_completion_cost_usd: float | None
    estimated_total_cost_usd: float | None
    average_cost_per_response_usd: float | None
    responses_with_cost_estimate: int
    feedback_metrics: dict[str, float | int] | None = None


@dataclass(frozen=True, slots=True)
class GenerationEvalRun:
    records: list[GenerationEvalRecord]
    aggregate: GenerationEvalAggregate


class GenerationEvalService:
    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        llm_service: LLMService,
        judge_client: JudgeClient | None = None,
    ) -> None:
        self._prompt_loader = prompt_loader
        self._llm_service = llm_service
        self._judge_client = judge_client

    def load_dataset(self, path: Path) -> list[GenerationEvalExample]:
        examples: list[GenerationEvalExample] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(
                GenerationEvalExample(
                    id=str(payload["id"]),
                    question=str(payload["question"]),
                    category=str(payload.get("category") or payload.get("source") or "generation_eval"),
                    context=_parse_context_items(payload.get("context"), example_id=str(payload["id"])),
                    expected_facts=[str(item) for item in payload.get("expected_facts", [])],
                    expected_answer_points=[
                        str(item) for item in payload.get("expected_answer_points", [])
                    ],
                    expected_answer=(
                        str(payload["expected_answer"])
                        if payload.get("expected_answer") is not None
                        else None
                    ),
                    actual_answer=(
                        str(payload["actual_answer"])
                        if payload.get("actual_answer") is not None
                        else None
                    ),
                    expected_behavior=(
                        str(payload["expected_behavior"])
                        if payload.get("expected_behavior") is not None
                        else None
                    ),
                    dataset_source=str(payload.get("source") or "generation_eval"),
                    feedback_rating=(
                        str(payload["feedback_rating"])
                        if payload.get("feedback_rating") is not None
                        else None
                    ),
                    feedback_reason=(
                        str(payload["feedback_reason"])
                        if payload.get("feedback_reason") is not None
                        else None
                    ),
                    feedback_comment=(
                        str(payload["feedback_comment"])
                        if payload.get("feedback_comment") is not None
                        else None
                    ),
                    trace_id=str(payload["trace_id"]) if payload.get("trace_id") is not None else None,
                    metadata=(
                        dict(payload["metadata"])
                        if isinstance(payload.get("metadata"), dict)
                        else None
                    ),
                )
            )
        return examples

    async def evaluate_dataset(
        self,
        *,
        examples: list[GenerationEvalExample],
        prompt_version: str,
        model_config_id: str | None = None,
        judge_model_config_id: str | None = None,
        temperature: float = 0.2,
    ) -> GenerationEvalRun:
        model_config = self._llm_service.get_model_config(model_config_id)
        model_base_url = self._llm_service.get_model_base_url(model_config.config_id)
        base_prompt = self._prompt_loader.load(prompt_version)
        records: list[GenerationEvalRecord] = []

        for example in examples:
            retrieved_chunks = _build_retrieved_chunks(example)
            used_fallback = should_use_direct_fallback(example.question, retrieved_chunks)
            scoring_facts = _build_scoring_facts(example)
            has_labels = bool(scoring_facts or example.expected_answer_points)
            requires_human_label = not has_labels

            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            estimated_prompt_cost_usd = None
            estimated_completion_cost_usd = None
            estimated_cost_usd = None

            if used_fallback:
                answer = build_direct_fallback_text(example.question)
                latency_ms = 0
            else:
                response = await self._llm_service.generate_response(
                    [LLMChatMessage(role="user", content=example.question)],
                    system_prompt=build_chat_system_prompt(
                        base_prompt=base_prompt,
                        message=example.question,
                        retrieved_chunks=retrieved_chunks,
                    ),
                    prompt_version=prompt_version,
                    retrieval_config="fixed_context",
                    temperature=temperature,
                    model_config_id=model_config.config_id,
                )
                answer = response.message
                latency_ms = response.latency_ms or 0
                prompt_tokens = response.token_usage.input_tokens
                completion_tokens = response.token_usage.output_tokens
                total_tokens = response.token_usage.total_tokens
                estimated_prompt_cost_usd = response.estimated_prompt_cost_usd
                estimated_completion_cost_usd = response.estimated_completion_cost_usd
                estimated_cost_usd = response.estimated_cost_usd

            quality_score = 0
            passed = False
            scored = False
            skipped_missing_labels = False
            expected_facts_coverage = None
            if has_labels:
                quality_score, passed = score_quality(
                    answer,
                    expected_facts=scoring_facts,
                    expected_behavior=example.expected_behavior,
                )
                scored = True
                expected_facts_coverage = _measure_expected_fact_coverage(
                    answer,
                    scoring_facts,
                )
            groundedness_score = score_groundedness(answer, retrieved_chunks)

            judge_evaluation = await self._evaluate_with_judge(
                example=example,
                retrieved_chunks=retrieved_chunks,
                answer=answer,
                judge_model_config_id=judge_model_config_id,
            )
            judge_score = _extract_judge_score(judge_evaluation)
            if not has_labels:
                if judge_evaluation is not None:
                    scored = True
                    passed = _judge_indicates_pass(judge_evaluation)
                    quality_score = int(judge_score or 0)
                else:
                    skipped_missing_labels = True

            records.append(
                GenerationEvalRecord(
                    eval_id=example.id,
                    model_config_id=model_config.config_id,
                    model_provider=model_config.provider,
                    model_name=model_config.model,
                    model_base_url=model_base_url,
                    dataset_source=example.dataset_source,
                    trace_id=example.trace_id,
                    question=example.question,
                    category=example.category,
                    expected_facts=list(example.expected_facts),
                    expected_answer_points=list(example.expected_answer_points),
                    expected_answer=example.expected_answer,
                    expected_behavior=example.expected_behavior,
                    actual_answer=example.actual_answer,
                    feedback_rating=example.feedback_rating,
                    feedback_reason=example.feedback_reason,
                    feedback_comment=example.feedback_comment,
                    generated_answer=answer,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_prompt_cost_usd=estimated_prompt_cost_usd,
                    estimated_completion_cost_usd=estimated_completion_cost_usd,
                    estimated_cost_usd=estimated_cost_usd,
                    quality_score=quality_score,
                    scored=scored,
                    groundedness_score=groundedness_score,
                    passed=passed,
                    requires_human_label=requires_human_label,
                    skipped_missing_labels=skipped_missing_labels,
                    expected_facts_coverage=expected_facts_coverage,
                    judge_score=judge_score,
                    used_fallback=used_fallback,
                    fixed_context_sources=[chunk.source for chunk in retrieved_chunks],
                    judge_evaluation=_judge_as_json(judge_evaluation),
                )
            )

        return GenerationEvalRun(
            records=records,
            aggregate=_build_aggregate(
                model_config_id=model_config.config_id,
                model_provider=model_config.provider,
                model_name=model_config.model,
                model_base_url=model_base_url,
                records=records,
            ),
        )

    def records_as_json(self, records: list[GenerationEvalRecord]) -> list[dict[str, object]]:
        return [asdict(record) for record in records]

    def render_summary(self, aggregate: GenerationEvalAggregate) -> str:
        lines = [
            "Generation Eval Summary",
            "",
            f"Model: {aggregate.model_config_id}",
            f"Provider: {aggregate.model_provider}",
            f"Base URL: {aggregate.model_base_url}",
            f"Pass rate: {aggregate.pass_rate:.0%}",
            f"Scored/skipped: {aggregate.scored_examples} / {aggregate.skipped_examples}",
            f"Avg quality: {aggregate.average_quality_score:.2f}",
            f"Avg groundedness: {aggregate.average_groundedness_score:.2f}",
            f"Latency avg/p50/p95: {aggregate.latency_ms_avg:.0f}ms / {aggregate.latency_ms_p50:.0f}ms / {aggregate.latency_ms_p95:.0f}ms",
            f"Tokens total: {aggregate.total_tokens} (avg {aggregate.avg_tokens_per_response:.2f})",
        ]
        if aggregate.average_context_relevance is not None:
            lines.append(
                "Judge avg context/faithfulness/answer relevance: "
                f"{aggregate.average_context_relevance:.2f} / "
                f"{aggregate.average_faithfulness:.2f} / "
                f"{aggregate.average_answer_relevance:.2f}"
            )
        if aggregate.estimated_total_cost_usd is not None:
            lines.append(
                "Estimated cost prompt/completion/total: "
                f"${aggregate.estimated_prompt_cost_usd:.6f} / "
                f"${aggregate.estimated_completion_cost_usd:.6f} / "
                f"${aggregate.estimated_total_cost_usd:.6f}"
            )
        else:
            lines.append("Estimated cost: unavailable")
        return "\n".join(lines) + "\n"

    async def _evaluate_with_judge(
        self,
        *,
        example: GenerationEvalExample,
        retrieved_chunks: list[RetrievedChunk],
        answer: str,
        judge_model_config_id: str | None,
    ) -> JudgeEvaluation | None:
        if self._judge_client is None or judge_model_config_id is None:
            return None

        prompt = "\n\n".join(
            [
                "Evaluate the generated answer using the fixed retrieved context only.",
                f"Question:\n{example.question}",
                (
                    f"Original production answer:\n{example.actual_answer}"
                    if example.actual_answer
                    else "Original production answer:\nNot provided."
                ),
                (
                    f"Feedback note:\n{example.feedback_comment}"
                    if example.feedback_comment
                    else "Feedback note:\nNone"
                ),
                "Expected answer points:\n"
                + (
                    "\n".join(f"- {point}" for point in _build_judge_reference_points(example))
                    if _build_judge_reference_points(example)
                    else "- None"
                ),
                f"Expected behavior:\n{example.expected_behavior or 'answer_normally'}",
                "Fixed retrieved context:\n" + format_retrieved_context(retrieved_chunks),
                f"Generated answer:\n{answer}",
                (
                    "Return JSON with context_relevance, faithfulness, and answer_relevance, "
                    "each containing score (0-2) and reason."
                ),
            ]
        )
        evaluation, _usage, _latency_ms, _model = await self._judge_client.evaluate(
            prompt=prompt,
            model_config_id=judge_model_config_id,
        )
        return evaluation


def _parse_context_items(raw_context: object, *, example_id: str) -> list[FixedContextItem]:
    if raw_context is None:
        return []
    if not isinstance(raw_context, list):
        raise ValueError(f"Generation eval example {example_id} context must be a list when provided.")
    if not raw_context:
        return []

    items: list[FixedContextItem] = []
    for index, item in enumerate(raw_context, start=1):
        if isinstance(item, str) and item.strip():
            items.append(
                FixedContextItem(
                    source=f"fixed_context_{index}",
                    section=f"context_{index}",
                    content=item.strip(),
                )
            )
            continue
        if isinstance(item, dict):
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Generation eval example {example_id} context item {index} must include non-empty content."
                )
            source = item.get("source")
            section = item.get("section")
            similarity = item.get("similarity", 1.0)
            items.append(
                FixedContextItem(
                    source=source.strip() if isinstance(source, str) and source.strip() else f"fixed_context_{index}",
                    section=section.strip() if isinstance(section, str) and section.strip() else f"context_{index}",
                    content=content.strip(),
                    similarity=float(similarity) if isinstance(similarity, int | float) else 1.0,
                )
            )
            continue
        raise ValueError(
            f"Generation eval example {example_id} context item {index} must be a string or object."
        )
    return items


def _build_retrieved_chunks(example: GenerationEvalExample) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=f"{example.id}_context_{index}",
            source=item.source,
            section=item.section,
            content=item.content,
            similarity=item.similarity,
            metadata={
                "source": item.source,
                "section": item.section,
                "source_type": "fixed_context",
                "eval_id": example.id,
            },
        )
        for index, item in enumerate(example.context, start=1)
    ]


def _build_scoring_facts(example: GenerationEvalExample) -> list[str]:
    if example.expected_facts:
        return list(example.expected_facts)
    if example.expected_answer:
        return [example.expected_answer]
    return []


def _build_judge_reference_points(example: GenerationEvalExample) -> list[str]:
    if example.expected_answer_points:
        return list(example.expected_answer_points)
    if example.expected_answer:
        return [example.expected_answer]
    if example.feedback_comment:
        return [example.feedback_comment]
    return []


def _judge_as_json(
    evaluation: JudgeEvaluation | None,
) -> dict[str, dict[str, int | str]] | None:
    if evaluation is None:
        return None
    return {
        "context_relevance": asdict(evaluation.context_relevance),
        "faithfulness": asdict(evaluation.faithfulness),
        "answer_relevance": asdict(evaluation.answer_relevance),
    }


def _build_aggregate(
    *,
    model_config_id: str,
    model_provider: str,
    model_name: str,
    model_base_url: str,
    records: list[GenerationEvalRecord],
) -> GenerationEvalAggregate:
    latencies = sorted(float(record.latency_ms) for record in records)
    total_examples = len(records)
    scored_records = [record for record in records if record.scored]
    feedback_records = [
        record for record in records if record.dataset_source == FEEDBACK_DATASET_SOURCE
    ]
    passed_examples = sum(1 for record in scored_records if record.passed)
    failed_examples = len(scored_records) - passed_examples
    total_prompt_tokens = sum(record.prompt_tokens or 0 for record in records)
    total_completion_tokens = sum(record.completion_tokens or 0 for record in records)
    total_tokens = sum(record.total_tokens or 0 for record in records)
    responses_with_usage = sum(1 for record in records if record.total_tokens is not None)
    prompt_costs = [record.estimated_prompt_cost_usd for record in records if record.estimated_prompt_cost_usd is not None]
    completion_costs = [
        record.estimated_completion_cost_usd
        for record in records
        if record.estimated_completion_cost_usd is not None
    ]
    total_costs = [record.estimated_cost_usd for record in records if record.estimated_cost_usd is not None]
    responses_with_cost_estimate = len(total_costs)

    latency_ms_avg = mean(latencies) if latencies else 0.0
    latency_ms_p50 = _percentile(latencies, percentile=0.50)
    latency_ms_p95 = _percentile(latencies, percentile=0.95)

    judge_context_scores: list[float] = []
    judge_faithfulness_scores: list[float] = []
    judge_answer_scores: list[float] = []
    for record in records:
        if record.judge_evaluation is None:
            continue
        judge_context_scores.append(float(record.judge_evaluation["context_relevance"]["score"]))
        judge_faithfulness_scores.append(float(record.judge_evaluation["faithfulness"]["score"]))
        judge_answer_scores.append(float(record.judge_evaluation["answer_relevance"]["score"]))

    feedback_metrics = _build_feedback_metrics(feedback_records)

    return GenerationEvalAggregate(
        model_config_id=model_config_id,
        model_provider=model_provider,
        model_name=model_name,
        model_base_url=model_base_url,
        total_examples=total_examples,
        scored_examples=len(scored_records),
        skipped_examples=sum(1 for record in records if record.skipped_missing_labels),
        passed_examples=passed_examples,
        failed_examples=failed_examples,
        pass_rate=(passed_examples / len(scored_records)) if scored_records else 0.0,
        average_quality_score=(
            mean(record.quality_score for record in scored_records) if scored_records else 0.0
        ),
        average_groundedness_score=(
            mean(record.groundedness_score for record in records) if records else 0.0
        ),
        average_context_relevance=mean(judge_context_scores) if judge_context_scores else None,
        average_faithfulness=mean(judge_faithfulness_scores) if judge_faithfulness_scores else None,
        average_answer_relevance=mean(judge_answer_scores) if judge_answer_scores else None,
        latency_ms_avg=latency_ms_avg,
        latency_ms_p50=latency_ms_p50,
        latency_ms_p95=latency_ms_p95,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        avg_tokens_per_response=(total_tokens / total_examples) if total_examples else 0.0,
        responses_with_usage=responses_with_usage,
        estimated_prompt_cost_usd=round(sum(prompt_costs), 6) if prompt_costs else None,
        estimated_completion_cost_usd=(
            round(sum(completion_costs), 6) if completion_costs else None
        ),
        estimated_total_cost_usd=round(sum(total_costs), 6) if total_costs else None,
        average_cost_per_response_usd=(
            round(sum(total_costs) / total_examples, 6) if total_costs and total_examples else None
        ),
        responses_with_cost_estimate=responses_with_cost_estimate,
        feedback_metrics=feedback_metrics,
    )


def _percentile(values: list[float], *, percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * percentile))))
    return values[index]


def _measure_expected_fact_coverage(answer: str, expected_facts: list[str]) -> float | None:
    if not expected_facts:
        return None
    normalized = answer.casefold()
    matches = sum(1 for fact in expected_facts if fact.casefold() in normalized)
    return matches / len(expected_facts)


def _extract_judge_score(evaluation: JudgeEvaluation | None) -> float | None:
    if evaluation is None:
        return None
    return float(evaluation.answer_relevance.score)


def _judge_indicates_pass(evaluation: JudgeEvaluation) -> bool:
    return (
        evaluation.answer_relevance.score >= 1
        and evaluation.faithfulness.score >= 1
    )


def _build_feedback_metrics(records: list[GenerationEvalRecord]) -> dict[str, float | int] | None:
    if not records:
        return None

    scored_records = [record for record in records if record.scored]
    judge_scores = [record.judge_score for record in records if record.judge_score is not None]
    expected_facts_coverages = [
        record.expected_facts_coverage
        for record in records
        if record.expected_facts_coverage is not None
    ]
    feedback_metrics: dict[str, float | int] = {
        "feedback_cases_total": len(records),
        "feedback_cases_answered": len(records),
        "feedback_cases_skipped_missing_labels": sum(
            1 for record in records if record.skipped_missing_labels
        ),
    }
    if scored_records:
        feedback_metrics["feedback_pass_rate"] = (
            sum(1 for record in scored_records if record.passed) / len(scored_records)
        )
        feedback_metrics["feedback_regression_failures"] = sum(
            1 for record in scored_records if not record.passed
        )
    if judge_scores:
        feedback_metrics["feedback_avg_judge_score"] = mean(judge_scores)
    if expected_facts_coverages:
        feedback_metrics["feedback_expected_facts_coverage"] = mean(expected_facts_coverages)
    return feedback_metrics
