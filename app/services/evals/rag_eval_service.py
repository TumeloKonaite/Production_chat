from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from app.domain.evals import (
    RagEvalDatasetExample,
    RagEvalQuestionResult,
    RagEvalRetrievalMetrics,
    RagEvalRunSummary,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.infrastructure.llm import JudgeClient
from app.infrastructure.prompts import PromptLoader
from app.repositories import EvalRepository
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    format_retrieved_context,
    should_use_direct_fallback,
)
from app.services.llm import LLMChatMessage, LLMService
from app.services.retrieval import RetrievedChunk, RetrievalService


class RagEvalService:
    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        judge_client: JudgeClient,
        eval_repository: EvalRepository | None = None,
    ) -> None:
        self._prompt_loader = prompt_loader
        self._llm_service = llm_service
        self._retrieval_service = retrieval_service
        self._judge_client = judge_client
        self._eval_repository = eval_repository

    def load_dataset(self, path: Path) -> list[RagEvalDatasetExample]:
        examples: list[RagEvalDatasetExample] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            examples.append(
                RagEvalDatasetExample(
                    id=str(payload["id"]),
                    question=str(payload["question"]),
                    expected_source_documents=[
                        str(item) for item in payload.get("expected_source_documents", [])
                    ],
                    expected_answer_points=[
                        str(item) for item in payload.get("expected_answer_points", [])
                    ],
                    category=str(payload["category"]),
                    difficulty=(
                        str(payload["difficulty"])
                        if payload.get("difficulty") is not None
                        else None
                    ),
                    notes=str(payload["notes"]) if payload.get("notes") is not None else None,
                    expected_behavior=(
                        str(payload["expected_behavior"])
                        if payload.get("expected_behavior") is not None
                        else None
                    ),
                )
            )
        return examples

    async def evaluate_dataset(
        self,
        *,
        dataset_name: str,
        examples: list[RagEvalDatasetExample],
        run_name: str,
        prompt_version: str,
        model_config_id: str,
        judge_model_config_id: str | None,
        top_k: int,
        retrieval_config: dict[str, object],
        judge_prompt_template: str,
        temperature: float = 0.2,
        persist_results: bool = True,
    ) -> tuple[RagEvalRunSummary, list[RagEvalQuestionResult]]:
        base_prompt = self._prompt_loader.load(prompt_version)
        model_config = self._llm_service.get_model_config(model_config_id)
        results: list[RagEvalQuestionResult] = []

        for example in examples:
            retrieved_chunks = self._retrieval_service.retrieve(example.question, top_k=top_k)
            retrieved_sources = [chunk.source for chunk in retrieved_chunks]
            use_direct_fallback = should_use_direct_fallback(example.question, retrieved_chunks)

            if use_direct_fallback:
                answer = build_direct_fallback_text(example.question)
                answer_latency_ms = 0
                answer_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                answer_estimated_prompt_cost_usd = 0.0
                answer_estimated_completion_cost_usd = 0.0
                answer_estimated_cost_usd = 0.0
            else:
                response = await self._llm_service.generate_response(
                    [LLMChatMessage(role="user", content=example.question)],
                    system_prompt=build_chat_system_prompt(
                        base_prompt=base_prompt,
                        message=example.question,
                        retrieved_chunks=retrieved_chunks,
                    ),
                    prompt_version=prompt_version,
                    retrieval_config=str(retrieval_config.get("name", "default")),
                    temperature=temperature,
                    model_config_id=model_config_id,
                )
                answer = response.message
                answer_latency_ms = response.latency_ms or 0
                answer_token_usage = {
                    "input_tokens": response.token_usage.input_tokens,
                    "output_tokens": response.token_usage.output_tokens,
                    "total_tokens": response.token_usage.total_tokens,
                }
                answer_estimated_prompt_cost_usd = response.estimated_prompt_cost_usd
                answer_estimated_completion_cost_usd = response.estimated_completion_cost_usd
                answer_estimated_cost_usd = response.estimated_cost_usd

            retrieval_metrics = RagEvalRetrievalMetrics(
                precision_at_k=precision_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    k=top_k,
                ),
                recall_at_k=recall_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    k=top_k,
                ),
                mrr=mean_reciprocal_rank(
                    retrieved_sources,
                    example.expected_source_documents,
                ),
                ndcg_at_k=ndcg_at_k(
                    retrieved_sources,
                    example.expected_source_documents,
                    k=top_k,
                ),
            )

            judge_prompt = self._build_judge_prompt(
                judge_prompt_template=judge_prompt_template,
                example=example,
                retrieved_chunks=retrieved_chunks,
                answer=answer,
            )
            judge_evaluation, judge_usage, judge_latency_ms, _judge_model_name = (
                await self._judge_client.evaluate(
                    prompt=judge_prompt,
                    model_config_id=judge_model_config_id,
                )
            )

            results.append(
                RagEvalQuestionResult(
                    question_id=example.id,
                    question=example.question,
                    category=example.category,
                    generated_answer=answer,
                    expected_source_documents=list(example.expected_source_documents),
                    retrieved_source_documents=retrieved_sources,
                    expected_answer_points=list(example.expected_answer_points),
                    expected_behavior=example.expected_behavior,
                    retrieval_metrics=retrieval_metrics,
                    judge_evaluation=judge_evaluation,
                    latency_ms=answer_latency_ms + judge_latency_ms,
                    token_usage={
                        "answer_input_tokens": answer_token_usage["input_tokens"],
                        "answer_output_tokens": answer_token_usage["output_tokens"],
                        "answer_total_tokens": answer_token_usage["total_tokens"],
                        "judge_input_tokens": judge_usage.input_tokens,
                        "judge_output_tokens": judge_usage.output_tokens,
                        "judge_total_tokens": judge_usage.total_tokens,
                    },
                    answer_estimated_prompt_cost_usd=answer_estimated_prompt_cost_usd,
                    answer_estimated_completion_cost_usd=answer_estimated_completion_cost_usd,
                    answer_estimated_cost_usd=answer_estimated_cost_usd,
                )
            )

        latencies = sorted(float(result.latency_ms) for result in results)
        answer_costs = [
            result.answer_estimated_cost_usd
            for result in results
            if result.answer_estimated_cost_usd is not None
        ]
        summary = RagEvalRunSummary(
            run_name=run_name,
            dataset_name=dataset_name,
            model_name=model_config.model,
            model_config_id=model_config.config_id,
            prompt_version=prompt_version,
            top_k=top_k,
            retrieval_config=retrieval_config,
            total_questions=len(results),
            avg_precision_at_k=mean(result.retrieval_metrics.precision_at_k for result in results),
            avg_recall_at_k=mean(result.retrieval_metrics.recall_at_k for result in results),
            avg_mrr=mean(result.retrieval_metrics.mrr for result in results),
            avg_ndcg_at_k=mean(result.retrieval_metrics.ndcg_at_k for result in results),
            avg_context_relevance=mean(
                result.judge_evaluation.context_relevance.score for result in results
            ),
            avg_faithfulness=mean(result.judge_evaluation.faithfulness.score for result in results),
            avg_answer_relevance=mean(
                result.judge_evaluation.answer_relevance.score for result in results
            ),
            latency_ms_avg=mean(latencies) if latencies else 0.0,
            latency_ms_p50=_percentile(latencies, percentile=0.50),
            latency_ms_p95=_percentile(latencies, percentile=0.95),
            estimated_total_cost_usd=round(sum(answer_costs), 6) if answer_costs else None,
            average_cost_per_question_usd=(
                round(sum(answer_costs) / len(results), 6) if answer_costs and results else None
            ),
            questions_with_cost_estimate=len(answer_costs),
        )

        if persist_results and self._eval_repository is not None:
            self._eval_repository.create_run(summary=summary, results=results)

        return summary, results

    def render_summary_table(self, summaries: list[RagEvalRunSummary]) -> str:
        header = (
            "Run | Model | Prompt | Top K | Recall@K | Precision@K | MRR | NDCG@K | "
            "Context Relevance | Faithfulness | Answer Relevance | Latency | Cost"
        )
        separator = "-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --"
        rows = [
            " | ".join(
                [
                    summary.run_name,
                    summary.model_name,
                    summary.prompt_version,
                    str(summary.top_k),
                    f"{summary.avg_recall_at_k:.2f}",
                    f"{summary.avg_precision_at_k:.2f}",
                    f"{summary.avg_mrr:.2f}",
                    f"{summary.avg_ndcg_at_k:.2f}",
                    f"{summary.avg_context_relevance:.2f}",
                    f"{summary.avg_faithfulness:.2f}",
                    f"{summary.avg_answer_relevance:.2f}",
                    f"{summary.latency_ms_avg:.0f}ms",
                    (
                        f"${summary.estimated_total_cost_usd:.6f}"
                        if summary.estimated_total_cost_usd is not None
                        else "n/a"
                    ),
                ]
            )
            for summary in summaries
        ]
        return "\n".join([header, separator, *rows]) + "\n"

    def results_as_json(self, results: list[RagEvalQuestionResult]) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for result in results:
            result_payload = asdict(result)
            result_payload["retrieval_metrics"] = asdict(result.retrieval_metrics)
            result_payload["judge_evaluation"] = {
                "context_relevance": asdict(result.judge_evaluation.context_relevance),
                "faithfulness": asdict(result.judge_evaluation.faithfulness),
                "answer_relevance": asdict(result.judge_evaluation.answer_relevance),
            }
            payloads.append(result_payload)
        return payloads

    def _build_judge_prompt(
        self,
        *,
        judge_prompt_template: str,
        example: RagEvalDatasetExample,
        retrieved_chunks: list[RetrievedChunk],
        answer: str,
    ) -> str:
        expected_sources = ", ".join(example.expected_source_documents) or "None"
        expected_points = (
            "\n".join(f"- {point}" for point in example.expected_answer_points)
            if example.expected_answer_points
            else "- None"
        )
        expected_behavior = example.expected_behavior or "answer_normally"

        evaluation_payload = "\n\n".join(
            [
                f"Question:\n{example.question}",
                f"Expected source documents:\n{expected_sources}",
                f"Expected answer points:\n{expected_points}",
                f"Expected behavior:\n{expected_behavior}",
                "Retrieved context:\n" + format_retrieved_context(retrieved_chunks),
                f"Generated answer:\n{answer}",
            ]
        )
        return "\n\n".join([judge_prompt_template.strip(), evaluation_payload]).strip()


def _percentile(values: list[float], *, percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * percentile))))
    return values[index]
