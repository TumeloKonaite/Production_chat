from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.prompts import PromptLoader
from app.services.chat.prompting import (
    build_chat_system_prompt,
    build_direct_fallback_text,
    should_use_direct_fallback,
)
from app.services.evals.eval_service import (
    EvalDatasetExample,
    ModelEvalAggregate,
    ModelEvalRecord,
    build_aggregate,
    score_groundedness,
    score_quality,
)
from app.services.llm import LLMChatMessage, LLMService
from app.services.retrieval import RetrievalService


@dataclass(frozen=True, slots=True)
class ModelExperimentRun:
    records: list[ModelEvalRecord]
    aggregate: ModelEvalAggregate


class ModelExperimentService:
    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
    ) -> None:
        self._prompt_loader = prompt_loader
        self._llm_service = llm_service
        self._retrieval_service = retrieval_service

    async def evaluate_model(
        self,
        *,
        model_config_id: str,
        examples: list[EvalDatasetExample],
        prompt_version: str,
        retrieval_config: str,
        retrieval_top_k: int,
        temperature: float,
    ) -> ModelExperimentRun:
        model_config = self._llm_service.get_model_config(model_config_id)
        base_prompt = self._prompt_loader.load(prompt_version)
        records: list[ModelEvalRecord] = []

        for example in examples:
            retrieved_chunks = self._retrieval_service.retrieve(
                example.question,
                top_k=retrieval_top_k,
            )
            used_fallback = should_use_direct_fallback(example.question, retrieved_chunks)

            if used_fallback:
                answer = build_direct_fallback_text(example.question)
                latency_ms = 0
                input_tokens = None
                output_tokens = None
                total_tokens = None
                estimated_cost_usd = None
            else:
                system_prompt = build_chat_system_prompt(
                    base_prompt=base_prompt,
                    message=example.question,
                    retrieved_chunks=retrieved_chunks,
                )
                response = await self._llm_service.generate_response(
                    [LLMChatMessage(role="user", content=example.question)],
                    system_prompt=system_prompt,
                    prompt_version=prompt_version,
                    retrieval_config=retrieval_config,
                    temperature=temperature,
                    model_config_id=model_config_id,
                )
                answer = response.message
                latency_ms = response.latency_ms or 0
                input_tokens = response.token_usage.input_tokens
                output_tokens = response.token_usage.output_tokens
                total_tokens = response.token_usage.total_tokens
                estimated_cost_usd = response.estimated_cost_usd

            quality_score, passed = score_quality(
                answer,
                expected_facts=example.expected_facts,
                expected_behavior=example.expected_behavior,
            )
            groundedness_score = score_groundedness(answer, retrieved_chunks)

            records.append(
                ModelEvalRecord(
                    eval_id=example.id,
                    model_config_id=model_config.config_id,
                    question=example.question,
                    category=example.category,
                    expected_facts=list(example.expected_facts),
                    expected_behavior=example.expected_behavior,
                    answer=answer,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                    quality_score=quality_score,
                    groundedness_score=groundedness_score,
                    passed=passed,
                    used_fallback=used_fallback,
                    retrieved_sources=[chunk.source for chunk in retrieved_chunks],
                )
            )

        aggregate = build_aggregate(
            model_config_id=model_config.config_id,
            model_provider=model_config.provider,
            model_name=model_config.model,
            records=records,
        )
        return ModelExperimentRun(records=records, aggregate=aggregate)
