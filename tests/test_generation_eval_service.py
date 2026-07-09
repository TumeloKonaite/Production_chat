from __future__ import annotations

import asyncio
from pathlib import Path

from app.domain.evals import JudgeEvaluation, JudgeMetricScore
from app.infrastructure.prompts import PromptLoader
from app.services.evals.generation_eval_service import GenerationEvalService
from app.services.llm import LLMGeneratedResponse, ModelConfig, TokenUsage
from evals.feedback.feedback_dataset import FEEDBACK_DATASET_SOURCE


class FakeLLMService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_model_config(self, model_config_id: str | None = None) -> ModelConfig:
        del model_config_id
        return ModelConfig(
            config_id="openrouter:anthropic/claude-3.5-sonnet",
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            input_cost_per_1m_tokens=3.0,
            output_cost_per_1m_tokens=15.0,
        )

    def get_model_base_url(self, model_config_id: str | None = None) -> str:
        del model_config_id
        return "https://openrouter.ai/api/v1"

    async def generate_response(self, messages, **kwargs) -> LLMGeneratedResponse:
        self.calls.append({"messages": messages, **kwargs})
        return LLMGeneratedResponse(
            message=(
                "Tumelo is a data scientist and software engineer focused on practical AI "
                "systems and backend APIs."
            ),
            model="anthropic/claude-3.5-sonnet",
            model_provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
            model_config_id="openrouter:anthropic/claude-3.5-sonnet",
            prompt_version=str(kwargs["prompt_version"]),
            retrieval_config=str(kwargs["retrieval_config"]),
            latency_ms=320,
            token_usage=TokenUsage(
                input_tokens=600,
                output_tokens=120,
                total_tokens=720,
            ),
            estimated_prompt_cost_usd=0.0018,
            estimated_completion_cost_usd=0.0018,
            estimated_cost_usd=0.0036,
        )


class FakeJudgeClient:
    async def evaluate(self, *, prompt: str, model_config_id: str | None = None):
        del prompt, model_config_id
        return (
            JudgeEvaluation(
                context_relevance=JudgeMetricScore(score=2, reason="Grounded."),
                faithfulness=JudgeMetricScore(score=2, reason="Supported."),
                answer_relevance=JudgeMetricScore(score=2, reason="Relevant."),
            ),
            TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            90,
            "judge-model",
        )


def test_generation_eval_service_uses_fixed_context_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "generation_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"profile_001","question":"What does Tumelo do?","category":"profile",'
                    '"context":[{"source":"profile.md","section":"Summary","content":"Tumelo is a data scientist and software engineer focused on practical AI systems and backend APIs."}],'
                    '"expected_facts":["data scientist","software engineer","practical AI systems","backend APIs"],'
                    '"expected_answer_points":["data scientist","software engineer"]}'
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_loader = PromptLoader(
        prompts_dir=Path("app") / "infrastructure" / "prompts" / "templates"
    )
    service = GenerationEvalService(
        prompt_loader=prompt_loader,
        llm_service=FakeLLMService(),
        judge_client=FakeJudgeClient(),
    )

    examples = service.load_dataset(dataset_path)
    run = asyncio.run(
        service.evaluate_dataset(
            examples=examples,
            prompt_version="v1_professional",
            judge_model_config_id="openai:gpt-4.1-mini",
            max_tokens=512,
        )
    )

    assert service._llm_service.calls[0]["max_tokens"] == 512
    assert len(run.records) == 1
    assert run.records[0].model_provider == "openrouter"
    assert run.records[0].model_base_url == "https://openrouter.ai/api/v1"
    assert run.records[0].fixed_context_sources == ["profile.md"]
    assert run.records[0].quality_score == 5
    assert run.records[0].prompt_tokens == 600
    assert run.records[0].estimated_cost_usd == 0.0036
    assert run.records[0].judge_evaluation is not None
    assert run.aggregate.pass_rate == 1.0
    assert run.aggregate.latency_ms_avg == 320
    assert run.aggregate.latency_ms_p50 == 320
    assert run.aggregate.latency_ms_p95 == 320
    assert run.aggregate.total_prompt_tokens == 600
    assert run.aggregate.total_completion_tokens == 120
    assert run.aggregate.total_tokens == 720
    assert run.aggregate.estimated_total_cost_usd == 0.0036
    assert run.aggregate.average_context_relevance == 2.0


def test_generation_eval_service_handles_feedback_dataset_with_judge_scoring(tmp_path) -> None:
    dataset_path = tmp_path / "feedback_generation_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                (
                    '{"id":"feedback_trace_001","question":"What does Tumelo do?",'
                    '"actual_answer":"Tumelo is mainly a frontend developer.",'
                    '"expected_facts":[],"expected_answer_points":[],"expected_source_documents":["profile.md"],'
                    '"feedback_rating":"negative","feedback_reason":"incorrect_answer",'
                    '"feedback_comment":"The original answer missed the backend API work.",'
                    f'"source":"{FEEDBACK_DATASET_SOURCE}","trace_id":"trace-001","created_at":"2026-07-06T20:00:00Z",'
                    '"context":[{"source":"profile.md","section":"Summary","content":"Tumelo is a data scientist and software engineer focused on practical AI systems and backend APIs."}]}'
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_loader = PromptLoader(
        prompts_dir=Path("app") / "infrastructure" / "prompts" / "templates"
    )
    service = GenerationEvalService(
        prompt_loader=prompt_loader,
        llm_service=FakeLLMService(),
        judge_client=FakeJudgeClient(),
    )

    examples = service.load_dataset(dataset_path)
    run = asyncio.run(
        service.evaluate_dataset(
            examples=examples,
            prompt_version="v1_professional",
            judge_model_config_id="openai:gpt-4.1-mini",
        )
    )

    assert examples[0].dataset_source == FEEDBACK_DATASET_SOURCE
    assert run.records[0].dataset_source == FEEDBACK_DATASET_SOURCE
    assert run.records[0].requires_human_label is True
    assert run.records[0].skipped_missing_labels is False
    assert run.records[0].scored is True
    assert run.records[0].judge_score == 2.0
    assert run.aggregate.scored_examples == 1
    assert run.aggregate.skipped_examples == 0
    assert run.aggregate.feedback_metrics == {
        "feedback_cases_total": 1,
        "feedback_cases_answered": 1,
        "feedback_cases_skipped_missing_labels": 0,
        "feedback_pass_rate": 1.0,
        "feedback_regression_failures": 0,
        "feedback_avg_judge_score": 2.0,
    }
