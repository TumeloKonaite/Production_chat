from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RagEvalDatasetExample:
    id: str
    question: str
    expected_source_documents: list[str]
    expected_answer_points: list[str]
    category: str
    difficulty: str | None = None
    notes: str | None = None
    expected_behavior: str | None = None


@dataclass(frozen=True, slots=True)
class RagEvalRetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(frozen=True, slots=True)
class JudgeMetricScore:
    score: int
    reason: str


@dataclass(frozen=True, slots=True)
class JudgeEvaluation:
    context_relevance: JudgeMetricScore
    faithfulness: JudgeMetricScore
    answer_relevance: JudgeMetricScore


@dataclass(frozen=True, slots=True)
class RagEvalQuestionResult:
    question_id: str
    question: str
    category: str
    generated_answer: str
    expected_source_documents: list[str]
    retrieved_source_documents: list[str]
    expected_answer_points: list[str]
    expected_behavior: str | None
    retrieval_metrics: RagEvalRetrievalMetrics
    judge_evaluation: JudgeEvaluation
    latency_ms: int
    token_usage: dict[str, int | None]
    answer_estimated_prompt_cost_usd: float | None = None
    answer_estimated_completion_cost_usd: float | None = None
    answer_estimated_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class RagEvalRunSummary:
    run_name: str
    dataset_name: str
    model_name: str
    model_config_id: str
    prompt_version: str
    top_k: int
    retrieval_config: dict[str, object]
    total_questions: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_mrr: float
    avg_ndcg_at_k: float
    avg_context_relevance: float
    avg_faithfulness: float
    avg_answer_relevance: float
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    estimated_total_cost_usd: float | None
    average_cost_per_question_usd: float | None
    questions_with_cost_estimate: int
