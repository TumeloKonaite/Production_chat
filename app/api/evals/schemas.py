from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvalMode = Literal["retrieval", "generation", "rag", "matrix"]
EvalRunStatus = Literal[
    "queued",
    "running",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
]


class RetrievalEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_provider: str | None = Field(default=None, min_length=1, max_length=100)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=300)
    embedding_dimension: int | None = Field(default=None, gt=0)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    retriever_type: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=5, gt=0, le=50)
    query_rewriting_enabled: bool | None = None
    query_rewrite_model: str | None = Field(default=None, min_length=1, max_length=300)
    query_rewrite_prompt_version: str | None = Field(default=None, min_length=1, max_length=100)
    query_rewrite_temperature: float | None = Field(default=None, ge=0.0)
    reranking_enabled: bool | None = None
    reranker_type: Literal["none", "llm"] | None = None
    reranker_model: str | None = Field(default=None, min_length=1, max_length=300)
    reranker_initial_top_k: int | None = Field(default=None, gt=0, le=200)
    notes: str | None = Field(default=None, min_length=1, max_length=2000)


class GenerationEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_model: str | None = Field(default=None, min_length=1, max_length=300)
    provider: Literal["openai", "openrouter"] | None = None
    model_config_id: str | None = Field(default=None, min_length=1, max_length=300)
    judge_model: str | None = Field(default=None, min_length=1, max_length=300)
    judge_model_config_id: str | None = Field(default=None, min_length=1, max_length=300)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=100)
    temperature: float = Field(default=0.2, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    dataset_version: str | None = Field(default=None, min_length=1, max_length=100)
    notes: str | None = Field(default=None, min_length=1, max_length=2000)


class RagEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalEvalRequest
    generation: GenerationEvalRequest


class MatrixEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str = Field(min_length=1, max_length=200)
    dry_run: bool = False
    confirm_full_run: bool = False


class EvalRunQueuedResponse(BaseModel):
    run_id: str
    status: Literal["queued"]
    mode: EvalMode
    status_url: str
    suite: str | None = None


class MatrixDryRunResponse(BaseModel):
    suite: str
    mode: Literal["retrieval", "generation", "rag"]
    dry_run: Literal[True]
    retrieval_combinations: int
    generation_combinations: int
    total_planned_runs: int
    max_combinations: int
    status: Literal["ok"]


class EvalRunListItem(BaseModel):
    run_id: str
    mode: EvalMode
    status: EvalRunStatus
    suite: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    total_planned_runs: int | None = None
    successful_runs: int | None = None
    failed_runs: int | None = None


class EvalRunListResponse(BaseModel):
    runs: list[EvalRunListItem]


class EvalRunStatusResponse(BaseModel):
    run_id: str
    mode: EvalMode
    status: EvalRunStatus
    suite: str | None = None
    triggered_by: Literal["api"]
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    total_planned_runs: int | None = None
    successful_runs: int | None = None
    failed_runs: int | None = None
    summary_url: str
    failures_url: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
