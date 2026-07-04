from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalEvalRunRequest(BaseModel):
    retriever_type: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=5, gt=0, le=50)
    enable_query_rewriting: bool | None = None
    run_name: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, min_length=1, max_length=2000)


class RetrievalEvalRunResponse(BaseModel):
    status: Literal["completed"]
    run_name: str
    mlflow_run_id: str | None = None
    config: dict[str, Any]
    metrics: dict[str, Any]


class RetrievalSweepExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    retriever_type: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=5, gt=0, le=50)
    embedding_provider: str | None = Field(default=None, min_length=1, max_length=100)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=300)
    embedding_dimension: int | None = Field(default=None, gt=0)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    reranker_enabled: bool | None = None
    reranker_type: Literal["none", "llm"] | None = None
    reranker_model: str | None = Field(default=None, min_length=1, max_length=300)
    reranker_initial_top_k: int | None = Field(default=None, gt=0, le=200)


class RetrievalEvalSweepRequest(BaseModel):
    experiments: list[RetrievalSweepExperimentRequest] = Field(min_length=1)
    enable_query_rewriting: bool | None = None


class RetrievalEvalSweepRunResponse(BaseModel):
    experiment_name: str
    run_name: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str]


class RetrievalEvalSweepResponse(BaseModel):
    status: Literal["completed"]
    runs: list[RetrievalEvalSweepRunResponse]
    artifacts: dict[str, str]
