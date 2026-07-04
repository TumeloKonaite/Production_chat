from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.domain.evals import JudgeEvaluation, JudgeMetricScore
from app.infrastructure.llm.base import TokenUsage
from app.services.retrieval import RetrievedChunk
import pytest
from evals.query_rewriter import (
    QUERY_REWRITE_STATUS_EMPTY_FALLBACK,
    QUERY_REWRITE_STATUS_ERROR_FALLBACK,
    QUERY_REWRITE_STATUS_SUCCESS,
    QueryRewriteResult,
)
from evals.run_retrieval_eval import (
    RetrievalEvalExample,
    RetrievalEvalDatasetValidationError,
    build_run_config,
    build_tracking_run_name,
    create_output_directory,
    evaluate_examples,
    evaluate_examples_for_k_values,
    format_dataset_validation_summary,
    load_and_validate_dataset,
    load_eval_config,
    log_run_to_tracker,
    parse_args,
    run_retrieval_eval,
    validate_dataset_examples,
    write_artifacts,
)


class FakeRetrievalService:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return list(self._responses.get(query, []))


class FakeTrackerRun:
    def __init__(self, run_id: str = "mlflow-run-123") -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self) -> FakeTrackerRun:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class FakeTracker:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_names: list[str] = []
        self.params: list[dict[str, object]] = []
        self.metrics: list[dict[str, float | int]] = []
        self.artifacts: list[Path] = []

    def run(self, run_name: str) -> FakeTrackerRun:
        self.run_names.append(run_name)
        return FakeTrackerRun()

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        self.metrics.append(metrics)

    def log_artifact(self, artifact_path: Path) -> None:
        self.artifacts.append(artifact_path)


class FakeQueryRewriter:
    def __init__(self, results: dict[str, QueryRewriteResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, str | None]] = []

    def rewrite_query(
        self,
        original_query: str,
        *,
        context: str | None = None,
    ) -> QueryRewriteResult:
        self.calls.append((original_query, context))
        return self._results[original_query]


class FakeJudge:
    def __init__(self, score: int = 2) -> None:
        self.score = score
        self.prompts: list[str] = []

    async def evaluate(
        self,
        *,
        prompt: str,
        model_config_id: str | None = None,
    ):
        del model_config_id
        self.prompts.append(prompt)
        return (
            JudgeEvaluation(
                context_relevance=JudgeMetricScore(score=self.score, reason="Useful."),
                faithfulness=JudgeMetricScore(score=2, reason="Ignored."),
                answer_relevance=JudgeMetricScore(score=0, reason="Ignored."),
            ),
            TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            50,
            "gpt-4.1-mini",
        )


def build_chunk(*, chunk_id: str, source: str, similarity: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=source,
        section="Section",
        content=f"content from {source}",
        similarity=similarity,
        metadata={"chunk_id": chunk_id, "source": source},
    )


def build_query_rewrite_result(
    *,
    original_query: str,
    query_used_for_retrieval: str,
    status: str,
    rewritten_query: str | None = None,
    rewrite_context: str | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost: float | None = None,
) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_query=original_query,
        rewrite_context=rewrite_context,
        rewritten_query=rewritten_query,
        query_used_for_retrieval=query_used_for_retrieval,
        query_rewriting_enabled=True,
        query_rewrite_status=status,
        query_rewrite_model="gpt-4.1-mini",
        query_rewrite_prompt_version="v1",
        query_rewrite_latency_ms=latency_ms,
        query_rewrite_prompt_tokens=prompt_tokens,
        query_rewrite_completion_tokens=completion_tokens,
        query_rewrite_total_tokens=total_tokens,
        query_rewrite_estimated_cost=estimated_cost,
        query_rewrite_error=error,
    )


def test_evaluate_examples_excludes_rows_without_expected_sources_from_aggregates() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="question 1",
            expected_source_documents=["projects.md"],
        ),
        RetrievalEvalExample(
            id="q2",
            question="question 2",
            expected_source_documents=[],
        ),
    ]
    retrieval_service = FakeRetrievalService(
        {
            "question 1": [
                build_chunk(chunk_id="projects.md::chunk-1", source="projects.md"),
                build_chunk(chunk_id="skills.md::chunk-1", source="skills.md"),
            ],
            "question 2": [
                build_chunk(chunk_id="contact.md::chunk-1", source="contact.md"),
            ],
        }
    )

    summary, results = evaluate_examples(examples, retrieval_service, k=5)

    assert retrieval_service.calls == [("question 1", 5), ("question 2", 5)]
    assert summary == {
        "num_queries_total": 2,
        "num_queries_evaluated": 1,
        "num_queries_without_expected_source": 1,
        "num_queries_without_expected_sources": 1,
        "k": 5,
        "hit_at_k": 1.0,
        "recall_at_k": 1.0,
        "mean_precision_at_k": 0.5,
        "mrr": 1.0,
        "query_rewrite_total_latency_ms": 0,
        "query_rewrite_avg_latency_ms": 0.0,
        "query_rewrite_success_count": 0,
        "query_rewrite_fallback_count": 0,
        "query_rewrite_failure_count": 0,
        "query_rewrite_total_prompt_tokens": 0,
        "query_rewrite_total_completion_tokens": 0,
        "query_rewrite_total_tokens": 0,
        "query_rewrite_estimated_total_cost": 0.0,
        "context_relevance": None,
    }
    assert results[0]["original_query"] == "question 1"
    assert results[0]["rewritten_query"] is None
    assert results[0]["query_rewrite_status"] == "disabled"
    assert results[0]["query_used_for_retrieval"] == "question 1"
    assert results[0]["reranker_enabled"] is False
    assert results[0]["reranker_type"] == "none"
    assert results[0]["retriever_top_k"] == 5
    assert results[0]["final_top_k"] == 5
    assert results[0]["retrieved_sources"] == ["projects.md", "skills.md"]
    assert results[0]["retrieved_chunk_ids"] == ["projects.md::chunk-1", "skills.md::chunk-1"]
    assert results[0]["before_rerank"][0]["chunk_id"] == "projects.md::chunk-1"
    assert results[0]["after_rerank"][0]["chunk_id"] == "projects.md::chunk-1"
    assert results[0]["first_relevant_rank"] == 1
    assert results[0]["evaluation_group"] == "retrieval_evaluated"
    assert results[1]["has_expected_sources"] is False
    assert results[1]["evaluation_group"] == "no_expected_source"
    assert results[1]["hit_at_k"] is None
    assert results[1]["mrr"] is None


def test_validate_dataset_examples_reports_expected_source_coverage() -> None:
    summary = validate_dataset_examples(
        [
            RetrievalEvalExample(
                id="q1",
                question="question 1",
                expected_source_documents=["projects.md"],
            ),
            RetrievalEvalExample(
                id="q2",
                question="question 2",
                expected_source_documents=["skills.md"],
            ),
            RetrievalEvalExample(
                id="q3",
                question="question 3",
                expected_source_documents=[],
            ),
        ],
        min_expected_source_coverage=0.60,
    )

    assert summary.total_queries == 3
    assert summary.queries_with_expected_sources == 2
    assert summary.queries_without_expected_sources == 1
    assert summary.missing_expected_source_ids == ["q3"]
    assert summary.expected_source_coverage == pytest.approx(2 / 3)
    assert format_dataset_validation_summary(summary) == "\n".join(
        [
            "Retrieval eval dataset validation",
            "---------------------------------",
            "total_queries: 3",
            "queries_with_expected_sources: 2",
            "queries_without_expected_sources: 1",
            "queries_missing_expected_sources: q3",
        ]
    )


def test_validate_dataset_examples_fails_when_coverage_is_too_low() -> None:
    with pytest.raises(RetrievalEvalDatasetValidationError) as exc_info:
        validate_dataset_examples(
            [
                RetrievalEvalExample(
                    id="q1",
                    question="question 1",
                    expected_source_documents=["projects.md"],
                ),
                RetrievalEvalExample(
                    id="q2",
                    question="question 2",
                    expected_source_documents=[],
                ),
                RetrievalEvalExample(
                    id="q3",
                    question="question 3",
                    expected_source_documents=[],
                ),
            ],
            min_expected_source_coverage=0.75,
        )

    assert str(exc_info.value) == (
        "Retrieval eval dataset is not valid: 2 of 3 queries are missing "
        "expected_source_documents."
    )


def test_load_and_validate_dataset_reads_examples_from_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "question 1",
                        "expected_source_documents": ["projects.md"],
                    }
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "question 2",
                        "expected_source_documents": ["skills.md"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples, validation_summary = load_and_validate_dataset(
        dataset_path,
        min_expected_source_coverage=1.0,
    )

    assert [example.id for example in examples] == ["q1", "q2"]
    assert validation_summary.queries_without_expected_sources == 0


def test_load_eval_config_reads_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "retrieval_eval.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": "dataset.jsonl",
                "retriever_top_k": 20,
                "final_top_k": 5,
                "reranker_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    config = load_eval_config(config_path)

    assert config["dataset"] == "dataset.jsonl"
    assert config["retriever_top_k"] == 20
    assert config["final_top_k"] == 5
    assert config["reranker_enabled"] is True


def test_write_artifacts_persists_json_csv_and_config_outputs(tmp_path: Path) -> None:
    output_dir = create_output_directory(tmp_path, timestamp_label="2026-07-02_203000")
    summary = {
        "num_queries_total": 1,
        "num_queries_evaluated": 1,
        "num_queries_without_expected_source": 0,
        "num_queries_without_expected_sources": 0,
        "k": 5,
        "hit_at_k": 1.0,
        "recall_at_k": 1.0,
        "mean_precision_at_k": 1.0,
        "mrr": 1.0,
        "query_rewrite_total_latency_ms": 0,
        "query_rewrite_avg_latency_ms": 0.0,
        "query_rewrite_success_count": 0,
        "query_rewrite_fallback_count": 0,
        "query_rewrite_failure_count": 0,
        "query_rewrite_total_prompt_tokens": 0,
        "query_rewrite_total_completion_tokens": 0,
        "query_rewrite_total_tokens": 0,
        "query_rewrite_estimated_total_cost": 0.0,
        "context_relevance": None,
    }
    results = [
        {
            "id": "q1",
            "question": "question 1",
            "expected_source_documents": ["projects.md"],
            "retrieved_sources": ["projects.md"],
            "retrieved_chunk_ids": ["projects.md::chunk-1"],
            "has_expected_sources": True,
            "evaluation_group": "retrieval_evaluated",
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "precision_at_k": 1.0,
            "mrr": 1.0,
            "first_relevant_rank": 1,
        }
    ]
    config = {
        "timestamp": "2026-07-02T20:30:00+02:00",
        "top_k": 5,
        "query_rewriting_enabled": False,
        "query_rewrite_prompt_version": "v1",
    }

    artifact_paths = write_artifacts(output_dir, summary=summary, results=results, config=config)

    payload = json.loads(artifact_paths["results_json"].read_text(encoding="utf-8"))
    assert payload["summary"] == summary
    assert payload["chunking"] == {"chunk_size": None, "chunk_overlap": None}
    assert payload["results"] == results
    csv_text = artifact_paths["results_csv"].read_text(encoding="utf-8")
    assert "retrieved_chunk_ids" in csv_text
    assert "projects.md::chunk-1" in csv_text
    assert json.loads(artifact_paths["config_json"].read_text(encoding="utf-8")) == config
    assert "query_rewrite_prompt_txt" not in artifact_paths


def test_build_run_config_captures_retrieval_settings() -> None:
    settings = type(
        "Settings",
        (),
        {
            "retriever_type": "vector",
            "embedding_provider": "hf",
            "knowledge_embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "default_retrieval_config": "default",
            "retrieval_top_k": 5,
            "retrieval_min_similarity": 0.55,
            "knowledge_collection_name": "personal_knowledge_base",
            "knowledge_chunk_size": 500,
            "knowledge_chunk_overlap": 100,
            "database_url": "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
            "enable_query_rewriting": True,
            "query_rewrite_model": "openai:gpt-4.1-mini",
            "query_rewrite_temperature": 0.0,
            "query_rewrite_prompt_version": "v1",
            "query_rewrite_timeout_seconds": 10,
            "query_rewrite_max_tokens": 128,
            "reranker_enabled": False,
            "reranker_type": "none",
            "reranker_model": None,
            "reranker_initial_top_k": 5,
            "reranker_final_top_k": 5,
        },
    )()

    config = build_run_config(
        settings=settings,
        dataset_path=Path("evals/datasets/portfolio_eval_dataset.jsonl"),
        top_k=5,
        timestamp="2026-07-02T20:30:00+02:00",
        argv=["evals/run_retrieval_eval.py", "--k", "5"],
        run_name="retrieval-vector-k5-2026-07-02_203000",
        notes="Triggered from API",
    )

    assert config["run_name"] == "retrieval-vector-k5-2026-07-02_203000"
    assert config["notes"] == "Triggered from API"
    assert config["embedding_provider"] == "hf"
    assert config["embedding_model"] == "all-MiniLM-L6-v2"
    assert config["embedding_dimension"] == 384
    assert config["retriever_type"] == "vector"
    assert config["vector_store_type"] == "pgvector"
    assert config["retrieval_strategy"] == "vector"
    assert config["query_rewriting_enabled"] is True
    assert config["query_rewrite_model"] == "openai:gpt-4.1-mini"
    assert config["chunk_size"] == 500
    assert config["chunk_overlap"] == 100
    assert config["settings_used_by_retriever"]["retriever_type"] == "vector"
    assert config["settings_used_by_retriever"]["retrieval_min_similarity"] == 0.55
    assert "run_retrieval_eval.py" in config["python_command_used"]


def test_log_run_to_tracker_logs_summary_and_artifacts(tmp_path: Path) -> None:
    tracker = FakeTracker()
    settings = type(
        "Settings",
        (),
        {
            "retriever_type": "hybrid",
            "default_retrieval_config": "default",
            "embedding_provider": "hf",
            "knowledge_embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "knowledge_collection_name": "personal_knowledge_base",
            "retrieval_min_similarity": 0.55,
        },
    )()
    artifact_paths = {
        "results_json": tmp_path / "results.json",
        "results_csv": tmp_path / "results.csv",
        "config_json": tmp_path / "config.json",
    }
    for path in artifact_paths.values():
        path.write_text("{}", encoding="utf-8")

    run_id = log_run_to_tracker(
        tracker=tracker,
        settings=settings,
        dataset_path=Path("evals/datasets/portfolio_eval_dataset.jsonl"),
        top_k=5,
        summary={
            "num_queries_total": 2,
            "num_queries_evaluated": 1,
            "num_queries_without_expected_source": 1,
            "num_queries_without_expected_sources": 1,
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "mean_precision_at_k": 0.5,
            "mrr": 1.0,
        },
        config={
            "notes": "Triggered from API",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "git_commit_sha": "abc123",
            "query_rewriting_enabled": True,
            "query_rewrite_model": "openai:gpt-4.1-mini",
            "query_rewrite_temperature": 0.0,
            "query_rewrite_prompt_version": "v1",
            "query_rewrite_timeout_seconds": 10,
            "query_rewrite_max_tokens": 128,
            "reranker_enabled": False,
            "reranker_type": "none",
            "reranker_model": None,
            "reranker_initial_top_k": 5,
            "reranker_final_top_k": 5,
        },
        artifact_paths=artifact_paths,
        run_name=build_tracking_run_name(
            retriever_type="hybrid",
            top_k=5,
            timestamp_label="2026-07-03_145252",
        ),
    )

    assert run_id == "mlflow-run-123"
    assert tracker.run_names == ["retrieval-hybrid-k5-2026-07-03_145252"]
    assert tracker.params == [
        {
            "run_name": "retrieval-hybrid-k5-2026-07-03_145252",
            "notes": "Triggered from API",
            "dataset_name": "portfolio_eval_dataset.jsonl",
            "dataset_path": "evals\\datasets\\portfolio_eval_dataset.jsonl",
            "retriever_type": "hybrid",
            "retrieval_config": "default",
            "top_k": 5,
            "embedding_provider": "hf",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "knowledge_collection_name": "personal_knowledge_base",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "retrieval_min_similarity": 0.55,
            "git_commit_sha": "abc123",
            "query_rewriting_enabled": True,
            "query_rewrite_model": "openai:gpt-4.1-mini",
            "query_rewrite_temperature": 0.0,
            "query_rewrite_prompt_version": "v1",
            "query_rewrite_timeout_seconds": 10,
            "query_rewrite_max_tokens": 128,
            "reranker_enabled": False,
            "reranker_type": "none",
            "reranker_model": None,
            "reranker_initial_top_k": 5,
            "reranker_final_top_k": 5,
        }
    ]
    assert tracker.metrics == [
        {
            "num_queries_total": 2,
            "num_queries_evaluated": 1,
            "num_queries_without_expected_source": 1,
            "num_queries_without_expected_sources": 1,
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "mean_precision_at_k": 0.5,
            "mrr": 1.0,
            "query_rewrite_total_latency_ms": 0,
            "query_rewrite_avg_latency_ms": 0.0,
            "query_rewrite_success_count": 0,
            "query_rewrite_fallback_count": 0,
            "query_rewrite_failure_count": 0,
            "query_rewrite_total_prompt_tokens": 0,
            "query_rewrite_total_completion_tokens": 0,
            "query_rewrite_total_tokens": 0,
            "query_rewrite_estimated_total_cost": 0.0,
            "context_relevance": 0.0,
        }
    ]
    assert tracker.artifacts == list(artifact_paths.values())


def test_evaluate_examples_for_k_values_reports_multi_k_summary() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="question 1",
            expected_source_documents=["projects.md"],
        ),
        RetrievalEvalExample(
            id="q2",
            question="question 2",
            expected_source_documents=["skills.md"],
        ),
    ]
    retrieval_service = FakeRetrievalService(
        {
            "question 1": [
                build_chunk(chunk_id="projects.md::chunk-1", source="projects.md"),
                build_chunk(chunk_id="contact.md::chunk-1", source="contact.md"),
            ],
            "question 2": [
                build_chunk(chunk_id="contact.md::chunk-2", source="contact.md"),
                build_chunk(chunk_id="skills.md::chunk-1", source="skills.md"),
            ],
        }
    )

    summary, results = evaluate_examples_for_k_values(
        examples,
        retrieval_service,
        k_values=[5, 1, 3],
    )

    assert retrieval_service.calls == [("question 1", 5), ("question 2", 5)]
    assert summary["k"] == 5
    assert summary["k_values"] == [1, 3, 5]
    assert summary["num_queries_evaluated"] == 2
    assert summary["num_queries_without_expected_source"] == 0
    assert summary["metrics_by_k"]["1"]["recall_at_k"] == 0.5
    assert summary["metrics_by_k"]["3"]["recall_at_k"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["query_rewrite_total_latency_ms"] == 0
    assert results[0]["metrics_by_k"]["1"]["hit_at_k"] == 1.0
    assert results[1]["metrics_by_k"]["1"]["hit_at_k"] == 0.0
    assert results[1]["metrics_by_k"]["3"]["recall_at_k"] == 1.0


def test_evaluate_examples_records_context_relevance_when_judge_succeeds() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="question 1",
            expected_source_documents=["projects.md"],
        )
    ]
    retrieval_service = FakeRetrievalService(
        {
            "question 1": [
                build_chunk(chunk_id="projects.md::chunk-1", source="projects.md"),
            ],
        }
    )
    judge = FakeJudge(score=2)

    summary, results = evaluate_examples(
        examples,
        retrieval_service,
        k=5,
        context_relevance_judge=judge,
    )

    assert summary["context_relevance"] == 2.0
    assert results[0]["context_relevance"] == 2.0
    assert "Return exactly one JSON object" in judge.prompts[0]


def test_evaluate_examples_uses_rewritten_query_when_query_rewriting_succeeds() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="What does he do?",
            expected_source_documents=["profile.md"],
            rewrite_context="Subject: Tumelo Konaite",
        )
    ]
    retrieval_service = FakeRetrievalService(
        {
            "What does Tumelo Konaite do?": [
                build_chunk(chunk_id="profile.md::chunk-1", source="profile.md"),
            ]
        }
    )
    query_rewriter = FakeQueryRewriter(
        {
            "What does he do?": build_query_rewrite_result(
                original_query="What does he do?",
                rewrite_context="Subject: Tumelo Konaite",
                rewritten_query="What does Tumelo Konaite do?",
                query_used_for_retrieval="What does Tumelo Konaite do?",
                status=QUERY_REWRITE_STATUS_SUCCESS,
                latency_ms=412,
                prompt_tokens=80,
                completion_tokens=9,
                total_tokens=89,
                estimated_cost=0.000012,
            )
        }
    )

    summary, results = evaluate_examples(
        examples,
        retrieval_service,
        k=5,
        query_rewriter=query_rewriter,
    )

    assert query_rewriter.calls == [("What does he do?", "Subject: Tumelo Konaite")]
    assert retrieval_service.calls == [("What does Tumelo Konaite do?", 5)]
    assert results[0]["original_query"] == "What does he do?"
    assert results[0]["rewritten_query"] == "What does Tumelo Konaite do?"
    assert results[0]["query_used_for_retrieval"] == "What does Tumelo Konaite do?"
    assert results[0]["query_rewrite_status"] == "success"
    assert summary["query_rewrite_success_count"] == 1
    assert summary["query_rewrite_total_tokens"] == 89


def test_evaluate_examples_falls_back_to_original_query_when_rewrite_is_empty() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="What does Tumelo do?",
            expected_source_documents=["profile.md"],
        )
    ]
    retrieval_service = FakeRetrievalService(
        {
            "What does Tumelo do?": [
                build_chunk(chunk_id="profile.md::chunk-1", source="profile.md"),
            ]
        }
    )
    query_rewriter = FakeQueryRewriter(
        {
            "What does Tumelo do?": build_query_rewrite_result(
                original_query="What does Tumelo do?",
                query_used_for_retrieval="What does Tumelo do?",
                status=QUERY_REWRITE_STATUS_EMPTY_FALLBACK,
                error="Query rewrite returned an empty response.",
            )
        }
    )

    summary, results = evaluate_examples(
        examples,
        retrieval_service,
        k=5,
        query_rewriter=query_rewriter,
    )

    assert retrieval_service.calls == [("What does Tumelo do?", 5)]
    assert results[0]["rewritten_query"] is None
    assert results[0]["query_rewrite_status"] == "empty_fallback"
    assert results[0]["query_rewrite_error"] == "Query rewrite returned an empty response."
    assert summary["query_rewrite_fallback_count"] == 1
    assert summary["query_rewrite_failure_count"] == 0


def test_evaluate_examples_falls_back_to_original_query_when_rewrite_errors() -> None:
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="What does Tumelo do?",
            expected_source_documents=["profile.md"],
        )
    ]
    retrieval_service = FakeRetrievalService(
        {
            "What does Tumelo do?": [
                build_chunk(chunk_id="profile.md::chunk-1", source="profile.md"),
            ]
        }
    )
    query_rewriter = FakeQueryRewriter(
        {
            "What does Tumelo do?": build_query_rewrite_result(
                original_query="What does Tumelo do?",
                query_used_for_retrieval="What does Tumelo do?",
                status=QUERY_REWRITE_STATUS_ERROR_FALLBACK,
                error="LLM request failed",
            )
        }
    )

    summary, results = evaluate_examples(
        examples,
        retrieval_service,
        k=5,
        query_rewriter=query_rewriter,
    )

    assert retrieval_service.calls == [("What does Tumelo do?", 5)]
    assert results[0]["query_rewrite_status"] == "error_fallback"
    assert results[0]["query_rewrite_error"] == "LLM request failed"
    assert summary["query_rewrite_fallback_count"] == 1
    assert summary["query_rewrite_failure_count"] == 1


def test_run_retrieval_eval_with_query_rewriting_disabled_uses_original_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_service = FakeRetrievalService(
        {
            "question 1": [build_chunk(chunk_id="projects.md::chunk-1", source="projects.md")],
        }
    )

    monkeypatch.setattr(
        "evals.retrieval_eval_runner.RetrievalService",
        lambda settings: retrieval_service,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QueryRewriter should not be constructed when rewriting is disabled.")

    monkeypatch.setattr("evals.retrieval_eval_runner.QueryRewriter", fail_if_called)

    settings = type(
        "Settings",
        (),
        {
            "retriever_type": "vector",
            "embedding_provider": "hf",
            "knowledge_embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "default_retrieval_config": "default",
            "retrieval_top_k": 5,
            "retrieval_min_similarity": 0.55,
            "knowledge_collection_name": "personal_knowledge_base",
            "knowledge_chunk_size": 500,
            "knowledge_chunk_overlap": 100,
            "database_url": "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/test",
            "enable_query_rewriting": False,
            "query_rewrite_model": "openai:gpt-4.1-mini",
            "query_rewrite_temperature": 0.0,
            "query_rewrite_prompt_version": "v1",
            "query_rewrite_timeout_seconds": 10,
            "query_rewrite_max_tokens": 128,
        },
    )()
    examples = [
        RetrievalEvalExample(
            id="q1",
            question="question 1",
            expected_source_documents=["projects.md"],
        )
    ]
    validation_summary = validate_dataset_examples(examples, min_expected_source_coverage=1.0)

    result = run_retrieval_eval(
        settings=settings,
        dataset_path=tmp_path / "dataset.jsonl",
        output_root=tmp_path / "output",
        top_k=5,
        tracker=FakeTracker(enabled=False),
        argv=["evals/run_retrieval_eval.py"],
        examples=examples,
        validation_summary=validation_summary,
        timestamp="2026-07-04T12:00:00+02:00",
        timestamp_label="2026-07-04_120000",
    )

    assert retrieval_service.calls == [("question 1", 5)]
    assert result.results[0]["query_rewrite_status"] == "disabled"
    assert result.results[0]["rewritten_query"] is None
    assert "query_rewrite_prompt_txt" not in result.artifact_paths


def test_parse_args_accepts_query_rewriting_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evals/run_retrieval_eval.py", "--enable-query-rewriting"],
    )

    args = parse_args()

    assert args.enable_query_rewriting is True
    assert args.disable_query_rewriting is False


def test_parse_args_accepts_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evals/run_retrieval_eval.py", "--config", "configs/evals/retrieval_baseline.json"],
    )

    args = parse_args()

    assert args.config == Path("configs/evals/retrieval_baseline.json")


def test_parse_args_accepts_reranker_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evals/run_retrieval_eval.py",
            "--enable-reranker",
            "--reranker-type",
            "llm",
            "--reranker-model",
            "openai:gpt-4.1-mini",
            "--reranker-initial-top-k",
            "20",
        ],
    )

    args = parse_args()

    assert args.enable_reranker is True
    assert args.disable_reranker is False
    assert args.reranker_type == "llm"
    assert args.reranker_model == "openai:gpt-4.1-mini"
    assert args.reranker_initial_top_k == 20
