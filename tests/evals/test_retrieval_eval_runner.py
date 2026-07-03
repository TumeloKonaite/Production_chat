from __future__ import annotations

import json
from pathlib import Path

from app.services.retrieval import RetrievedChunk
import pytest
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
    log_run_to_tracker,
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
    def __enter__(self) -> None:
        return None

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


def build_chunk(*, chunk_id: str, source: str, similarity: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=source,
        section="Section",
        content=f"content from {source}",
        similarity=similarity,
        metadata={"chunk_id": chunk_id, "source": source},
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
    }
    assert results[0]["retrieved_sources"] == ["projects.md", "skills.md"]
    assert results[0]["retrieved_chunk_ids"] == ["projects.md::chunk-1", "skills.md::chunk-1"]
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
    config = {"timestamp": "2026-07-02T20:30:00+02:00", "top_k": 5}

    artifact_paths = write_artifacts(output_dir, summary=summary, results=results, config=config)

    payload = json.loads(artifact_paths["results_json"].read_text(encoding="utf-8"))
    assert payload["summary"] == summary
    assert payload["chunking"] == {"chunk_size": None, "chunk_overlap": None}
    assert payload["results"] == results
    csv_text = artifact_paths["results_csv"].read_text(encoding="utf-8")
    assert "retrieved_chunk_ids" in csv_text
    assert "projects.md::chunk-1" in csv_text
    assert json.loads(artifact_paths["config_json"].read_text(encoding="utf-8")) == config


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
        },
    )()

    config = build_run_config(
        settings=settings,
        dataset_path=Path("evals/datasets/portfolio_eval_dataset.jsonl"),
        top_k=5,
        timestamp="2026-07-02T20:30:00+02:00",
        argv=["evals/run_retrieval_eval.py", "--k", "5"],
    )

    assert config["embedding_provider"] == "hf"
    assert config["embedding_model"] == "all-MiniLM-L6-v2"
    assert config["embedding_dimension"] == 384
    assert config["retriever_type"] == "vector"
    assert config["vector_store_type"] == "pgvector"
    assert config["retrieval_strategy"] == "vector"
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

    log_run_to_tracker(
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
            "chunk_size": 500,
            "chunk_overlap": 100,
            "git_commit_sha": "abc123",
        },
        artifact_paths=artifact_paths,
        run_name=build_tracking_run_name(
            retriever_type="hybrid",
            top_k=5,
            timestamp_label="2026-07-03_145252",
        ),
    )

    assert tracker.run_names == ["retrieval-hybrid-k5-2026-07-03_145252"]
    assert tracker.params == [
        {
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
    assert results[0]["metrics_by_k"]["1"]["hit_at_k"] == 1.0
    assert results[1]["metrics_by_k"]["1"]["hit_at_k"] == 0.0
    assert results[1]["metrics_by_k"]["3"]["recall_at_k"] == 1.0
