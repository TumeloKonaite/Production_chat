from __future__ import annotations

import json
from pathlib import Path

from app.services.retrieval import RetrievedChunk
from evals.run_retrieval_eval import (
    RetrievalEvalExample,
    build_run_config,
    create_output_directory,
    evaluate_examples,
    evaluate_examples_for_k_values,
    write_artifacts,
)


class FakeRetrievalService:
    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return list(self._responses.get(query, []))


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


def test_write_artifacts_persists_json_csv_and_config_outputs(tmp_path: Path) -> None:
    output_dir = create_output_directory(tmp_path, timestamp_label="2026-07-02_203000")
    summary = {
        "num_queries_total": 1,
        "num_queries_evaluated": 1,
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
    assert config["vector_store_type"] == "pgvector"
    assert config["retrieval_strategy"] == "similarity_search_with_relevance_scores"
    assert config["chunk_size"] == 500
    assert config["chunk_overlap"] == 100
    assert config["settings_used_by_retriever"]["retrieval_min_similarity"] == 0.55
    assert "run_retrieval_eval.py" in config["python_command_used"]


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
    assert summary["metrics_by_k"]["1"]["recall_at_k"] == 0.5
    assert summary["metrics_by_k"]["3"]["recall_at_k"] == 1.0
    assert summary["mrr"] == 0.75
    assert results[0]["metrics_by_k"]["1"]["hit_at_k"] == 1.0
    assert results[1]["metrics_by_k"]["1"]["hit_at_k"] == 0.0
    assert results[1]["metrics_by_k"]["3"]["recall_at_k"] == 1.0
