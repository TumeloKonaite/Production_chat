from __future__ import annotations

import json
from pathlib import Path

from evals.feedback.feedback_dataset import (
    FEEDBACK_DATASET_SOURCE,
    FeedbackEvalExample,
    load_feedback_dataset,
    summarize_feedback_dataset,
)


def test_feedback_eval_example_validates_expected_fields() -> None:
    example = FeedbackEvalExample.model_validate(
        {
            "id": "feedback_trace-123",
            "question": "What does Tumelo do?",
            "actual_answer": "Tumelo is mainly a frontend developer.",
            "expected_answer": "Tumelo builds practical AI systems and backend APIs.",
            "expected_facts": ["practical AI systems", "backend APIs"],
            "expected_answer_points": ["practical AI systems", "backend APIs"],
            "expected_source_documents": ["profile.md"],
            "feedback_rating": "negative",
            "feedback_reason": "incorrect_answer",
            "feedback_comment": "The answer missed the backend work.",
            "source": FEEDBACK_DATASET_SOURCE,
            "trace_id": "trace-123",
            "session_id": "session-123",
            "message_id": "message-123",
            "langfuse_trace_id": "lf-trace-123",
            "model_provider": "openai",
            "model_name": "gpt-4.1-mini",
            "retriever_type": "vector",
            "top_k": 5,
            "prompt_version": "v1_professional",
            "created_at": "2026-07-06T20:00:00Z",
            "metadata": {"environment": "production"},
            "context": [
                {
                    "source": "profile.md",
                    "section": "Summary",
                    "content": "Tumelo builds practical AI systems and backend APIs.",
                    "similarity": 0.9,
                }
            ],
        }
    )

    assert example.feedback_rating == "negative"
    assert example.has_generation_labels is True
    assert example.has_retrieval_labels is True
    assert example.created_at.isoformat() == "2026-07-06T20:00:00+00:00"


def test_summarize_feedback_dataset_ignores_non_feedback_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "feedback_dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "feedback_trace-123",
                        "question": "What does Tumelo do?",
                        "actual_answer": "Tumelo is mainly a frontend developer.",
                        "expected_facts": [],
                        "expected_answer_points": [],
                        "expected_source_documents": [],
                        "feedback_rating": "negative",
                        "feedback_reason": "incorrect_answer",
                        "source": FEEDBACK_DATASET_SOURCE,
                        "trace_id": "trace-123",
                        "created_at": "2026-07-06T20:00:00Z",
                        "metadata": {"environment": "production"},
                        "context": [],
                    }
                ),
                json.dumps(
                    {
                        "id": "generation_profile_001",
                        "question": "What does Tumelo do?",
                        "category": "profile",
                        "context": [{"content": "ignored", "source": "profile.md", "section": "Summary"}],
                        "expected_facts": ["ignored"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded_examples = load_feedback_dataset(dataset_path)
    summary = summarize_feedback_dataset(dataset_path)

    assert [example.id for example in loaded_examples] == ["feedback_trace-123"]
    assert summary is not None
    assert summary.case_count == 1
    assert summary.negative_count == 1
    assert summary.feedback_date_from == "2026-07-06T20:00:00Z"
    assert summary.production_trace_ids == ["trace-123"]
