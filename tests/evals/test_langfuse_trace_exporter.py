from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.langfuse_trace_exporter import (
    DEFAULT_FALLBACK_SUBSTRINGS,
    LangfuseExportFilters,
    build_bad_trace_export_candidate,
    default_max_traces_to_scan,
    langfuse_trace_to_eval_example,
    parse_iso_date_or_datetime,
    write_eval_examples_jsonl,
)


def _build_filters(**overrides: object) -> LangfuseExportFilters:
    payload = {
        "limit": 5,
        "max_traces_to_scan": default_max_traces_to_scan(5),
        "score_name": "answer_quality",
        "max_score": 0.6,
    }
    payload.update(overrides)
    return LangfuseExportFilters(**payload)


def _build_trace(**overrides: object) -> SimpleNamespace:
    payload = {
        "id": "trace-123",
        "timestamp": datetime(2026, 7, 6, 12, 34, 56, tzinfo=UTC),
        "input": {"question": "What backend projects has Tumelo built?"},
        "output": {
            "final_answer": "A weak production answer.",
            "llm_provider": "openrouter",
            "llm_model": "openai/gpt-4o-mini",
            "latency_ms": 2450,
        },
        "session_id": "session-abc",
        "release": "prod-2026-07-06",
        "version": "v1",
        "environment": "production",
        "html_path": "/project/demo/traces/trace-123",
        "latency": 2.45,
        "total_cost": 0.0012,
        "observations": [
            SimpleNamespace(
                type="retriever",
                name="retrieval",
                output={
                    "results": [
                        {
                            "rank": 1,
                            "chunk_id": "chunk-1",
                            "source_name": "projects.md",
                            "score": 0.91,
                            "content_preview": "Preview",
                        }
                    ]
                },
                level="DEFAULT",
                status_message=None,
                model=None,
                cost_details={},
            )
        ],
        "scores": [
            SimpleNamespace(
                name="answer_quality",
                value=0.4,
                data_type="NUMERIC",
                string_value=None,
                comment="Bad answer",
            )
        ],
        "metadata": {"llm_provider": "openrouter"},
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_build_bad_trace_export_candidate_extracts_low_score_fields() -> None:
    trace = _build_trace()

    candidate = build_bad_trace_export_candidate(
        trace,
        filters=_build_filters(),
        base_url="https://cloud.langfuse.com",
    )

    assert candidate is not None
    assert candidate.failure_reason == "low_score"
    assert candidate.trace_id == "trace-123"
    assert candidate.question == "What backend projects has Tumelo built?"
    assert candidate.answer == "A weak production answer."
    assert candidate.model == "openai/gpt-4o-mini"
    assert candidate.provider == "openrouter"
    assert candidate.latency_ms == 2450
    assert candidate.estimated_cost_usd == 0.0012
    assert candidate.trace_url == "https://cloud.langfuse.com/project/demo/traces/trace-123"
    assert candidate.retrieved_context[0]["source_name"] == "projects.md"
    assert candidate.scores[0].value == 0.4


def test_build_bad_trace_export_candidate_detects_error_trace() -> None:
    trace = _build_trace(
        output={"error": "LLM timeout", "latency_ms": 900},
        scores=[],
        observations=[
            SimpleNamespace(
                type="generation",
                name="llm_call",
                output={},
                level="ERROR",
                status_message="LLM timeout",
                model="openai/gpt-4.1-mini",
                cost_details={},
            )
        ],
    )

    candidate = build_bad_trace_export_candidate(
        trace,
        filters=_build_filters(score_name=None, max_score=None, only_errors=True),
        base_url="https://cloud.langfuse.com",
    )

    assert candidate is not None
    assert candidate.failure_reason == "error_trace"
    assert candidate.answer is None


def test_langfuse_trace_to_eval_example_omits_sensitive_fields_by_default() -> None:
    candidate = build_bad_trace_export_candidate(
        _build_trace(),
        filters=_build_filters(),
        base_url="https://cloud.langfuse.com",
    )
    assert candidate is not None

    row = langfuse_trace_to_eval_example(candidate)

    assert row["id"] == "langfuse_trace-123"
    assert row["expected_facts"] == []
    assert row["expected_answer_points"] == []
    assert row["expected_source_documents"] == []
    assert "observed_answer" not in row
    assert "session_id" not in row["metadata"]
    assert row["metadata"]["failure_reason"] == "low_score"
    assert row["metadata"]["retrieved_sources"] == ["projects.md"]


def test_langfuse_trace_to_eval_example_can_include_answer_and_session_id() -> None:
    candidate = build_bad_trace_export_candidate(
        _build_trace(),
        filters=_build_filters(),
        base_url="https://cloud.langfuse.com",
    )
    assert candidate is not None

    row = langfuse_trace_to_eval_example(
        candidate,
        include_answer=True,
        include_session_id=True,
    )

    assert row["observed_answer"] == "A weak production answer."
    assert row["metadata"]["session_id"] == "session-abc"


def test_write_eval_examples_jsonl_appends_and_deduplicates(tmp_path: Path) -> None:
    output_path = tmp_path / "langfuse_review.jsonl"
    output_path.write_text(
        '{"id":"langfuse_trace-123","question":"Existing","expected_facts":[],"expected_answer_points":[],"expected_source_documents":[],"category":"production_failure","metadata":{"source":"langfuse"}}\n',
        encoding="utf-8",
    )

    summary = write_eval_examples_jsonl(
        output_path,
        [
            {
                "id": "langfuse_trace-123",
                "question": "Duplicate",
                "expected_facts": [],
                "expected_answer_points": [],
                "expected_source_documents": [],
                "category": "production_failure",
                "metadata": {"source": "langfuse"},
            },
            {
                "id": "langfuse_trace-456",
                "question": "New question",
                "expected_facts": [],
                "expected_answer_points": [],
                "expected_source_documents": [],
                "category": "production_failure",
                "metadata": {"source": "langfuse"},
            },
        ],
        append=True,
        overwrite=False,
    )

    assert summary.total_rows == 2
    assert summary.newly_added_rows == 1
    assert summary.duplicate_rows_skipped == 1
    assert "langfuse_trace-456" in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw_value", "inclusive_end", "expected"),
    [
        ("2026-07-01", False, datetime(2026, 7, 1, 0, 0, tzinfo=UTC)),
        ("2026-07-06", True, datetime(2026, 7, 7, 0, 0, tzinfo=UTC)),
        ("2026-07-06T12:34:56Z", False, datetime(2026, 7, 6, 12, 34, 56, tzinfo=UTC)),
    ],
)
def test_parse_iso_date_or_datetime(
    raw_value: str,
    inclusive_end: bool,
    expected: datetime,
) -> None:
    assert parse_iso_date_or_datetime(raw_value, inclusive_end=inclusive_end) == expected


def test_filters_validate_requires_a_bad_trace_selector() -> None:
    filters = LangfuseExportFilters(
        limit=5,
        max_traces_to_scan=10,
    )

    with pytest.raises(ValueError, match="At least one bad-trace filter is required"):
        filters.validate()


def test_filters_validate_accepts_fallback_detection() -> None:
    filters = LangfuseExportFilters(
        limit=5,
        max_traces_to_scan=10,
        fallback_substrings=DEFAULT_FALLBACK_SUBSTRINGS,
    )

    filters.validate()
