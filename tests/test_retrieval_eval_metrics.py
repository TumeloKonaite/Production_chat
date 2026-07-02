from __future__ import annotations

import pytest

from evals.metrics.retrieval_metrics import (
    first_relevant_rank,
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    unique_ranked_sources,
)


def test_unique_ranked_sources_deduplicates_in_rank_order() -> None:
    assert unique_ranked_sources(
        [" projects.md ", "projects.md", "skills.md", "PROFILE.md", "profile.md"]
    ) == ["projects.md", "skills.md", "PROFILE.md"]


def test_hit_at_k_returns_binary_hit_for_any_relevant_source() -> None:
    retrieved_sources = ["contact.md", "skills.md", "projects.md"]
    expected_sources = ["projects.md", "profile.md"]

    assert hit_at_k(retrieved_sources, expected_sources, 2) == pytest.approx(0.0)
    assert hit_at_k(retrieved_sources, expected_sources, 3) == pytest.approx(1.0)


def test_recall_and_precision_at_k_use_deduped_sources_and_actual_retrieved_count() -> None:
    retrieved_sources = ["projects.md", "projects.md", "skills.md"]
    expected_sources = ["projects.md", "profile.md"]

    assert recall_at_k(retrieved_sources, expected_sources, 5) == pytest.approx(0.5)
    assert precision_at_k(retrieved_sources, expected_sources, 5) == pytest.approx(0.5)


def test_mrr_and_first_relevant_rank_use_first_unique_relevant_source() -> None:
    retrieved_sources = ["contact.md", "skills.md", "projects.md", "projects.md"]
    expected_sources = ["projects.md"]

    assert first_relevant_rank(retrieved_sources, expected_sources) == 3
    assert mrr(retrieved_sources, expected_sources) == pytest.approx(1 / 3)


def test_metrics_return_zero_when_expected_sources_are_empty() -> None:
    retrieved_sources = ["projects.md"]

    assert hit_at_k(retrieved_sources, [], 5) == pytest.approx(0.0)
    assert recall_at_k(retrieved_sources, [], 5) == pytest.approx(0.0)
    assert precision_at_k(retrieved_sources, [], 5) == pytest.approx(0.0)
    assert mrr(retrieved_sources, []) == pytest.approx(0.0)
    assert first_relevant_rank(retrieved_sources, []) is None
