from __future__ import annotations

import pytest

from app.domain.evals.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_at_k_use_unique_ranked_sources() -> None:
    retrieved_sources = ["projects.md", "projects.md", "skills.md", "profile.md"]
    expected_sources = ["projects.md", "profile.md"]

    precision = precision_at_k(retrieved_sources, expected_sources, k=3)
    recall = recall_at_k(retrieved_sources, expected_sources, k=3)

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)


def test_mean_reciprocal_rank_uses_first_relevant_source() -> None:
    retrieved_sources = ["contact.md", "skills.md", "projects.md"]
    expected_sources = ["projects.md", "profile.md"]

    score = mean_reciprocal_rank(retrieved_sources, expected_sources)

    assert score == pytest.approx(1 / 3)


def test_ndcg_at_k_rewards_relevant_documents_near_top() -> None:
    retrieved_sources = ["projects.md", "contact.md", "skills.md"]
    expected_sources = ["projects.md", "skills.md"]

    score = ndcg_at_k(retrieved_sources, expected_sources, k=3)

    ideal_dcg = (1 / 1.0) + (1 / 1.584962500721156)
    actual_dcg = (1 / 1.0) + (1 / 2.0)
    assert score == pytest.approx(actual_dcg / ideal_dcg)


def test_unsupported_question_metrics_reward_clean_fallback_retrieval() -> None:
    empty_retrieval = []
    noisy_retrieval = ["projects.md"]
    expected_sources: list[str] = []

    assert precision_at_k(empty_retrieval, expected_sources, k=5) == pytest.approx(1.0)
    assert recall_at_k(empty_retrieval, expected_sources, k=5) == pytest.approx(1.0)
    assert mean_reciprocal_rank(empty_retrieval, expected_sources) == pytest.approx(1.0)
    assert ndcg_at_k(empty_retrieval, expected_sources, k=5) == pytest.approx(1.0)

    assert precision_at_k(noisy_retrieval, expected_sources, k=5) == pytest.approx(0.0)
    assert recall_at_k(noisy_retrieval, expected_sources, k=5) == pytest.approx(0.0)
    assert mean_reciprocal_rank(noisy_retrieval, expected_sources) == pytest.approx(0.0)
    assert ndcg_at_k(noisy_retrieval, expected_sources, k=5) == pytest.approx(0.0)
