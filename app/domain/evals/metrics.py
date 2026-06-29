from __future__ import annotations

from math import log2


def precision_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    *,
    k: int,
) -> float:
    top_k_sources = _top_k_unique_sources(retrieved_sources, k)
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 1.0 if not top_k_sources else 0.0
    if k <= 0:
        return 0.0

    relevant_count = sum(1 for source in top_k_sources if source in expected_set)
    return relevant_count / k


def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    *,
    k: int,
) -> float:
    top_k_sources = _top_k_unique_sources(retrieved_sources, k)
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 1.0 if not top_k_sources else 0.0

    relevant_count = sum(1 for source in top_k_sources if source in expected_set)
    return relevant_count / len(expected_set)


def mean_reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    ranked_sources = _top_k_unique_sources(retrieved_sources, None)
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 1.0 if not ranked_sources else 0.0

    for index, source in enumerate(ranked_sources, start=1):
        if source in expected_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    *,
    k: int,
) -> float:
    top_k_sources = _top_k_unique_sources(retrieved_sources, k)
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 1.0 if not top_k_sources else 0.0
    if k <= 0:
        return 0.0

    dcg = 0.0
    for index, source in enumerate(top_k_sources, start=1):
        if source in expected_set:
            dcg += 1.0 / log2(index + 1)

    ideal_hits = min(len(expected_set), k)
    ideal_dcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _normalize_sources(sources: list[str]) -> set[str]:
    return {
        source.strip().casefold()
        for source in sources
        if source.strip()
    }


def _top_k_unique_sources(sources: list[str], k: int | None) -> list[str]:
    unique_sources: list[str] = []
    seen: set[str] = set()
    for source in sources:
        normalized_source = source.strip().casefold()
        if not normalized_source or normalized_source in seen:
            continue
        seen.add(normalized_source)
        unique_sources.append(normalized_source)
        if k is not None and len(unique_sources) >= k:
            break
    return unique_sources
