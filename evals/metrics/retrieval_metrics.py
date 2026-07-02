from __future__ import annotations


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    if k <= 0:
        return 0.0

    top_k_sources = unique_ranked_sources(retrieved_sources, k=k)
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 0.0

    return 1.0 if any(_normalize_source(source) in expected_set for source in top_k_sources) else 0.0


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    expected_set = _normalize_sources(expected_sources)
    if not expected_set or k <= 0:
        return 0.0

    top_k_sources = unique_ranked_sources(retrieved_sources, k=k)
    relevant_count = sum(1 for source in top_k_sources if _normalize_source(source) in expected_set)
    return relevant_count / len(expected_set)


def precision_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    if k <= 0:
        return 0.0

    top_k_sources = unique_ranked_sources(retrieved_sources, k=k)
    if not top_k_sources:
        return 0.0

    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return 0.0

    relevant_count = sum(1 for source in top_k_sources if _normalize_source(source) in expected_set)
    return relevant_count / len(top_k_sources)


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    rank = first_relevant_rank(retrieved_sources, expected_sources)
    if rank is None:
        return 0.0
    return 1.0 / rank


def first_relevant_rank(retrieved_sources: list[str], expected_sources: list[str]) -> int | None:
    expected_set = _normalize_sources(expected_sources)
    if not expected_set:
        return None

    ranked_sources = unique_ranked_sources(retrieved_sources)
    for index, source in enumerate(ranked_sources, start=1):
        if _normalize_source(source) in expected_set:
            return index
    return None


def unique_ranked_sources(sources: list[str], k: int | None = None) -> list[str]:
    unique_sources: list[str] = []
    seen: set[str] = set()

    for source in sources:
        normalized_source = _normalize_source(source)
        if not normalized_source or normalized_source in seen:
            continue

        seen.add(normalized_source)
        unique_sources.append(source.strip())
        if k is not None and len(unique_sources) >= k:
            break

    return unique_sources


def _normalize_sources(sources: list[str]) -> set[str]:
    return {
        normalized_source
        for source in sources
        if (normalized_source := _normalize_source(source))
    }


def _normalize_source(source: str) -> str:
    return source.strip().casefold()
