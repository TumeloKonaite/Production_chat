from __future__ import annotations


def normalize_query(query: str) -> str:
    return query.strip()


def resolve_top_k(top_k: int | None, *, default_top_k: int) -> int:
    limit = default_top_k if top_k is None else top_k
    if limit <= 0:
        raise ValueError("top_k must be greater than 0.")
    return limit
