from __future__ import annotations


def build_exact_cache_key(request_hash: str) -> str:
    return f"chat:exact:v1:{request_hash}"


def build_request_lock_key(request_hash: str) -> str:
    return f"chat:lock:v1:{request_hash}"


def build_rate_limit_key(
    *,
    endpoint: str,
    actor_type: str,
    actor_id: str,
    window_bucket: int,
) -> str:
    return f"rate:{endpoint}:{actor_type}:{actor_id}:{window_bucket}"


def build_state_key(*, namespace: str, identifier: str) -> str:
    return f"state:{namespace}:v1:{identifier}"
