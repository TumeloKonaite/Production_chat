from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value.strip())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    normalized = _normalize_value(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return stable_hash(encoded)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(nested_value)
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value
