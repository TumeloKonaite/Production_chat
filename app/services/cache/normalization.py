from __future__ import annotations

import hashlib
import json
import re

from app.services.cache.models import CacheScope

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", question.strip())
    return normalized.casefold()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_scope(scope: CacheScope) -> str:
    payload = json.dumps(scope.as_metadata(), sort_keys=True, separators=(",", ":"))
    return stable_hash(payload)
