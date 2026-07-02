from __future__ import annotations

_SUSPICIOUS_MOJIBAKE_MARKERS = ("â", "Ã", "€", "™", "œ", "ž", "�")
_PUNCTUATION_ASCII_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}
_UNICODE_PUNCTUATION_REPLACEMENTS = str.maketrans(
    {
        "\u00A0": " ",
        **_PUNCTUATION_ASCII_REPLACEMENTS,
    }
)
_COMMON_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
}


def normalize_llm_text(text: str) -> str:
    normalized = _replace_common_mojibake_sequences(text)
    normalized = _repair_common_mojibake(normalized)
    normalized = normalized.translate(_UNICODE_PUNCTUATION_REPLACEMENTS)
    return normalized


def _replace_common_mojibake_sequences(text: str) -> str:
    normalized = text
    for mojibake, replacement in _COMMON_MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(mojibake, replacement)
    return normalized


def _repair_common_mojibake(text: str) -> str:
    if not any(marker in text for marker in _SUSPICIOUS_MOJIBAKE_MARKERS):
        return text

    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    if _count_suspicious_markers(repaired) >= _count_suspicious_markers(text):
        return text
    return repaired


def _count_suspicious_markers(text: str) -> int:
    return sum(text.count(marker) for marker in _SUSPICIOUS_MOJIBAKE_MARKERS)
