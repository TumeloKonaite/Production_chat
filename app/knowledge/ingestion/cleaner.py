from __future__ import annotations

import re

_MULTISPACE_RE = re.compile(r"[^\S\r\n]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")


def clean_markdown_text(text: str) -> str:
    # Normalize line endings first so the rest of the cleaner only has to work
    # with one newline format.
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = _clean_lines(normalized_text.split("\n"))
    lines_with_heading_spacing = _ensure_blank_line_after_headings(cleaned_lines)
    return "\n".join(lines_with_heading_spacing).strip()


def _clean_lines(lines: list[str]) -> list[str]:
    cleaned_lines: list[str] = []
    saw_blank_line = False

    for raw_line in lines:
        stripped_line = raw_line.strip()

        if not stripped_line:
            # Collapse repeated blank lines, but keep a single separator between
            # blocks once real content has started.
            if not saw_blank_line and cleaned_lines:
                cleaned_lines.append("")
            saw_blank_line = True
            continue

        cleaned_lines.append(_normalize_non_blank_line(stripped_line))
        saw_blank_line = False

    return cleaned_lines


def _normalize_non_blank_line(line: str) -> str:
    heading_match = _HEADING_RE.match(line)
    if heading_match:
        # Preserve markdown headings while normalizing spacing after the `#`
        # markers so chunking metadata stays predictable.
        return f"{heading_match.group(1)} {heading_match.group(2)}"

    # For normal text lines, collapse tabs and repeated spaces without removing
    # the line itself.
    return _MULTISPACE_RE.sub(" ", line)


def _ensure_blank_line_after_headings(lines: list[str]) -> list[str]:
    normalized_lines: list[str] = []

    for index, line in enumerate(lines):
        normalized_lines.append(line)
        if not _is_heading(line):
            continue

        # Insert a blank line after a heading when content follows immediately
        # so markdown sections remain visually and structurally consistent.
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        if next_line not in (None, ""):
            normalized_lines.append("")

    return normalized_lines


def _is_heading(line: str) -> bool:
    return _HEADING_RE.match(line) is not None
