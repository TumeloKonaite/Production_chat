from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source: str
    text: str
    updated_at: datetime


def load_source_documents(source_dir: Path | None = None) -> list[SourceDocument]:
    resolved_source_dir = source_dir or DEFAULT_SOURCE_DIR
    documents: list[SourceDocument] = []

    for path in sorted(resolved_source_dir.rglob("*.md")):
        documents.append(
            SourceDocument(
                source=path.name,
                text=path.read_text(encoding="utf-8"),
                updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            )
        )

    return documents
