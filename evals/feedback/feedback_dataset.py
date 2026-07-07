from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FEEDBACK_DATASET_SOURCE = "production_feedback"
FEEDBACK_DATASET_CATEGORY = "production_feedback"
REDACTED_QUESTION_TEXT = "[redacted production question]"
REDACTED_ANSWER_TEXT = "[redacted production answer]"
REDACTED_COMMENT_TEXT = "[redacted production feedback comment]"
REDACTED_CONTEXT_TEXT = "[redacted production context]"

_VALID_FEEDBACK_RATINGS = {"negative", "positive", "neutral", "unknown"}


class FeedbackContextItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    similarity: float = 1.0


class FeedbackEvalExample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    actual_answer: str | None = None
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_source_documents: list[str] = Field(default_factory=list)
    feedback_rating: Literal["negative", "positive", "neutral", "unknown"] = "negative"
    feedback_reason: str | None = None
    feedback_comment: str | None = None
    source: str = FEEDBACK_DATASET_SOURCE
    trace_id: str = Field(..., min_length=1)
    session_id: str | None = None
    message_id: str | None = None
    langfuse_trace_id: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    retriever_type: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    prompt_version: str | None = None
    created_at: datetime
    category: str = FEEDBACK_DATASET_CATEGORY
    metadata: dict[str, Any] = Field(default_factory=dict)
    context: list[FeedbackContextItem] = Field(default_factory=list)

    @field_validator("feedback_rating", mode="before")
    @classmethod
    def _normalize_rating(cls, value: object) -> str:
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().casefold()
        return normalized if normalized in _VALID_FEEDBACK_RATINGS else "unknown"

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            created_at = value
        elif isinstance(value, str):
            created_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("created_at must be a datetime or ISO-formatted string.")
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=UTC)
        return created_at.astimezone(UTC)

    @property
    def has_generation_labels(self) -> bool:
        return bool(self.expected_facts or self.expected_answer or self.expected_answer_points)

    @property
    def has_retrieval_labels(self) -> bool:
        return bool(self.expected_source_documents)

    def to_json_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class FeedbackDatasetSummary:
    dataset_path: Path
    dataset_version: str
    case_count: int
    negative_count: int
    positive_count: int
    neutral_count: int
    unknown_count: int
    labeled_generation_count: int
    labeled_retrieval_count: int
    reasons: list[str]
    feedback_date_from: str | None
    feedback_date_to: str | None
    production_trace_ids: list[str]
    langfuse_trace_ids: list[str]

def load_feedback_dataset(path: Path) -> list[FeedbackEvalExample]:
    rows: list[FeedbackEvalExample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row.")
        if str(payload.get("source", "")).strip().casefold() != FEEDBACK_DATASET_SOURCE:
            continue
        rows.append(FeedbackEvalExample.model_validate(payload))
    return rows


def summarize_feedback_dataset(path: Path) -> FeedbackDatasetSummary | None:
    if not path.exists():
        return None
    examples = load_feedback_dataset(path)
    if not examples:
        return None

    created_at_values = sorted(example.created_at for example in examples)
    reasons = sorted(
        {
            example.feedback_reason.strip()
            for example in examples
            if isinstance(example.feedback_reason, str) and example.feedback_reason.strip()
        }
    )
    production_trace_ids = [example.trace_id for example in examples if example.trace_id]
    langfuse_trace_ids = [
        example.langfuse_trace_id
        for example in examples
        if isinstance(example.langfuse_trace_id, str) and example.langfuse_trace_id.strip()
    ]
    counts_by_rating = {
        rating: sum(1 for example in examples if example.feedback_rating == rating)
        for rating in _VALID_FEEDBACK_RATINGS
    }
    return FeedbackDatasetSummary(
        dataset_path=path,
        dataset_version=_hash_file(path),
        case_count=len(examples),
        negative_count=counts_by_rating["negative"],
        positive_count=counts_by_rating["positive"],
        neutral_count=counts_by_rating["neutral"],
        unknown_count=counts_by_rating["unknown"],
        labeled_generation_count=sum(1 for example in examples if example.has_generation_labels),
        labeled_retrieval_count=sum(1 for example in examples if example.has_retrieval_labels),
        reasons=reasons,
        feedback_date_from=created_at_values[0].isoformat().replace("+00:00", "Z"),
        feedback_date_to=created_at_values[-1].isoformat().replace("+00:00", "Z"),
        production_trace_ids=production_trace_ids,
        langfuse_trace_ids=langfuse_trace_ids,
    )


def build_feedback_tracking_params(summary: FeedbackDatasetSummary) -> dict[str, object]:
    return {
        "dataset_source": FEEDBACK_DATASET_SOURCE,
        "feedback_dataset_path": str(summary.dataset_path),
        "feedback_dataset_version": summary.dataset_version,
        "feedback_case_count": summary.case_count,
        "feedback_negative_count": summary.negative_count,
        "feedback_positive_count": summary.positive_count,
        "feedback_neutral_count": summary.neutral_count,
        "feedback_unknown_count": summary.unknown_count,
        "feedback_labeled_generation_count": summary.labeled_generation_count,
        "feedback_labeled_retrieval_count": summary.labeled_retrieval_count,
        "feedback_date_from": summary.feedback_date_from,
        "feedback_date_to": summary.feedback_date_to,
        "feedback_reasons": ",".join(summary.reasons) if summary.reasons else "none",
    }


def write_feedback_metadata_artifacts(
    *,
    output_dir: Path,
    summary: FeedbackDatasetSummary,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = output_dir / "feedback_dataset_summary.json"
    trace_ids_path = output_dir / "production_trace_ids.json"
    artifact_paths = {
        "feedback_dataset_jsonl": summary.dataset_path,
        "feedback_dataset_summary_json": dataset_summary_path,
        "production_trace_ids_json": trace_ids_path,
    }

    dataset_summary_path.write_text(
        json.dumps(
            {
                "dataset_source": FEEDBACK_DATASET_SOURCE,
                "dataset_version": summary.dataset_version,
                "case_count": summary.case_count,
                "negative_count": summary.negative_count,
                "positive_count": summary.positive_count,
                "neutral_count": summary.neutral_count,
                "unknown_count": summary.unknown_count,
                "labeled_generation_count": summary.labeled_generation_count,
                "labeled_retrieval_count": summary.labeled_retrieval_count,
                "feedback_date_from": summary.feedback_date_from,
                "feedback_date_to": summary.feedback_date_to,
                "feedback_reasons": summary.reasons,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_ids_path.write_text(
        json.dumps(summary.production_trace_ids, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if summary.langfuse_trace_ids:
        langfuse_trace_ids_path = output_dir / "langfuse_trace_ids.json"
        langfuse_trace_ids_path.write_text(
            json.dumps(summary.langfuse_trace_ids, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        artifact_paths["langfuse_trace_ids_json"] = langfuse_trace_ids_path

    return artifact_paths


def _hash_file(path: Path) -> str:
    digest = sha256(path.read_bytes()).hexdigest()
    return digest[:16]
