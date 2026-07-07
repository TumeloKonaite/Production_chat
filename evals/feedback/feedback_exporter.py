from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.tracing import TraceStatus, TraceStepType
from app.repositories.models import ChatTrace, ChatTraceStep, KnowledgeChunk, Message, RetrievalLog
from evals.feedback.feedback_dataset import (
    FEEDBACK_DATASET_SOURCE,
    REDACTED_ANSWER_TEXT,
    REDACTED_COMMENT_TEXT,
    REDACTED_CONTEXT_TEXT,
    REDACTED_QUESTION_TEXT,
    FeedbackContextItem,
    FeedbackEvalExample,
)
from evals.langfuse.langfuse_trace_exporter import (
    parse_iso_date_or_datetime,
    write_eval_examples_jsonl,
)

NEGATIVE_FEEDBACK_VALUES = {
    "negative",
    "thumbs_down",
    "thumbs-down",
    "downvote",
    "dislike",
    "bad",
    "incorrect",
    "failed",
    "error",
}
POSITIVE_FEEDBACK_VALUES = {"positive", "thumbs_up", "thumbs-up", "upvote", "like", "good"}
NEUTRAL_FEEDBACK_VALUES = {"neutral", "mixed"}
FALLBACK_ANSWER_SUBSTRINGS = (
    "do not have enough approved information",
    "do not have enough approved",
    "not enough approved information",
    "i do not have enough",
)


class ProductionFeedbackExportDisabledError(RuntimeError):
    """Raised when production feedback export is disabled by configuration."""


@dataclass(frozen=True, slots=True)
class ProductionFeedbackExportFilters:
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None
    rating: str | None = None
    reason: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    environment: str | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class ProductionFeedbackExportResult:
    examples: list[FeedbackEvalExample]
    written_rows: int
    duplicate_rows_skipped: int
    output_path: Path


def export_feedback_dataset(
    *,
    session: Session,
    settings: Settings,
    filters: ProductionFeedbackExportFilters,
    output_path: Path,
    allow_raw_text: bool,
    append: bool,
    overwrite: bool,
) -> ProductionFeedbackExportResult:
    if not settings.enable_production_feedback_export:
        raise ProductionFeedbackExportDisabledError(
            "Production feedback export is disabled. Set ENABLE_PRODUCTION_FEEDBACK_EXPORT=true to enable it."
        )
    if allow_raw_text and not settings.allow_raw_production_text_in_evals:
        raise ProductionFeedbackExportDisabledError(
            "Raw production text export is disabled. Set ALLOW_RAW_PRODUCTION_TEXT_IN_EVALS=true to allow it."
        )

    examples = list_feedback_examples(
        session=session,
        filters=filters,
        allow_raw_text=allow_raw_text and settings.allow_raw_production_text_in_evals,
    )
    rows = [example.to_json_row() for example in examples]
    write_summary = write_eval_examples_jsonl(
        output_path,
        rows,
        append=append,
        overwrite=overwrite,
    )
    return ProductionFeedbackExportResult(
        examples=examples,
        written_rows=write_summary.newly_added_rows,
        duplicate_rows_skipped=write_summary.duplicate_rows_skipped,
        output_path=output_path,
    )


def list_feedback_examples(
    *,
    session: Session,
    filters: ProductionFeedbackExportFilters,
    allow_raw_text: bool,
) -> list[FeedbackEvalExample]:
    traces = _load_candidate_traces(session=session, filters=filters)
    examples: list[FeedbackEvalExample] = []

    for trace in traces:
        example = _build_feedback_example(
            session=session,
            trace=trace,
            allow_raw_text=allow_raw_text,
        )
        if example is None:
            continue
        if not _matches_filters(example, filters=filters):
            continue
        examples.append(example)
        if filters.limit is not None and len(examples) >= filters.limit:
            break

    return examples


def parse_feedback_filters(args) -> ProductionFeedbackExportFilters:
    return ProductionFeedbackExportFilters(
        from_timestamp=(
            parse_iso_date_or_datetime(args.from_date, inclusive_end=False)
            if args.from_date
            else None
        ),
        to_timestamp=(
            parse_iso_date_or_datetime(args.to_date, inclusive_end=True)
            if args.to_date
            else None
        ),
        rating=_normalize_casefolded_string(args.rating),
        reason=_normalize_optional_string(args.reason),
        model_name=_normalize_optional_string(args.model_name),
        prompt_version=_normalize_optional_string(args.prompt_version),
        environment=_normalize_optional_string(args.environment),
        limit=args.limit,
    )


def _load_candidate_traces(
    *,
    session: Session,
    filters: ProductionFeedbackExportFilters,
) -> list[ChatTrace]:
    statement: Select[tuple[ChatTrace]] = select(ChatTrace).order_by(
        ChatTrace.created_at.desc(),
        ChatTrace.id.desc(),
    )
    if filters.from_timestamp is not None:
        statement = statement.where(ChatTrace.created_at >= filters.from_timestamp)
    if filters.to_timestamp is not None:
        statement = statement.where(ChatTrace.created_at < filters.to_timestamp)
    if filters.model_name is not None:
        statement = statement.where(ChatTrace.llm_model == filters.model_name)
    if filters.prompt_version is not None:
        statement = statement.where(ChatTrace.prompt_version == filters.prompt_version)
    limit = max(filters.limit or 200, 1)
    return list(session.scalars(statement.limit(limit * 4)))


def _build_feedback_example(
    *,
    session: Session,
    trace: ChatTrace,
    allow_raw_text: bool,
) -> FeedbackEvalExample | None:
    if not isinstance(trace.input_text, str) or not trace.input_text.strip():
        return None
    feedback_details = _extract_feedback_details(trace)
    if feedback_details is None:
        return None

    retrieval_log = _find_retrieval_log(session=session, trace=trace)
    context_items = _load_context_items(session=session, trace=trace, retrieval_log=retrieval_log)
    assistant_message = _find_assistant_message(session=session, trace=trace)
    environment = _extract_environment(trace.trace_metadata)

    metadata: dict[str, Any] = {
        "source": FEEDBACK_DATASET_SOURCE,
        "conversation_id": trace.conversation_id,
        "request_id": trace.request_id,
        "user_id": trace.user_id,
        "environment": environment,
        "route": trace.trace_metadata.get("route"),
        "channel": trace.trace_metadata.get("channel"),
        "trace_status": trace.status,
        "error_message": trace.error_message,
        "raw_text_included": allow_raw_text,
    }
    if retrieval_log is not None:
        metadata["retrieved_sources"] = list(retrieval_log.retrieved_sources or [])
        metadata["retrieved_chunk_ids"] = list(retrieval_log.retrieved_chunk_ids or [])
        metadata["used_fallback"] = bool(retrieval_log.used_fallback)
    if assistant_message is not None:
        metadata["assistant_message_id"] = assistant_message.id

    return FeedbackEvalExample(
        id=f"feedback_{trace.id}",
        question=_protect_text(trace.input_text, allow_raw_text=allow_raw_text, placeholder=REDACTED_QUESTION_TEXT),
        actual_answer=_protect_text(
            trace.output_text,
            allow_raw_text=allow_raw_text,
            placeholder=REDACTED_ANSWER_TEXT,
        ),
        expected_answer=_protect_text(
            feedback_details.expected_answer,
            allow_raw_text=allow_raw_text,
            placeholder=REDACTED_ANSWER_TEXT,
        ),
        expected_facts=list(feedback_details.expected_facts),
        expected_answer_points=list(feedback_details.expected_answer_points),
        expected_source_documents=list(feedback_details.expected_source_documents),
        feedback_rating=feedback_details.rating,
        feedback_reason=feedback_details.reason,
        feedback_comment=_protect_text(
            feedback_details.comment,
            allow_raw_text=allow_raw_text,
            placeholder=REDACTED_COMMENT_TEXT,
        ),
        source=FEEDBACK_DATASET_SOURCE,
        trace_id=trace.id,
        session_id=trace.session_id,
        message_id=(retrieval_log.message_id if retrieval_log is not None else None),
        langfuse_trace_id=feedback_details.langfuse_trace_id,
        model_provider=trace.llm_provider,
        model_name=trace.llm_model,
        retriever_type=trace.retriever_type,
        top_k=(retrieval_log.top_k if retrieval_log is not None else feedback_details.top_k),
        prompt_version=trace.prompt_version,
        created_at=_as_utc_datetime(trace.created_at),
        metadata=metadata,
        context=[
            FeedbackContextItem(
                source=item.source,
                section=item.section,
                content=_protect_text(
                    item.content,
                    allow_raw_text=allow_raw_text,
                    placeholder=REDACTED_CONTEXT_TEXT,
                )
                or REDACTED_CONTEXT_TEXT,
                similarity=item.similarity,
            )
            for item in context_items
        ],
    )


@dataclass(frozen=True, slots=True)
class _FeedbackDetails:
    rating: str
    reason: str | None
    comment: str | None
    expected_answer: str | None
    expected_facts: list[str]
    expected_answer_points: list[str]
    expected_source_documents: list[str]
    langfuse_trace_id: str | None
    top_k: int | None


def _extract_feedback_details(trace: ChatTrace) -> _FeedbackDetails | None:
    metadata = dict(trace.trace_metadata or {})
    feedback_payloads = [metadata]
    for key in ("feedback", "user_feedback", "review_feedback"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping):
            feedback_payloads.append(nested)

    rating = None
    reason = None
    comment = None
    expected_answer = None
    expected_facts: list[str] = []
    expected_answer_points: list[str] = []
    expected_source_documents: list[str] = []
    langfuse_trace_id = None
    top_k = None

    for payload in feedback_payloads:
        rating = rating or _normalize_feedback_rating(
            payload.get("feedback_rating")
            or payload.get("rating")
            or payload.get("user_feedback")
            or payload.get("thumb_rating")
        )
        reason = reason or _normalize_optional_string(
            payload.get("feedback_reason") or payload.get("reason")
        )
        comment = comment or _normalize_optional_string(
            payload.get("feedback_comment") or payload.get("comment")
        )
        expected_answer = expected_answer or _normalize_optional_string(
            payload.get("expected_answer") or payload.get("corrected_answer")
        )
        if not expected_facts:
            expected_facts = _normalize_string_list(payload.get("expected_facts"))
        if not expected_answer_points:
            expected_answer_points = _normalize_string_list(
                payload.get("expected_answer_points")
            )
        if not expected_source_documents:
            expected_source_documents = _normalize_string_list(
                payload.get("expected_source_documents")
            )
        langfuse_trace_id = langfuse_trace_id or _normalize_optional_string(
            payload.get("langfuse_trace_id")
        )
        if top_k is None:
            top_k = _normalize_optional_int(payload.get("top_k") or payload.get("retrieval_top_k"))

    if rating is None:
        rating, reason = _infer_feedback_from_trace(trace=trace, reason=reason)
    if rating is None:
        return None

    return _FeedbackDetails(
        rating=rating,
        reason=reason,
        comment=comment,
        expected_answer=expected_answer,
        expected_facts=expected_facts,
        expected_answer_points=expected_answer_points,
        expected_source_documents=expected_source_documents,
        langfuse_trace_id=langfuse_trace_id,
        top_k=top_k,
    )


def _infer_feedback_from_trace(
    *,
    trace: ChatTrace,
    reason: str | None,
) -> tuple[str | None, str | None]:
    if trace.status == TraceStatus.ERROR.value:
        return "negative", reason or "trace_error"
    if isinstance(trace.output_text, str):
        normalized_output = trace.output_text.casefold()
        if any(marker in normalized_output for marker in FALLBACK_ANSWER_SUBSTRINGS):
            return "negative", reason or "fallback_answer"
    for step in trace.steps:
        if step.step_type == TraceStepType.ERROR.value:
            return "negative", reason or "trace_error"
        if step.step_type == TraceStepType.RETRIEVAL_COMPLETED.value:
            retrieved_chunks = _extract_step_retrieved_chunks(step)
            if not retrieved_chunks:
                return "negative", reason or "retrieval_failure"
    return None, reason


def _find_retrieval_log(*, session: Session, trace: ChatTrace) -> RetrievalLog | None:
    if not trace.conversation_id or not trace.input_text:
        return None

    logs = list(
        session.scalars(
            select(RetrievalLog)
            .where(
                RetrievalLog.conversation_id == trace.conversation_id,
                RetrievalLog.query == trace.input_text,
            )
            .order_by(RetrievalLog.created_at.desc(), RetrievalLog.id.desc())
            .limit(5)
        )
    )
    if not logs:
        return None

    trace_created_at = _as_utc_datetime(trace.created_at)
    return min(
        logs,
        key=lambda item: abs((_as_utc_datetime(item.created_at) - trace_created_at).total_seconds()),
    )


def _find_assistant_message(*, session: Session, trace: ChatTrace) -> Message | None:
    if not trace.conversation_id:
        return None

    if trace.output_text:
        exact_match = session.scalar(
            select(Message)
            .where(
                Message.conversation_id == trace.conversation_id,
                Message.role == "assistant",
                Message.content == trace.output_text,
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(1)
        )
        if exact_match is not None:
            return exact_match

    return session.scalar(
        select(Message)
        .where(
            Message.conversation_id == trace.conversation_id,
            Message.role == "assistant",
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(1)
    )


def _load_context_items(
    *,
    session: Session,
    trace: ChatTrace,
    retrieval_log: RetrievalLog | None,
) -> list[FeedbackContextItem]:
    if retrieval_log is not None and retrieval_log.retrieved_chunk_ids:
        chunk_lookup = {
            chunk.id: chunk
            for chunk in session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.id.in_(retrieval_log.retrieved_chunk_ids))
            )
        }
        ordered_context_items: list[FeedbackContextItem] = []
        for index, chunk_id in enumerate(retrieval_log.retrieved_chunk_ids):
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            similarity = None
            if index < len(retrieval_log.similarity_scores):
                similarity = retrieval_log.similarity_scores[index]
            ordered_context_items.append(
                FeedbackContextItem(
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    similarity=float(similarity) if similarity is not None else 1.0,
                )
            )
        if ordered_context_items:
            return ordered_context_items

    retrieval_steps = [
        step
        for step in trace.steps
        if step.step_type == TraceStepType.RETRIEVAL_COMPLETED.value
    ]
    for step in retrieval_steps:
        step_chunks = _extract_step_retrieved_chunks(step)
        if not step_chunks:
            continue
        context_items: list[FeedbackContextItem] = []
        for item in step_chunks:
            source = _normalize_optional_string(item.get("source"))
            section = _normalize_optional_string(item.get("section"))
            if source is None or section is None:
                continue
            chunk = session.scalar(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.source == source,
                    KnowledgeChunk.section == section,
                )
            )
            if chunk is None:
                continue
            context_items.append(
                FeedbackContextItem(
                    source=chunk.source,
                    section=chunk.section,
                    content=chunk.content,
                    similarity=float(item.get("score") or 1.0),
                )
            )
        if context_items:
            return context_items
    return []


def _extract_step_retrieved_chunks(step: ChatTraceStep) -> list[dict[str, object]]:
    output_payload = dict(step.output_payload or {})
    raw_chunks = output_payload.get("retrieved_chunks")
    if not isinstance(raw_chunks, list):
        return []
    return [item for item in raw_chunks if isinstance(item, dict)]


def _matches_filters(
    example: FeedbackEvalExample,
    *,
    filters: ProductionFeedbackExportFilters,
) -> bool:
    if filters.rating is not None and example.feedback_rating != filters.rating:
        return False
    if (
        filters.reason is not None
        and _normalize_casefolded_string(example.feedback_reason) != filters.reason.casefold()
    ):
        return False
    if (
        filters.model_name is not None
        and _normalize_casefolded_string(example.model_name) != filters.model_name.casefold()
    ):
        return False
    if (
        filters.prompt_version is not None
        and _normalize_casefolded_string(example.prompt_version) != filters.prompt_version.casefold()
    ):
        return False
    if filters.environment is not None:
        example_environment = _normalize_optional_string(example.metadata.get("environment"))
        if _normalize_casefolded_string(example_environment) != filters.environment.casefold():
            return False
    return True


def _normalize_feedback_rating(value: object) -> str | None:
    normalized = _normalize_casefolded_string(value)
    if normalized is None:
        return None
    if normalized in NEGATIVE_FEEDBACK_VALUES:
        return "negative"
    if normalized in POSITIVE_FEEDBACK_VALUES:
        return "positive"
    if normalized in NEUTRAL_FEEDBACK_VALUES:
        return "neutral"
    if normalized == "unknown":
        return "unknown"
    return None


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    items: list[str] = []
    for raw_item in value:
        normalized_item = _normalize_optional_string(raw_item)
        if normalized_item is not None:
            items.append(normalized_item)
    return items


def _normalize_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_casefolded_string(value: object) -> str | None:
    normalized = _normalize_optional_string(value)
    return normalized.casefold() if normalized is not None else None


def _normalize_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _extract_environment(metadata: Mapping[str, Any]) -> str | None:
    environment = metadata.get("environment")
    if isinstance(environment, str) and environment.strip():
        return environment.strip()
    return None


def _protect_text(
    value: str | None,
    *,
    allow_raw_text: bool,
    placeholder: str,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized if allow_raw_text else placeholder


def _as_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
