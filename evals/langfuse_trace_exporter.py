from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any

from langfuse.api.client import LangfuseAPI

from app.config import Settings

DEFAULT_EXPORT_CATEGORY = "production_failure"
DEFAULT_TRACE_SCAN_MULTIPLIER = 5
DEFAULT_FALLBACK_SUBSTRINGS = (
    "do not have enough",
    "don't have enough",
    "not enough approved information",
    "not enough information",
    "cannot confirm",
    "can't confirm",
    "unable to confirm",
)


@dataclass(frozen=True, slots=True)
class MatchedScore:
    name: str
    value: float | None
    data_type: str
    string_value: str | None
    comment: str | None


@dataclass(frozen=True, slots=True)
class BadTraceExportCandidate:
    trace_id: str
    question: str
    answer: str | None
    retrieved_context: list[dict[str, object]]
    scores: list[MatchedScore]
    model: str | None
    provider: str | None
    latency_ms: int | None
    estimated_cost_usd: float | None
    created_at: datetime
    failure_reason: str
    environment: str
    release: str | None
    version: str | None
    session_id: str | None
    trace_url: str | None


@dataclass(frozen=True, slots=True)
class LangfuseExportFilters:
    limit: int
    max_traces_to_scan: int
    score_name: str | None = None
    max_score: float | None = None
    score_string_values: tuple[str, ...] = ()
    only_errors: bool = False
    only_missing_answer: bool = False
    fallback_substrings: tuple[str, ...] = ()
    min_latency_ms: int | None = None
    min_cost_usd: float | None = None
    environments: tuple[str, ...] = ()
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None
    category: str = DEFAULT_EXPORT_CATEGORY

    def validate(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be greater than 0.")
        if self.max_traces_to_scan < self.limit:
            raise ValueError("max_traces_to_scan must be greater than or equal to limit.")
        if self.max_score is not None and self.score_name is None:
            raise ValueError("max_score requires score_name.")
        if self.score_string_values and self.score_name is None:
            raise ValueError("score_string_values requires score_name.")
        if self.min_latency_ms is not None and self.min_latency_ms < 0:
            raise ValueError("min_latency_ms must be greater than or equal to 0.")
        if self.min_cost_usd is not None and self.min_cost_usd < 0:
            raise ValueError("min_cost_usd must be greater than or equal to 0.")
        if not any(
            (
                self.score_name is not None,
                self.only_errors,
                self.only_missing_answer,
                bool(self.fallback_substrings),
                self.min_latency_ms is not None,
                self.min_cost_usd is not None,
            )
        ):
            raise ValueError(
                "At least one bad-trace filter is required. "
                "Use a score filter, --only-errors, --only-missing-answer, "
                "fallback detection, latency, or cost thresholds."
            )


@dataclass(frozen=True, slots=True)
class JsonlWriteSummary:
    total_rows: int
    newly_added_rows: int
    duplicate_rows_skipped: int


@dataclass(frozen=True, slots=True)
class LangfuseExportResult:
    candidates: list[BadTraceExportCandidate]
    written_rows: int
    duplicate_rows_skipped: int
    scanned_traces: int


class LangfuseTraceQueryClient:
    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        sdk_name: str = "production_chatbot",
        sdk_version: str = "langfuse-trace-exporter-v1",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api = LangfuseAPI(
            base_url=self._base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_public_key=public_key,
            x_langfuse_sdk_name=sdk_name,
            x_langfuse_sdk_version=sdk_version,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> LangfuseTraceQueryClient:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            raise ValueError(
                "Langfuse export requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
            )
        return cls(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )

    def fetch_bad_trace_candidates(
        self,
        filters: LangfuseExportFilters,
    ) -> tuple[list[BadTraceExportCandidate], int]:
        filters.validate()

        candidates: list[BadTraceExportCandidate] = []
        scanned_traces = 0
        page = 1

        while scanned_traces < filters.max_traces_to_scan and len(candidates) < filters.limit:
            page_size = min(100, filters.max_traces_to_scan - scanned_traces)
            response = self._api.trace.list(
                page=page,
                limit=page_size,
                from_timestamp=filters.from_timestamp,
                to_timestamp=filters.to_timestamp,
                order_by="timestamp.desc",
                environment=list(filters.environments) or None,
                fields="core,metrics",
            )
            if not response.data:
                break

            for trace_summary in response.data:
                scanned_traces += 1
                trace = self._api.trace.get(
                    trace_summary.id,
                    fields="core,io,scores,observations,metrics",
                )
                candidate = build_bad_trace_export_candidate(
                    trace,
                    filters=filters,
                    base_url=self._base_url,
                )
                if candidate is None:
                    continue
                candidates.append(candidate)
                if len(candidates) >= filters.limit:
                    break

            if len(response.data) < page_size:
                break
            page += 1

        return candidates, scanned_traces


def build_bad_trace_export_candidate(
    trace: Any,
    *,
    filters: LangfuseExportFilters,
    base_url: str | None = None,
) -> BadTraceExportCandidate | None:
    question = _extract_question(getattr(trace, "input", None))
    if question is None:
        return None

    answer = _extract_answer(getattr(trace, "output", None))
    matched_scores = _select_matching_scores(getattr(trace, "scores", []) or [], filters=filters)
    failure_reason = _determine_failure_reason(
        trace=trace,
        filters=filters,
        answer=answer,
        matched_scores=matched_scores,
    )
    if failure_reason is None:
        return None

    return BadTraceExportCandidate(
        trace_id=str(getattr(trace, "id")),
        question=question,
        answer=answer,
        retrieved_context=_extract_retrieved_context(getattr(trace, "observations", []) or []),
        scores=matched_scores,
        model=_extract_model(trace),
        provider=_extract_provider(trace),
        latency_ms=_extract_latency_ms(trace),
        estimated_cost_usd=_extract_total_cost(trace),
        created_at=_as_utc_datetime(getattr(trace, "timestamp")),
        failure_reason=failure_reason,
        environment=str(getattr(trace, "environment", "")),
        release=_normalize_optional_string(getattr(trace, "release", None)),
        version=_normalize_optional_string(getattr(trace, "version", None)),
        session_id=_normalize_optional_string(getattr(trace, "session_id", None)),
        trace_url=_build_trace_url(trace, base_url=base_url),
    )


def langfuse_trace_to_eval_example(
    trace: BadTraceExportCandidate,
    *,
    include_answer: bool = False,
    include_session_id: bool = False,
    category: str = DEFAULT_EXPORT_CATEGORY,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "langfuse",
        "trace_id": trace.trace_id,
        "created_at": trace.created_at.isoformat().replace("+00:00", "Z"),
        "failure_reason": trace.failure_reason,
        "model": trace.model,
        "provider": trace.provider,
        "latency_ms": trace.latency_ms,
        "estimated_cost_usd": trace.estimated_cost_usd,
        "environment": trace.environment,
    }
    if trace.release is not None:
        metadata["release"] = trace.release
    if trace.version is not None:
        metadata["version"] = trace.version
    if trace.trace_url is not None:
        metadata["trace_url"] = trace.trace_url
    if include_session_id and trace.session_id is not None:
        metadata["session_id"] = trace.session_id
    if trace.scores:
        metadata["matched_scores"] = [
            {
                "name": score.name,
                "value": score.value,
                "data_type": score.data_type,
                "string_value": score.string_value,
                "comment": score.comment,
            }
            for score in trace.scores
        ]
        metadata["score_name"] = trace.scores[0].name
        metadata["score_value"] = trace.scores[0].value
        if trace.scores[0].string_value is not None:
            metadata["score_string_value"] = trace.scores[0].string_value
    if trace.retrieved_context:
        metadata["retrieved_sources"] = [
            str(item["source_name"])
            for item in trace.retrieved_context
            if isinstance(item.get("source_name"), str) and str(item["source_name"]).strip()
        ]

    row: dict[str, object] = {
        "id": f"langfuse_{trace.trace_id}",
        "question": trace.question,
        "expected_facts": [],
        "expected_answer_points": [],
        "expected_source_documents": [],
        "category": category,
        "notes": (
            "Exported from Langfuse for review. Fill expected_facts, "
            "expected_answer_points, and expected_source_documents before promoting into a scored eval dataset."
        ),
        "metadata": metadata,
    }
    if include_answer and trace.answer is not None:
        row["observed_answer"] = trace.answer

    validate_export_row(row)
    return row


def validate_export_row(row: dict[str, object]) -> None:
    if not isinstance(row.get("id"), str) or not str(row["id"]).strip():
        raise ValueError("Export row id must be a non-empty string.")
    if not isinstance(row.get("question"), str) or not str(row["question"]).strip():
        raise ValueError("Export row question must be a non-empty string.")
    if not isinstance(row.get("category"), str) or not str(row["category"]).strip():
        raise ValueError("Export row category must be a non-empty string.")
    for field_name in ("expected_facts", "expected_answer_points", "expected_source_documents"):
        if not isinstance(row.get(field_name), list):
            raise ValueError(f"Export row {field_name} must be a list.")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Export row metadata must be an object.")


def write_eval_examples_jsonl(
    output_path: Path,
    rows: Sequence[dict[str, object]],
    *,
    append: bool,
    overwrite: bool,
) -> JsonlWriteSummary:
    if append and overwrite:
        raise ValueError("append and overwrite cannot both be enabled.")
    if output_path.exists() and not append and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --append to merge or --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, object]] = []
    if append and output_path.exists():
        existing_rows = load_jsonl_rows(output_path)

    merged_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    duplicate_rows_skipped = 0

    for row in existing_rows:
        validate_export_row(row)
        row_id = str(row["id"])
        if row_id in seen_ids:
            duplicate_rows_skipped += 1
            continue
        seen_ids.add(row_id)
        merged_rows.append(row)

    newly_added_rows = 0
    for row in rows:
        validate_export_row(row)
        row_id = str(row["id"])
        if row_id in seen_ids:
            duplicate_rows_skipped += 1
            continue
        seen_ids.add(row_id)
        merged_rows.append(row)
        newly_added_rows += 1

    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in merged_rows),
        encoding="utf-8",
    )
    return JsonlWriteSummary(
        total_rows=len(merged_rows),
        newly_added_rows=newly_added_rows,
        duplicate_rows_skipped=duplicate_rows_skipped,
    )


def load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row.")
        rows.append(payload)
    return rows


def parse_iso_date_or_datetime(value: str, *, inclusive_end: bool) -> datetime:
    normalized = value.strip()
    if len(normalized) == 10:
        parsed_date = date.fromisoformat(normalized)
        if inclusive_end:
            parsed_date = parsed_date + timedelta(days=1)
        return datetime.combine(parsed_date, time.min, tzinfo=UTC)

    parsed_datetime = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=UTC)
    return parsed_datetime.astimezone(UTC)


def default_max_traces_to_scan(limit: int) -> int:
    return max(limit, limit * DEFAULT_TRACE_SCAN_MULTIPLIER)


def _determine_failure_reason(
    *,
    trace: Any,
    filters: LangfuseExportFilters,
    answer: str | None,
    matched_scores: list[MatchedScore],
) -> str | None:
    if filters.only_errors and _is_error_trace(trace):
        return "error_trace"
    if matched_scores:
        if _looks_like_negative_feedback(filters.score_name, matched_scores):
            return "negative_feedback"
        return "low_score"
    if filters.only_missing_answer and answer is None:
        return "missing_answer"
    if filters.fallback_substrings and answer is not None:
        normalized_answer = answer.casefold()
        if any(substring in normalized_answer for substring in filters.fallback_substrings):
            return "fallback_answer"
    latency_ms = _extract_latency_ms(trace)
    if (
        filters.min_latency_ms is not None
        and latency_ms is not None
        and latency_ms >= filters.min_latency_ms
    ):
        return "high_latency"
    estimated_cost_usd = _extract_total_cost(trace)
    if (
        filters.min_cost_usd is not None
        and estimated_cost_usd is not None
        and estimated_cost_usd >= filters.min_cost_usd
    ):
        return "high_cost"
    return None


def _select_matching_scores(
    raw_scores: Sequence[Any],
    *,
    filters: LangfuseExportFilters,
) -> list[MatchedScore]:
    if filters.score_name is None:
        return []

    matched_scores: list[MatchedScore] = []
    expected_name = filters.score_name.casefold()
    expected_string_values = {value.casefold() for value in filters.score_string_values}

    for raw_score in raw_scores:
        score_name = _normalize_optional_string(getattr(raw_score, "name", None))
        if score_name is None or score_name.casefold() != expected_name:
            continue

        value = _coerce_optional_float(getattr(raw_score, "value", None))
        string_value = _normalize_optional_string(getattr(raw_score, "string_value", None))
        if filters.max_score is not None and (value is None or value > filters.max_score):
            continue
        if expected_string_values and (
            string_value is None or string_value.casefold() not in expected_string_values
        ):
            continue

        matched_scores.append(
            MatchedScore(
                name=score_name,
                value=value,
                data_type=str(getattr(raw_score, "data_type", "UNKNOWN")),
                string_value=string_value,
                comment=_normalize_optional_string(getattr(raw_score, "comment", None)),
            )
        )

    return matched_scores


def _looks_like_negative_feedback(
    score_name: str | None,
    matched_scores: Sequence[MatchedScore],
) -> bool:
    score_name_hint = (score_name or "").casefold()
    if "feedback" in score_name_hint:
        return True

    negative_string_values = {
        "negative",
        "thumbs_down",
        "thumbs-down",
        "downvote",
        "dislike",
        "bad",
        "false",
    }
    for score in matched_scores:
        if score.string_value is not None and score.string_value.casefold() in negative_string_values:
            return True
    return False


def _extract_question(payload: object) -> str | None:
    if isinstance(payload, dict):
        question = _normalize_optional_string(payload.get("question"))
        if question is not None:
            return question
        return _normalize_optional_string(payload.get("message"))
    if isinstance(payload, str):
        normalized = payload.strip()
        return normalized or None
    return None


def _extract_answer(payload: object) -> str | None:
    if isinstance(payload, dict):
        answer = _normalize_optional_string(payload.get("final_answer"))
        if answer is not None:
            return answer
        return _normalize_optional_string(payload.get("message"))
    if isinstance(payload, str):
        normalized = payload.strip()
        return normalized or None
    return None


def _extract_retrieved_context(observations: Sequence[Any]) -> list[dict[str, object]]:
    for observation in observations:
        observation_type = _normalize_optional_string(getattr(observation, "type", None))
        observation_name = _normalize_optional_string(getattr(observation, "name", None))
        if observation_type != "retriever" and observation_name != "retrieval":
            continue

        output = getattr(observation, "output", None)
        if not isinstance(output, dict):
            continue
        results = output.get("results")
        if not isinstance(results, list):
            continue

        normalized_results: list[dict[str, object]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized_result: dict[str, object] = {}
            for key in ("rank", "chunk_id", "source_name", "score", "content_preview"):
                if key in item:
                    normalized_result[key] = item[key]
            if normalized_result:
                normalized_results.append(normalized_result)
        if normalized_results:
            return normalized_results
    return []


def _extract_model(trace: Any) -> str | None:
    output = getattr(trace, "output", None)
    if isinstance(output, dict):
        model = _normalize_optional_string(output.get("llm_model"))
        if model is not None:
            return model

    for observation in getattr(trace, "observations", []) or []:
        model = _normalize_optional_string(getattr(observation, "model", None))
        if model is not None:
            return model
    return None


def _extract_provider(trace: Any) -> str | None:
    output = getattr(trace, "output", None)
    if isinstance(output, dict):
        provider = _normalize_optional_string(output.get("llm_provider"))
        if provider is not None:
            return provider

    metadata = getattr(trace, "metadata", None)
    if isinstance(metadata, dict):
        provider = _normalize_optional_string(metadata.get("llm_provider"))
        if provider is not None:
            return provider

    return None


def _extract_latency_ms(trace: Any) -> int | None:
    output = getattr(trace, "output", None)
    if isinstance(output, dict):
        raw_latency = output.get("latency_ms")
        if isinstance(raw_latency, (int, float)):
            return int(raw_latency)

    raw_latency_seconds = getattr(trace, "latency", None)
    if isinstance(raw_latency_seconds, (int, float)):
        return int(round(float(raw_latency_seconds) * 1000))
    return None


def _extract_total_cost(trace: Any) -> float | None:
    raw_cost = getattr(trace, "total_cost", None)
    if isinstance(raw_cost, (int, float)):
        return float(raw_cost)

    for observation in getattr(trace, "observations", []) or []:
        cost_details = getattr(observation, "cost_details", None)
        if isinstance(cost_details, dict):
            total_cost = cost_details.get("total")
            if isinstance(total_cost, (int, float)):
                return float(total_cost)
    return None


def _build_trace_url(trace: Any, *, base_url: str | None) -> str | None:
    html_path = _normalize_optional_string(getattr(trace, "html_path", None))
    if html_path is None:
        return None
    if html_path.startswith("http"):
        return html_path
    if base_url is None:
        return html_path
    return f"{base_url.rstrip('/')}/{html_path.lstrip('/')}"


def _is_error_trace(trace: Any) -> bool:
    output = getattr(trace, "output", None)
    if isinstance(output, dict):
        error_message = _normalize_optional_string(output.get("error"))
        if error_message is not None:
            return True

    for observation in getattr(trace, "observations", []) or []:
        level = _normalize_optional_string(getattr(observation, "level", None))
        status_message = _normalize_optional_string(getattr(observation, "status_message", None))
        if level == "ERROR" or status_message is not None:
            return True
    return False


def _normalize_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("Trace timestamp must be a datetime.")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
