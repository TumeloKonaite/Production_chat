from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from evals.langfuse.langfuse_trace_exporter import (
    DEFAULT_EXPORT_CATEGORY,
    DEFAULT_FALLBACK_SUBSTRINGS,
    LangfuseExportFilters,
    LangfuseTraceQueryClient,
    default_max_traces_to_scan,
    langfuse_trace_to_eval_example,
    parse_iso_date_or_datetime,
    write_eval_examples_jsonl,
)


def parse_args(default_limit: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export low-quality or failed Langfuse traces into a review JSONL dataset "
            "that can be promoted into the repo's eval suites."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help=f"Maximum number of exported traces. Defaults to LANGFUSE_EXPORT_DEFAULT_LIMIT ({default_limit}).",
    )
    parser.add_argument(
        "--max-traces-to-scan",
        type=int,
        default=None,
        help="Maximum number of recent traces to scan before stopping.",
    )
    parser.add_argument(
        "--score-name",
        default=None,
        help="Score name to filter on, for example answer_quality or user_feedback.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Upper score threshold for score-based export, for example 0.6.",
    )
    parser.add_argument(
        "--score-string-value",
        action="append",
        default=[],
        help="Optional categorical or boolean score value to match exactly. Repeat for multiple values.",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Export traces that recorded an error.",
    )
    parser.add_argument(
        "--only-missing-answer",
        action="store_true",
        help="Export traces where no final answer was captured.",
    )
    parser.add_argument(
        "--detect-fallback-answer",
        action="store_true",
        help="Export traces whose final answer looks like a fallback or refusal.",
    )
    parser.add_argument(
        "--fallback-substring",
        action="append",
        default=[],
        help="Additional substring that marks a fallback answer. Repeat for multiple values.",
    )
    parser.add_argument(
        "--min-latency-ms",
        type=int,
        default=None,
        help="Export traces at or above this latency threshold.",
    )
    parser.add_argument(
        "--min-cost-usd",
        type=float,
        default=None,
        help="Export traces at or above this total estimated cost.",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Inclusive ISO date or datetime lower bound, for example 2026-07-01 or 2026-07-01T00:00:00Z.",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Inclusive ISO date or datetime upper bound.",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        help="Langfuse environment to include. Repeat for multiple environments.",
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        help="Include the observed production answer in the exported JSONL row.",
    )
    parser.add_argument(
        "--include-session-id",
        action="store_true",
        help="Include the raw session_id in metadata. Omitted by default.",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_EXPORT_CATEGORY,
        help=f"Category value written to exported rows. Defaults to {DEFAULT_EXPORT_CATEGORY}.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to an existing JSONL file and deduplicate by id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    args = parse_args(settings.langfuse_export_default_limit)

    if args.max_score is not None and args.score_name is None:
        raise SystemExit("--max-score requires --score-name.")
    if args.score_string_value and args.score_name is None:
        raise SystemExit("--score-string-value requires --score-name.")
    if args.append and args.overwrite:
        raise SystemExit("--append and --overwrite cannot be used together.")

    fallback_substrings: tuple[str, ...] = ()
    if args.detect_fallback_answer:
        fallback_substrings = tuple(DEFAULT_FALLBACK_SUBSTRINGS) + tuple(args.fallback_substring)
    elif args.fallback_substring:
        fallback_substrings = tuple(args.fallback_substring)

    filters = LangfuseExportFilters(
        limit=args.limit,
        max_traces_to_scan=args.max_traces_to_scan or default_max_traces_to_scan(args.limit),
        score_name=args.score_name,
        max_score=args.max_score,
        score_string_values=tuple(args.score_string_value),
        only_errors=args.only_errors,
        only_missing_answer=args.only_missing_answer,
        fallback_substrings=fallback_substrings,
        min_latency_ms=args.min_latency_ms,
        min_cost_usd=args.min_cost_usd,
        environments=tuple(args.environment),
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
        category=args.category,
    )

    client = LangfuseTraceQueryClient.from_settings(settings)
    candidates, scanned_traces = client.fetch_bad_trace_candidates(filters)
    rows = [
        langfuse_trace_to_eval_example(
            candidate,
            include_answer=args.include_answer,
            include_session_id=args.include_session_id,
            category=args.category,
        )
        for candidate in candidates
    ]
    write_summary = write_eval_examples_jsonl(
        args.output,
        rows,
        append=args.append,
        overwrite=args.overwrite,
    )

    print(
        "Exported "
        f"{write_summary.newly_added_rows} Langfuse trace review rows to {args.output} "
        f"(scanned {scanned_traces} traces, skipped {write_summary.duplicate_rows_skipped} duplicates, "
        f"total rows now {write_summary.total_rows})."
    )


if __name__ == "__main__":
    main()
