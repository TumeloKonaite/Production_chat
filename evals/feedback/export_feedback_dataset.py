from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.repositories.db.session import get_session_factory
from evals.feedback.feedback_exporter import (
    ProductionFeedbackExportDisabledError,
    export_feedback_dataset,
    parse_feedback_filters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export production feedback and failure traces into a shared feedback JSONL "
            "dataset that can be reused by retrieval and generation eval runners."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Inclusive ISO date or datetime lower bound.",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Inclusive ISO date or datetime upper bound.",
    )
    parser.add_argument(
        "--rating",
        default=None,
        help="Optional feedback rating filter, for example negative or positive.",
    )
    parser.add_argument(
        "--reason",
        default=None,
        help="Optional feedback reason filter, for example incorrect_answer.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name filter.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Optional prompt version filter.",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Optional environment filter from trace metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of exported rows.",
    )
    parser.add_argument(
        "--allow-raw-text",
        action="store_true",
        help="Export raw production question, answer, and context text when configuration permits it.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing dataset and deduplicate by id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output dataset if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    filters = parse_feedback_filters(args)

    session_factory = get_session_factory()
    with session_factory() as session:
        try:
            result = export_feedback_dataset(
                session=session,
                settings=settings,
                filters=filters,
                output_path=args.output,
                allow_raw_text=args.allow_raw_text,
                append=args.append,
                overwrite=args.overwrite,
            )
        except (ProductionFeedbackExportDisabledError, FileExistsError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    print(
        "Exported "
        f"{result.written_rows} feedback rows to {args.output} "
        f"(skipped {result.duplicate_rows_skipped} duplicates)."
    )


if __name__ == "__main__":
    main()
