from __future__ import annotations

import json
from pathlib import Path

from app.domain.evals import RagEvalDatasetExample
from app.services.evals.rag_eval_service import RagEvalService

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "evals" / "datasets" / "portfolio_eval_dataset.jsonl"
SOURCE_DIR = ROOT_DIR / "app" / "knowledge" / "source"
ALLOWED_EXPECTED_BEHAVIORS = {"fallback"}


def _load_raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        DATASET_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict), f"line {line_number} must decode to an object"
        rows.append(payload)
    return rows


def test_portfolio_eval_dataset_loads_with_rag_eval_service() -> None:
    service = RagEvalService(
        prompt_loader=None,  # type: ignore[arg-type]
        llm_service=None,  # type: ignore[arg-type]
        retrieval_service=None,  # type: ignore[arg-type]
        judge_client=None,  # type: ignore[arg-type]
    )

    examples = service.load_dataset(DATASET_PATH)

    assert examples
    assert all(isinstance(example, RagEvalDatasetExample) for example in examples)


def test_portfolio_eval_dataset_rows_follow_the_documented_contract() -> None:
    rows = _load_raw_rows()
    source_documents = {path.name for path in SOURCE_DIR.iterdir() if path.is_file()}
    seen_ids: set[str] = set()

    assert rows

    for row in rows:
        row_id = row.get("id")
        question = row.get("question")
        category = row.get("category")
        expected_source_documents = row.get("expected_source_documents")
        expected_answer_points = row.get("expected_answer_points")
        expected_behavior = row.get("expected_behavior")

        assert isinstance(row_id, str) and row_id.strip()
        assert row_id not in seen_ids
        seen_ids.add(row_id)

        assert isinstance(question, str) and question.strip()
        assert isinstance(category, str) and category.strip()

        assert isinstance(expected_source_documents, list)
        assert all(
            isinstance(source_name, str) and source_name.strip()
            for source_name in expected_source_documents
        )
        assert set(expected_source_documents).issubset(source_documents)

        assert isinstance(expected_answer_points, list)
        assert expected_answer_points
        assert all(
            isinstance(answer_point, str) and answer_point.strip()
            for answer_point in expected_answer_points
        )

        if expected_behavior is not None:
            assert isinstance(expected_behavior, str) and expected_behavior.strip()
            assert expected_behavior in ALLOWED_EXPECTED_BEHAVIORS

        if category == "unsupported":
            assert expected_behavior == "fallback"
            assert expected_source_documents == []
