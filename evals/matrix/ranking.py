from __future__ import annotations

from typing import Any


RETRIEVAL_SCORE_FORMULA = "0.5 * recall_at_k + 0.3 * mrr + 0.2 * precision_at_k"
GENERATION_SCORE_FORMULA = (
    "0.55 * normalized_quality + 0.35 * normalized_groundedness "
    "- 0.05 * normalized_latency - 0.05 * normalized_cost"
)
RAG_SCORE_FORMULA = (
    "0.35 * normalized_answer_relevance + 0.25 * normalized_faithfulness "
    "+ 0.20 * normalized_context_relevance + 0.10 * recall_at_k "
    "- 0.05 * normalized_latency - 0.05 * normalized_cost"
)


def rank_mode_rows(
    *,
    mode: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    if not rows:
        return [], _build_ranking_metadata(mode=mode)

    latency_norms = _build_normalized_penalties(rows, "latency_ms_avg")
    cost_norms = _build_normalized_penalties(rows, "estimated_total_cost_usd")

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        ranked_row = dict(row)
        if mode == "retrieval":
            ranked_row["score"] = round(
                0.5 * _float_value(row.get("recall_at_k"))
                + 0.3 * _float_value(row.get("mrr"))
                + 0.2 * _float_value(row.get("precision_at_k")),
                6,
            )
        elif mode == "generation":
            ranked_row["score"] = round(
                0.55 * _bounded_ratio(row.get("average_quality_score"), upper_bound=5.0)
                + 0.35 * _bounded_ratio(row.get("average_groundedness_score"), upper_bound=5.0)
                - 0.05 * latency_norms.get(_row_key(row), 0.0)
                - 0.05 * cost_norms.get(_row_key(row), 0.0),
                6,
            )
        else:
            ranked_row["score"] = round(
                0.35 * _bounded_ratio(row.get("avg_answer_relevance"), upper_bound=2.0)
                + 0.25 * _bounded_ratio(row.get("avg_faithfulness"), upper_bound=2.0)
                + 0.20 * _bounded_ratio(row.get("avg_context_relevance"), upper_bound=2.0)
                + 0.10 * _float_value(row.get("avg_recall_at_k"))
                - 0.05 * latency_norms.get(_row_key(row), 0.0)
                - 0.05 * cost_norms.get(_row_key(row), 0.0),
                6,
            )
        scored_rows.append(ranked_row)

    sorted_rows = sorted(
        scored_rows,
        key=lambda row: _row_sort_key(mode=mode, row=row),
        reverse=True,
    )
    for index, row in enumerate(sorted_rows, start=1):
        row["rank"] = index
        row["is_best"] = index == 1
    return sorted_rows, _build_ranking_metadata(mode=mode)


def _row_sort_key(*, mode: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if mode == "retrieval":
        return (
            _float_value(row.get("score")),
            _float_value(row.get("recall_at_k")),
            _float_value(row.get("mrr")),
            _float_value(row.get("precision_at_k")),
            str(row.get("run_id", "")),
        )
    if mode == "generation":
        return (
            _float_value(row.get("score")),
            _float_value(row.get("average_quality_score")),
            _float_value(row.get("average_groundedness_score")),
            _float_value(row.get("pass_rate")),
            -_float_value(row.get("latency_ms_avg")),
            str(row.get("run_id", "")),
        )
    return (
        _float_value(row.get("score")),
        _float_value(row.get("avg_answer_relevance")),
        _float_value(row.get("avg_faithfulness")),
        _float_value(row.get("avg_recall_at_k")),
        _float_value(row.get("avg_precision_at_k")),
        str(row.get("run_id", "")),
    )


def _build_ranking_metadata(*, mode: str) -> dict[str, object]:
    if mode == "retrieval":
        return {
            "primary_metric": "score",
            "score_formula": RETRIEVAL_SCORE_FORMULA,
            "tiebreak_metrics": ["recall_at_k", "mrr", "precision_at_k"],
        }
    if mode == "generation":
        return {
            "primary_metric": "score",
            "score_formula": GENERATION_SCORE_FORMULA,
            "tiebreak_metrics": [
                "average_quality_score",
                "average_groundedness_score",
                "pass_rate",
                "latency_ms_avg",
            ],
        }
    return {
        "primary_metric": "score",
        "score_formula": RAG_SCORE_FORMULA,
        "tiebreak_metrics": [
            "avg_answer_relevance",
            "avg_faithfulness",
            "avg_recall_at_k",
            "avg_precision_at_k",
        ],
    }


def _build_normalized_penalties(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    numeric_values = [
        _float_value(row.get(key))
        for row in rows
        if isinstance(row.get(key), int | float)
    ]
    if not numeric_values:
        return {}

    minimum = min(numeric_values)
    maximum = max(numeric_values)
    if maximum == minimum:
        return {_row_key(row): 0.0 for row in rows}

    penalties: dict[str, float] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, int | float):
            penalties[_row_key(row)] = 0.0
            continue
        penalties[_row_key(row)] = (float(value) - minimum) / (maximum - minimum)
    return penalties


def _bounded_ratio(value: object, *, upper_bound: float) -> float:
    if not isinstance(value, int | float):
        return 0.0
    return max(0.0, min(float(value) / upper_bound, 1.0))


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("run_id", ""))
