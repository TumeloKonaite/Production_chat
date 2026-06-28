from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean

from app.services.retrieval import RetrievedChunk

REFUSAL_MARKERS = (
    "do not have enough",
    "don't have enough",
    "not enough approved information",
    "not enough information",
    "cannot confirm",
    "can't confirm",
    "unable to confirm",
)


@dataclass(frozen=True, slots=True)
class EvalDatasetExample:
    id: str
    question: str
    category: str
    expected_facts: list[str]
    expected_behavior: str | None = None


@dataclass(frozen=True, slots=True)
class ModelEvalRecord:
    eval_id: str
    model_config_id: str
    question: str
    category: str
    expected_facts: list[str]
    expected_behavior: str | None
    answer: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    quality_score: int
    groundedness_score: float
    passed: bool
    used_fallback: bool
    retrieved_sources: list[str]


@dataclass(frozen=True, slots=True)
class ModelEvalAggregate:
    model_config_id: str
    model_provider: str
    model_name: str
    total_examples: int
    passed_examples: int
    failed_examples: int
    pass_rate: float
    average_quality_score: float
    average_groundedness_score: float
    average_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    average_cost_per_response_usd: float


def load_eval_dataset(path: Path) -> list[EvalDatasetExample]:
    examples: list[EvalDatasetExample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        examples.append(
            EvalDatasetExample(
                id=str(payload["id"]),
                question=str(payload["question"]),
                category=str(payload["category"]),
                expected_facts=[str(item) for item in payload.get("expected_facts", [])],
                expected_behavior=(
                    str(payload["expected_behavior"])
                    if payload.get("expected_behavior") is not None
                    else None
                ),
            )
        )
    return examples


def is_refusal(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def score_quality(
    answer: str,
    *,
    expected_facts: list[str],
    expected_behavior: str | None,
) -> tuple[int, bool]:
    refusal = is_refusal(answer)
    normalized = answer.casefold()

    if expected_behavior == "refuse_or_fallback":
        passed = refusal
        return (5 if refusal else 1), passed

    if not expected_facts:
        passed = not refusal
        return (4 if passed else 2), passed

    matches = sum(1 for fact in expected_facts if fact.casefold() in normalized)
    coverage = matches / len(expected_facts)
    if coverage >= 1.0:
        return 5, True
    if coverage >= 0.66:
        return 4, True
    if coverage >= 0.33:
        return 3, False
    if refusal:
        return 2, False
    return 1, False


def score_groundedness(answer: str, retrieved_chunks: list[RetrievedChunk]) -> float:
    if not retrieved_chunks:
        return 5.0 if is_refusal(answer) else 2.0

    context = " ".join(chunk.content.casefold() for chunk in retrieved_chunks)
    answer_terms = {
        token for token in answer.casefold().replace(".", " ").replace(",", " ").split() if len(token) >= 4
    }
    overlap = sum(1 for token in answer_terms if token in context)
    if overlap >= 8:
        return 5.0
    if overlap >= 5:
        return 4.0
    if overlap >= 3:
        return 3.0
    if overlap >= 1:
        return 2.0
    return 1.0


def build_aggregate(
    *,
    model_config_id: str,
    model_provider: str,
    model_name: str,
    records: list[ModelEvalRecord],
) -> ModelEvalAggregate:
    latencies = sorted(float(record.latency_ms) for record in records)
    total_examples = len(records)
    passed_examples = sum(1 for record in records if record.passed)
    failed_examples = total_examples - passed_examples
    total_input_tokens = sum(record.input_tokens or 0 for record in records)
    total_output_tokens = sum(record.output_tokens or 0 for record in records)
    total_tokens = sum(record.total_tokens or 0 for record in records)
    total_cost_usd = round(sum(record.estimated_cost_usd or 0.0 for record in records), 6)

    if not latencies:
        p95_latency_ms = 0.0
        average_latency_ms = 0.0
    else:
        index = max(0, min(len(latencies) - 1, int(round((len(latencies) - 1) * 0.95))))
        p95_latency_ms = latencies[index]
        average_latency_ms = mean(latencies)

    return ModelEvalAggregate(
        model_config_id=model_config_id,
        model_provider=model_provider,
        model_name=model_name,
        total_examples=total_examples,
        passed_examples=passed_examples,
        failed_examples=failed_examples,
        pass_rate=(passed_examples / total_examples) if total_examples else 0.0,
        average_quality_score=mean(record.quality_score for record in records) if records else 0.0,
        average_groundedness_score=(
            mean(record.groundedness_score for record in records) if records else 0.0
        ),
        average_latency_ms=average_latency_ms,
        p95_latency_ms=p95_latency_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        average_cost_per_response_usd=(
            round(total_cost_usd / total_examples, 6) if total_examples else 0.0
        ),
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_comparison_summary(aggregates: list[ModelEvalAggregate]) -> str:
    lines = [
        "Model Comparison Summary",
        "",
        "Model                 Avg Quality   Pass Rate   Avg Latency   P95 Latency   Total Cost",
    ]
    for aggregate in aggregates:
        lines.append(
            f"{aggregate.model_config_id:<21} "
            f"{aggregate.average_quality_score:>11.2f}   "
            f"{aggregate.pass_rate:>8.0%}   "
            f"{aggregate.average_latency_ms:>10.0f}ms   "
            f"{aggregate.p95_latency_ms:>10.0f}ms   "
            f"${aggregate.total_cost_usd:.6f}"
        )
    return "\n".join(lines) + "\n"


def records_as_json(records: list[ModelEvalRecord]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]
