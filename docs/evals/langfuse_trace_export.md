# Langfuse Trace Export

## Purpose

Use `python -m evals.export_bad_langfuse_traces` to turn weak production
Langfuse traces into a review queue that can later be promoted into the repo's
scored eval datasets.

This workflow is intentionally small:

```text
bad production answer
-> find the trace in Langfuse
-> export it into JSONL
-> fill expected facts and expected sources
-> move the reviewed case into the right eval dataset
-> rerun evals
```

The exporter does not auto-approve ground truth. It creates review rows with
empty expected fields by default.

## Required environment

Set Langfuse credentials in `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=production
LANGFUSE_EXPORT_DEFAULT_LIMIT=100
```

`LANGFUSE_EXPORT_DEFAULT_LIMIT` is only the CLI default. You can still override
it per command with `--limit`.

## What counts as a bad trace

The exporter currently supports these selectors:

- score-based failures with `--score-name` and `--max-score`
- categorical or boolean score matches with `--score-string-value`
- error traces with `--only-errors`
- missing answers with `--only-missing-answer`
- fallback-style answers with `--detect-fallback-answer`
- slow traces with `--min-latency-ms`
- expensive traces with `--min-cost-usd`

You can combine selectors with:

- `--from-date`
- `--to-date`
- `--environment`
- `--limit`
- `--max-traces-to-scan`

Date behavior:

- `--from-date 2026-07-01` means `2026-07-01T00:00:00Z`
- `--to-date 2026-07-06` is inclusive and becomes `2026-07-07T00:00:00Z`

## Commands

Low-scored traces:

```bash
python -m evals.export_bad_langfuse_traces \
  --output evals/datasets/production_failures_review.jsonl \
  --score-name answer_quality \
  --max-score 0.6 \
  --from-date 2026-07-01 \
  --to-date 2026-07-06 \
  --limit 100 \
  --overwrite
```

Error traces:

```bash
python -m evals.export_bad_langfuse_traces \
  --output evals/datasets/langfuse_errors_review.jsonl \
  --only-errors \
  --limit 50 \
  --overwrite
```

Negative feedback by numeric score:

```bash
python -m evals.export_bad_langfuse_traces \
  --output evals/datasets/langfuse_negative_feedback_review.jsonl \
  --score-name user_feedback \
  --max-score 0 \
  --append
```

Fallback-style answers:

```bash
python -m evals.export_bad_langfuse_traces \
  --output evals/datasets/langfuse_fallbacks_review.jsonl \
  --detect-fallback-answer \
  --limit 25 \
  --overwrite
```

If the output file already exists, choose one:

- `--append` to merge new rows and deduplicate by `id`
- `--overwrite` to replace the file

## Output contract

Each row is JSONL and includes a superset of the repo's existing eval fields:

```json
{
  "id": "langfuse_abc123",
  "question": "Does Tumelo have experience with production AI systems?",
  "expected_facts": [],
  "expected_answer_points": [],
  "expected_source_documents": [],
  "category": "production_failure",
  "notes": "Exported from Langfuse for review...",
  "metadata": {
    "source": "langfuse",
    "trace_id": "abc123",
    "created_at": "2026-07-06T12:34:56Z",
    "failure_reason": "low_score",
    "score_name": "answer_quality",
    "score_value": 0.4,
    "model": "openai/gpt-4o-mini",
    "provider": "openrouter",
    "latency_ms": 2450,
    "estimated_cost_usd": 0.0012
  }
}
```

Optional flags:

- `--include-answer` adds `observed_answer`
- `--include-session-id` adds the raw `session_id` to metadata

Sensitive defaults:

- raw `user_id` is omitted
- raw `session_id` is omitted unless explicitly requested
- only the latest question is exported by default
- retrieved context stays reduced to source metadata, not full documents

## Review workflow

After export:

1. Inspect the JSONL rows and confirm the failure is worth keeping.
2. Fill `expected_facts`, `expected_answer_points`, and `expected_source_documents`.
3. Move or adapt the reviewed row into the appropriate scored dataset:
   `model_eval_dataset.jsonl`, `generation_eval_dataset.jsonl`, or
   `portfolio_eval_dataset.jsonl`.
4. Run the relevant eval command again.

Examples:

```bash
python evals/run_model_eval.py \
  --models openai:gpt-4.1-mini \
  --prompt-version v1_professional \
  --dataset evals/datasets/model_eval_dataset.jsonl
```

```bash
uv run python evals/run_generation_eval.py \
  --dataset evals/datasets/generation_eval_dataset.jsonl \
  --prompt-version v1_professional
```

```bash
python evals/run_rag_eval.py \
  --model openai:gpt-4.1-mini \
  --prompt-version v1_professional
```
