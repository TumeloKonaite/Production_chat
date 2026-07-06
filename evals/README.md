# Eval Datasets

## Canonical RAG benchmark

The canonical RAG evaluation dataset for this repository is:

```text
evals/datasets/portfolio_eval_dataset.jsonl
```

`evals/run_rag_eval.py` loads this file by default, and the rows are parsed by
`app/services/evals/rag_eval_service.py` into
`app/domain/evals/schemas.py:RagEvalDatasetExample`.

Do not create a second RAG benchmark schema alongside it. Future retrieval,
prompt, or model experiments should keep using this dataset unless the intent
is to version the benchmark explicitly.

Important maintenance rule:

```text
Do not modify the canonical benchmark dataset during an active experiment comparison unless the intent is to create a new benchmark version.
```

## Row contract

Each line is one JSON object in JSONL format.

Required fields:

```json
{
  "id": "q_001",
  "question": "What kind of AI systems has Tumelo built?",
  "expected_source_documents": ["projects.md", "profile.md", "experience.md"],
  "expected_answer_points": [
    "practical AI systems",
    "LLM-backed chat experiences"
  ],
  "category": "projects"
}
```

Optional fields:

```json
{
  "difficulty": "medium",
  "notes": "Optional maintainer guidance.",
  "expected_behavior": "fallback"
}
```

Field rules:

- `id` must be a unique non-empty string.
- `question` must be a non-empty string.
- `expected_source_documents` must be a list of source filenames.
- `expected_answer_points` must be a non-empty list of strings.
- `category` is required by the current loader and reporting pipeline.
- `difficulty`, `notes`, and `expected_behavior` are optional.
- When present, `expected_behavior` currently uses `fallback` for questions
  that should refuse, qualify, or acknowledge missing approved information.

If `expected_behavior` is omitted, the row is treated as a normal
answer-from-context question.

## Source expectations

`expected_source_documents` should use the actual filenames under:

```text
app/knowledge/source/
```

Examples:

- `experience.md`
- `projects.md`
- `skills.md`
- `education.md`
- `contact.md`

Do not introduce synthetic source IDs or a separate source manifest unless the
eval service and runners are updated in the same change.

## Fallback and unsupported questions

Use `expected_behavior: "fallback"` when the assistant should not answer with
an unqualified factual claim.

Use one of these two patterns:

1. No approved supporting source exists.

```json
{
  "question": "What is Tumelo's expected salary?",
  "expected_source_documents": [],
  "expected_answer_points": [
    "does not invent compensation details",
    "explains that salary information is not available in the approved knowledge base"
  ],
  "expected_behavior": "fallback"
}
```

2. A source document establishes the boundary.

```json
{
  "question": "What is Tumelo's email address?",
  "expected_source_documents": ["contact.md"],
  "expected_answer_points": [
    "does not invent an email address",
    "explains that the canonical contact document does not publish a public email address"
  ],
  "expected_behavior": "fallback"
}
```

Fallback rows still need explicit `expected_answer_points`. Empty answer-point
lists are not allowed in the canonical benchmark.

## Coverage

The canonical benchmark should preserve representative coverage across:

- experience
- projects
- skills
- education
- contact
- fallback / unsupported

If you need informal groupings beyond the existing `category` field, prefer
documenting the grouping here rather than adding new schema fields.

Operational validation:

- Retrieval eval runners validate expected-source coverage before scoring.
- The default minimum coverage is `0.95`, which allows at most one missing
  `expected_source_documents` row in the current 25-query benchmark.
- If coverage falls below that threshold, the runner fails fast before logging
  comparison metrics.

## Adding or changing questions

When adding a new benchmark row:

1. Reuse the existing schema.
2. Point `expected_source_documents` at real knowledge-base filenames.
3. Write answer points that are specific enough to catch regressions but stable
   enough to remain comparable across experiments.
4. Prefer updating `notes` over adding new schema keys when maintainers need
   extra judging context.

Only change existing benchmark rows when:

- the approved knowledge base changed and the benchmark must track that new
  ground truth, or
- you are intentionally creating a new benchmark version for a new comparison
  baseline.

## Other eval datasets

`evals/datasets/generation_eval_dataset.jsonl` is the fixed-context generation
comparison dataset used by `evals/run_generation_eval.py`.

Its role is model-to-model answer comparison while holding retrieval constant.
Each row includes the question plus the exact retrieved context that every
model should see for that example. Prefer this dataset when the experiment goal
is answer quality, latency, token usage, or cost comparison across providers.

`evals/datasets/model_eval_dataset.jsonl` is not the canonical RAG benchmark.
It is a separate legacy-style dataset used by `evals/run_model_eval.py`, which
still evaluates responses with the older `expected_facts` contract in
`app/services/evals/eval_service.py`.

Its role is an exact-fact smoke test for the older model comparison pipeline.
Keep it focused on concise named-entity, technology-list, or explicit-boundary
questions that the legacy scorer can grade with substring fact matching.

`evals/datasets/prompt_eval_questions.jsonl` is a separate prompt-comparison
question set used by `scripts/compare_prompts.py`. It is intentionally lighter
weight than the canonical RAG benchmark and only encodes `expected_behavior`
labels for prompt-side comparisons.

Its role is behavioral prompt evaluation rather than fact coverage. Prefer
broad introduction, consultative tone, clarification, and boundary-handling
prompts here.

## Non-overlap rules

The three active datasets should stay intentionally distinct:

- `portfolio_eval_dataset.jsonl`: canonical source-grounded RAG benchmark.
- `generation_eval_dataset.jsonl`: fixed-context generation benchmark.
- `model_eval_dataset.jsonl`: legacy exact-fact smoke test for model runs.
- `prompt_eval_questions.jsonl`: prompt behavior and response-style probes.

Do not copy the same question into multiple datasets. If a new question could
fit more than one dataset, choose the one whose scoring contract best matches
the maintenance goal.

## Langfuse review queue exports

`python -m evals.export_bad_langfuse_traces` exports failed or low-quality
production traces into a separate JSONL review queue. This is intentionally not
the canonical benchmark and does not weaken the source-coverage or
answer-points rules above.

The export rows include:

- `expected_facts: []`
- `expected_answer_points: []`
- `expected_source_documents: []`
- Langfuse metadata for trace back-reference

That keeps the exported file lightweight and safe to review first. Once a row
has a confirmed ground truth, copy or adapt it into the appropriate scored
dataset rather than editing the canonical benchmark casually.

## Retrieval Embedding Matrix

Use `scripts/run_embedding_experiment.py` to compare retrieval quality across
multiple embedding provider/model combinations against the same dataset.

Example config:

```text
evals/configs/retrieval_embedding_matrix.example.json
```

Example command:

```text
python scripts/run_embedding_experiment.py --config evals/configs/retrieval_embedding_matrix.example.json
```

Operational note:

- The runner rebuilds the indexed knowledge per embedding run before scoring.
- Each configured run still has to match the current pgvector storage
  dimension. If a run's configured `dimension` does not match the database
  vector column, the runner fails fast instead of mixing incompatible indexes.

## Retrieval Sweep Runner

Use `evals/run_retrieval_sweep.py` to compare multiple retrieval configurations
against the same dataset without rerunning the single-run CLI manually.

The single-run retrieval eval also supports JSON config files for baseline vs
reranked comparisons:

```text
configs/evals/retrieval_baseline.json
configs/evals/retrieval_reranked_llm.json
```

Example commands:

```text
uv run python evals/run_retrieval_eval.py --config configs/evals/retrieval_baseline.json
uv run python evals/run_retrieval_eval.py --config configs/evals/retrieval_reranked_llm.json
```

Example config:

```text
evals/configs/retrieval_sweep.yaml
```

Example command:

```text
python evals/run_retrieval_sweep.py --config evals/configs/retrieval_sweep.yaml
```

Operational notes:

- Each experiment in the YAML config becomes a separate MLflow run.
- The sweep reuses the shared retrieval eval runner and does not duplicate the
  retrieval metric calculation logic.
- Single-run retrieval evals can also enable reranking directly with
  `--enable-reranker --reranker-type llm --reranker-initial-top-k <N>`.
- Reranked runs log both the initial retrieval candidate order and the final
  reranked order so MRR and context-order changes are inspectable per query.
- The runner writes per-experiment artifacts plus sweep-level comparison JSON
  and CSV outputs under `evals/results/retrieval_sweeps/`.
