# Evaluation

The evaluation system separates retrieval quality, generation quality, and end-to-end RAG behavior so changes can be attributed to the correct subsystem. Evaluation calls can incur model and judge costs.

## Evaluation layers

| Layer | Default dataset | What changes | Main outputs |
|---|---|---|---|
| Retrieval | `evals/datasets/portfolio_eval_dataset.jsonl` | embeddings, strategy, top-k, rewriting, reranking | Recall@k, precision@k, MRR, hit rate, latency |
| Fixed-context generation | `evals/datasets/generation_eval_dataset.jsonl` | answer model, prompt, temperature | quality/pass rate, groundedness, judge scores, latency, tokens, cost |
| End-to-end RAG | `evals/datasets/portfolio_eval_dataset.jsonl` | retrieval plus answer model/prompt | retrieval metrics plus context relevance, faithfulness, answer relevance, latency, cost |

Routing behavior is tested in `tests/test_chat_routing.py`; there is currently no standalone route-accuracy runner. Do not report route accuracy as an evaluation artifact unless a runner is added.

## Prerequisites

- Install locked dependencies with `uv sync --locked`.
- Apply migrations and ingest knowledge for retrieval and RAG evaluation.
- Configure the answer model key for generation/RAG.
- Configure the judge model's provider key when judge scoring is used.
- Keep `ENABLE_MLFLOW_TRACKING=false` for local artifacts only, or configure tracking as described below.

## Actual dataset schemas

### Retrieval and RAG dataset

Each JSONL row uses this contract:

```json
{
  "id": "q_002",
  "question": "Does Tumelo have experience with FastAPI?",
  "expected_source_documents": ["projects.md", "skills.md"],
  "expected_answer_points": [
    "FastAPI backend",
    "backend service design",
    "API schema definition",
    "service layers"
  ],
  "category": "skills",
  "difficulty": "easy",
  "notes": null,
  "expected_behavior": null
}
```

Required fields are `id`, `question`, `expected_source_documents`, `expected_answer_points`, and `category`. Optional fields are `difficulty`, `notes`, and `expected_behavior`. The current categories are `profile`, `projects`, `skills`, `experience`, `education`, `contact`, `chatbot`, and `unsupported`.

Retrieval evaluation validates expected-source coverage before running. Source labels must match stored source names such as `projects.md`.

### Fixed-context generation dataset

```json
{
  "id": "generation_projects_001",
  "question": "What does the portfolio chatbot project demonstrate?",
  "category": "projects",
  "context": [
    {
      "source": "projects.md",
      "section": "Portfolio Chatbot",
      "content": "The portfolio chatbot is a FastAPI backend..."
    }
  ],
  "expected_facts": [
    "FastAPI backend",
    "retrieval-grounded",
    "curated knowledge base",
    "service boundaries"
  ],
  "expected_answer_points": [
    "FastAPI backend",
    "retrieval-grounded or RAG behavior"
  ],
  "expected_behavior": null
}
```

The loader also accepts optional `expected_answer`. Fixed context intentionally removes retrieval variance.

See [evals/README.md](../evals/README.md) for dataset boundaries, source expectations, and review rules.

## Run a retrieval baseline

```bash
uv run python -m evals.runners.run_retrieval_eval \
  --config evals/configs/retrieval_baseline.json
```

Override final top-k or dataset when needed:

```bash
uv run python -m evals.runners.run_retrieval_eval \
  --config evals/configs/retrieval_reranked_llm.json \
  --k 5 \
  --run-name reranked-baseline
```

The runner writes JSON/CSV-style artifacts to `evals/results/` by default and includes per-query results, validation summary, configuration, rewrite metadata, latency, and aggregate ranking metrics.

## Run fixed-context generation evaluation

```bash
uv run python -m evals.runners.run_generation_eval
```

Pin model, judge, and prompt for a reproducible comparison:

```bash
uv run python -m evals.runners.run_generation_eval \
  --model openai:gpt-4.1-mini \
  --judge-model openai:gpt-4.1-mini \
  --prompt-version v1_professional \
  --dataset-version generation-v1
```

Artifacts are written to `evals/results/`. The runner logs the dataset, prompt template, provider/model, temperature, token usage, latency, cost estimate, and judge configuration.

## Run end-to-end RAG evaluation

```bash
uv run python -m evals.runners.run_rag_eval \
  --model openai:gpt-4.1-mini \
  --judge-model openai:gpt-4.1-mini \
  --prompt-version v1_professional \
  --run-name rag-baseline \
  --no-db
```

Remove `--no-db` to persist `rag_eval_runs` and `rag_eval_results` in PostgreSQL. Local result artifacts are written either way.

## Run a matrix

Preview a suite without model calls:

```bash
uv run python -m evals.runners.run_experiment_matrix \
  --suite retrieval_smoke \
  --dry-run
```

Execute it:

```bash
uv run python -m evals.runners.run_experiment_matrix \
  --suite retrieval_smoke
```

Available checked-in suites are `retrieval_smoke`, `generation_smoke`, `rag_medium`, and `rag_full`. `rag_full` requires `--confirm-full-run`.

Matrix output is grouped under:

```text
evals/outputs/experiments/<timestamp>_<suite>/
├── runs/
├── resolved_suite.json
├── <mode>_summary.json
├── <mode>_summary.csv
├── <mode>_ranking.md
├── failures.json
└── manifest.json
```

Each child run is named from the suite and resolved run ID. A suite-level tracking run logs summary metrics and artifacts when tracking is enabled.

## Metric meanings

- **Recall@k:** fraction of expected source documents appearing in the first `k` retrieved sources.
- **Precision@k:** fraction of the first `k` retrieved sources that are expected.
- **MRR:** reciprocal rank of the first expected source, averaged over labeled queries.
- **Hit rate:** fraction of labeled queries with at least one expected source in the result set.
- **Context relevance:** judge assessment that retrieved/provided context addresses the question (0–2 in judge-based flows).
- **Faithfulness/groundedness:** whether the answer is supported by supplied context. Judge faithfulness is 0–2; heuristic groundedness is a separate normalized score.
- **Answer relevance/quality:** whether the answer addresses the question and expected points. Judge scores are 0–2; heuristic quality/pass metrics are separate.
- **Latency:** measured retrieval, rewrite, generation, or end-to-end elapsed time; aggregate output includes mean and usually p95.
- **Token usage:** provider-reported input, output, and total tokens when available.
- **Estimated cost:** calculated only when model pricing is known or manual per-million-token prices are configured; a missing estimate is not zero spend.

NDCG is implemented for persisted RAG summaries, while the current standalone retrieval runner's primary summary reports recall, precision, MRR, and hit rate.

## MLflow and DagsHub

Local MLflow example:

```env
ENABLE_MLFLOW_TRACKING=true
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

Remote DagsHub example:

```env
ENABLE_MLFLOW_TRACKING=true
ENABLE_DAGSHUB_TRACKING=true
DAGSHUB_REPO_OWNER=your-owner
DAGSHUB_REPO_NAME=your-repository
DAGSHUB_TOKEN=your-token
```

DagsHub is initialized with its dedicated settings and `dagshub.init(..., mlflow=True)`; `MLFLOW_TRACKING_URI` is not the DagsHub selection mechanism. MLflow remains the logging API, so both enable flags are required.

Tracked parameters include workflow/family, run name, dataset path/version, Git SHA, retrieval configuration, embedding provider/model/dimension, query rewrite/reranker settings, prompt template, model/provider/base URL, temperature, top-k, and judge. Metrics and local output files are logged as run artifacts.

Compare runs in the MLflow/DagsHub UI using a common experiment name and stable parameter dimensions. Compare matrix runs first with the generated ranking and failure artifacts; do not compare rows that used different datasets without labeling the dataset version.

## Add or change examples

1. Choose the dataset that isolates the intended behavior.
2. Add one valid JSON object per line with a unique stable ID.
3. Use approved source names and facts only.
4. Include unsupported/fallback cases in the RAG dataset when routing or refusal behavior changes.
5. Run dataset tests and a smoke evaluation.

```bash
uv run python -m pytest tests/evals/test_portfolio_eval_dataset.py tests/evals/test_eval_dataset_boundaries.py
```

Do not copy production text into a dataset unless consent/privacy requirements are satisfied. Feedback and Langfuse export workflows are documented in [evals/README.md](../evals/README.md) and [Langfuse trace export](evals/langfuse_trace_export.md).

## Evaluation HTTP API

The API exposes protected background endpoints under `/api/evals` for retrieval, generation, RAG, matrix, run listing/status, summaries, and failures. Set `EVAL_ADMIN_TOKEN` and send it as `X-Eval-Admin-Token`. For local development and CI, command-line runners are easier to reproduce and inspect.
