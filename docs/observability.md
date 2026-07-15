# Observability and experiment tracking

Operational observability explains individual application requests. Experiment tracking compares deliberate evaluation runs. They share some metadata, but they are separate systems.

## Langfuse operational tracing

Langfuse is disabled by default. Enable it with:

```env
ENABLE_LANGFUSE_OBSERVABILITY=true
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=local
LANGFUSE_SAMPLE_RATE=1.0
```

Both keys are required when enabled. Missing keys cause configuration startup failure. If observability is disabled, a no-op tracer is used and application behavior continues normally. Initialization failure while enabled is treated as a configuration error rather than silently disabling requested tracing.

### Trace hierarchy

```text
chat_request (root observation)
├── retrieval (retriever observation, portfolio route only)
└── llm_call (generation observation, portfolio route only)
```

The root records the question, conversation/session identifiers, endpoint/channel, provider/model, final answer, latency, tokens, estimated cost, environment, release, and route/cache metadata. Retrieval records original/rewritten query, strategy, top-k, embedding/vector settings, reranker settings, source/chunk IDs, scores, and content previews capped at 240 characters. Generation records model parameters, latency, token usage, cost, and errors.

`LANGFUSE_SAMPLE_RATE` controls the sampled fraction (`0`–`1`). `LANGFUSE_RELEASE` tags a deployment. Flush is called after request handling so buffered events are sent.

### Privacy

Langfuse receives raw user questions and final answers, plus retrieval content previews. Treat its project as sensitive application data:

- do not ingest secrets or private documents;
- minimize user identifiers;
- choose retention and access controls intentionally;
- review provider region and contractual requirements;
- use sampling to reduce exposure, not as a substitute for redaction;
- rotate keys after exposure.

## Internal trace storage

Internal traces are independent of Langfuse and live in PostgreSQL:

- `chat_traces` stores request/response text, status/error, conversation/request/session IDs, provider/model, prompt/retrieval/embedding metadata, usage, cost, latency, and optional Langfuse provider/external trace ID.
- `chat_trace_steps` stores ordered request, retrieval, prompt, model, response, and error steps with input/output JSON, timing, status, and errors.

Internal storage provides application-owned audit/debug data even when Langfuse is disabled. When Langfuse is active, `observability_provider` and `external_trace_id` link the records; one does not replace the other.

Feedback is stored separately in `message_feedback` and linked to assistant messages. Export tools can turn reviewed negative feedback into evaluation candidates.

The repository does not implement automated trace retention. Operators must define database retention, deletion, access, and backup policies appropriate to stored raw text. See [internal trace schema](observability/internal_trace_schema.md).

## Logging

The application uses Python's standard `logging` package. Uvicorn controls local log level and formatting:

```bash
uv run uvicorn main:app --reload --log-level debug
```

Startup logs report environment, vector-store provider, and whether optional integrations are enabled/configured. Failures log stack traces in server logs while API exception handlers return generic messages for provider and persistence errors.

There is no `LOG_LEVEL` application setting and no repository-wide structured-logging formatter. Do not document either as implemented. Correlation is available through conversation, message, internal trace, request, session, and optional external trace IDs, but the standard log formatter does not automatically add every ID.

Do not log API keys, authorization headers, database URLs containing passwords, raw provider responses, or secret-store payloads. Configuration code reports booleans rather than secret values at startup.

## MLflow experiment tracking

MLflow is disabled by default. Store runs locally:

```env
ENABLE_MLFLOW_TRACKING=true
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

Or point to a tracking server and optionally set `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD`.

Evaluation runners name child runs explicitly (or derive a stable label) and log:

- dataset name/path/version and Git SHA;
- workflow and experiment family;
- embedding, vector-store, retrieval, rewrite, reranker, chunk, model, prompt, and judge parameters;
- retrieval/generation/RAG metrics;
- latency, tokens, and cost estimates;
- result, summary, prompt, ranking, failure, and manifest artifacts as applicable.

If MLflow is disabled, artifacts are still written to `evals/results/` or `evals/outputs/experiments/`.

## DagsHub tracking

DagsHub wraps the same MLflow API:

```env
ENABLE_MLFLOW_TRACKING=true
ENABLE_DAGSHUB_TRACKING=true
DAGSHUB_REPO_OWNER=your-owner
DAGSHUB_REPO_NAME=your-repository
DAGSHUB_TOKEN=your-token
```

The implementation calls `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)`. It copies `DAGSHUB_TOKEN` to `DAGSHUB_USER_TOKEN` only when that environment variable is not already set. It does not use `MLFLOW_TRACKING_URI` to select DagsHub.

If DagsHub is enabled without MLflow, tracking is disabled with a warning. Missing owner/name or failed initialization raises a tracking setup error when a runner attempts to create the backend.

## Diagnose missing traces or runs

1. Confirm the correct enable flag in the process/container environment.
2. Restart the API after `.env` changes because settings/tracer instances are cached.
3. Confirm keys, base URL/region, sample rate, and outbound connectivity for Langfuse.
4. Query internal trace rows to separate application persistence from Langfuse export.
5. Confirm both enable flags and owner/repository for DagsHub.
6. Inspect local artifacts even when remote tracking failed.

See [troubleshooting](troubleshooting.md) and [evaluation](evaluation.md).
