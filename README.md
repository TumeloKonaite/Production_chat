# Production Chatbot

Simple FastAPI backend for a personal website chatbot. The frontend sends a message to `POST /chat`, the backend calls the configured LLM with a server-side API key, stores production chat metadata in PostgreSQL, and can run evaluation workflows with local or remote MLflow-backed tracking.

## Project structure

```text
app/
  main.py
  config.py
  api/
    chat/
      __init__.py
      routes.py
      schemas.py
    tavus/
      __init__.py
      routes.py
      schemas.py
    dependencies/
      chat_dependencies.py
      common_dependencies.py
      tavus_dependencies.py
  Dockerfile
  infrastructure/
    llm/
      base.py
      model_config.py
      model_registry.py
      openai_client.py
    prompts/
      prompt_loader.py
      templates/
        v1_professional.md
        v2_warm_conversational.md
    tavus/
      client.py
    tracking/
      experiment_tracker.py
      mlflow_client.py
      setup.py
  repositories/
    chat_repository.py
    db/
      base.py
      models.py
      session.py
  services/
    chat/
      errors.py
      prompting.py
      service.py
    tavus/
      errors.py
      service.py
    evals/
      eval_service.py
      model_experiment_service.py
    llm/
      errors.py
      service.py
evals/
  README.md
  configs/
    retrieval_sweep.yaml
  datasets/
    model_eval_dataset.jsonl
    portfolio_eval_dataset.jsonl
    prompt_eval_questions.jsonl
  results/
  retrieval_eval_runner.py
  run_model_eval.py
  run_rag_eval.py
  run_retrieval_eval.py
  run_retrieval_sweep.py
alembic/
  versions/
frontend/
  src/
    App.tsx
    App.css
    main.tsx
    vite-env.d.ts
  .env.example
  index.html
  package.json
  README.md
  tsconfig.json
  tsconfig.node.json
  vite.config.ts
tests/
  test_chat_api.py
```

## Environment variables

Create `.env` from `.env.example` and set:

```env
POSTGRES_DB=production_chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot
DATABASE_DIRECT_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=
LLM_PROMPT_COST_PER_1M_TOKENS=
LLM_COMPLETION_COST_PER_1M_TOKENS=
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
TAVUS_API_KEY=
TAVUS_BASE_URL=https://tavusapi.com
TAVUS_FACE_ID=
TAVUS_PAL_ID=
PUBLIC_BACKEND_URL=
TAVUS_TOOL_SECRET=
INGESTION_API_SECRET=
DEFAULT_MODEL_CONFIG_ID=openai:gpt-4.1-mini
MODEL_CONFIGS_JSON=
ENABLE_LANGFUSE_OBSERVABILITY=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=local
LANGFUSE_RELEASE=
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_EXPORT_DEFAULT_LIMIT=100
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
KNOWLEDGE_EMBEDDING_DIMENSION=384
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_PROMPT_VERSION=v1_professional
CONVERSATION_HISTORY_LIMIT=10
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SIMILARITY=0.55
DEFAULT_RETRIEVAL_CONFIG=default
ENABLE_RERANKING=false
RERANKER_TYPE=none
RERANKER_MODEL=openai:gpt-4.1-mini
RERANKER_INITIAL_TOP_K=20
RERANKER_FINAL_TOP_K=5
ENABLE_REDIS=false
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
EXACT_CACHE_ENABLED=true
EXACT_CACHE_TTL_SECONDS=300
REQUEST_LOCK_ENABLED=true
REQUEST_LOCK_TTL_SECONDS=30
RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION=personal_knowledge_base
ENABLE_MLFLOW_TRACKING=false
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
ENABLE_DAGSHUB_TRACKING=false
DAGSHUB_REPO_OWNER=
DAGSHUB_REPO_NAME=
DAGSHUB_TOKEN=
```

`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, and `LLM_API_KEY` are the preferred generic runtime settings. `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, and `OPENROUTER_BASE_URL` remain supported so one backend can still keep both provider paths configured at the same time. `OPENAI_MODEL` is still accepted as a legacy fallback.

For production Postgres, the app always uses `DATABASE_URL` for runtime traffic. Alembic and other maintenance jobs prefer `DATABASE_DIRECT_URL` and fall back to `DATABASE_URL` when no direct connection is configured.

Uploaded knowledge files use a separate storage provider setting from the vector store:

```env
VECTOR_STORE_PROVIDER=supabase_pgvector
STORAGE_PROVIDER=supabase
```

`VECTOR_STORE_PROVIDER` controls where chunks and embeddings are persisted. `STORAGE_PROVIDER` controls where uploaded source files are stored and later reloaded for ingestion.

## Local setup

Install dependencies:

```bash
uv sync
```

Run the database migration:

```bash
alembic upgrade head
```

The backend exposes:

- `GET /health` for lightweight liveness
- `GET /ready` for database-backed readiness

See `docs/deployment/supabase.md` for Supabase-specific production setup, URL formats, SSL requirements, and migration commands.
See `docs/deployment/modal.md` for Modal deployment, secret setup, and production smoke tests.

If you want the local Postgres and Redis instances from `docker-compose.yml`, start them first so the database is listening on `127.0.0.1:5434` and Redis is listening on `127.0.0.1:6379`:

```bash
docker compose up -d db redis
```

If you also want pgAdmin for local database inspection, start both services:

```bash
docker compose up -d db redis pgadmin
```

pgAdmin is then available at `http://127.0.0.1:5051` with:

```text
Email: admin@local.dev
Password: admin
```

To register the Compose database inside pgAdmin, add a new server with:

```text
Host: db
Port: 5432
Database: production_chatbot
Username: postgres
Password: postgres
```

Use `db` as the host from pgAdmin because both containers run on the same Compose network. From the host machine, Postgres remains available on `127.0.0.1:5434`.

If `5051` is already in use on your machine, override the host port before starting Compose:

```env
PGADMIN_PORT=5052
```

If you are using an existing local Postgres server instead of the Compose database, create the target database once before running Alembic:

```sql
CREATE DATABASE production_chatbot;
```

Run the API:

```bash
uvicorn main:app --reload
```

### Upstash Redis

Set:

```env
ENABLE_REDIS=true
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...
RATE_LIMIT_ENABLED=true
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
EXACT_CACHE_ENABLED=true
EXACT_CACHE_TTL_SECONDS=300
REQUEST_LOCK_ENABLED=true
REQUEST_LOCK_TTL_SECONDS=30
RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION=personal_knowledge_base
```

- Redis stays disabled by default for local development.
- When enabled, the public chat endpoint uses Upstash Redis for fixed-window rate limiting, exact response caching, and short-lived duplicate request locks.
- Cache hits skip retrieval and LLM generation.
- The cache is scoped by knowledge-base version, prompt version, model, and retrieval configuration to avoid unsafe reuse.
- Redis failures degrade gracefully to the normal chat path, while `/ready` still reports Redis readiness separately.

## Docker Compose

```bash
docker compose up --build
```

`docker compose up -d` starts the full local stack, including `db`, `redis`, and `pgadmin`. If you only want the local infrastructure services, use `docker compose up -d db redis pgadmin`. The bundled Redis service uses Redis Stack so semantic response caching can use RediSearch. The pgAdmin data directory is persisted in the `pgadmin_data` volume so saved server registrations survive container restarts. Redis persistence is stored in the `redis_data` volume.

## Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects has Tumelo worked on?", "prompt_version": "v2_warm_conversational"}'
```

Example response:

```json
{
  "conversation_id": "c9ef4d5d-1e4b-4f78-8d3c-4e3f9f0a7a2d",
  "message": "Tumelo has worked on AI chatbots, RAG systems, FastAPI backends, automation workflows, and production-ready AI applications.",
  "model": "gpt-4.1-mini",
  "model_provider": "openai",
  "model_name": "gpt-4.1-mini",
  "model_config_id": "openai:gpt-4.1-mini",
  "prompt_version": "v2_warm_conversational",
  "retrieval_config": "default",
  "latency_ms": 842,
  "token_usage": {
    "input_tokens": 1200,
    "output_tokens": 180,
    "total_tokens": 1380
  },
  "estimated_cost_usd": 0.000768
}
```

## Tavus integration

Tavus is treated as the video interface layer. The backend remains responsible for retrieval, prompt selection, model selection, and response logging.

### Start a Tavus conversation

```bash
curl -X POST http://localhost:8000/api/tavus/conversations \
  -H "Content-Type: application/json" \
  -d '{"visitor_name":"Website visitor"}'
```

Example response:

```json
{
  "conversation_id": "tavus-conversation-id",
  "conversation_url": "https://..."
}
```

### Tavus tool endpoint

Configure Tavus to call:

```text
POST https://your-backend.com/api/tavus/tools/ask-tumelo
```

With header:

```text
x-tavus-tool-secret: <TAVUS_TOOL_SECRET>
```

Tool name:

```text
ask_tumelo_backend
```

Tool purpose:

```text
Ask Tumelo's portfolio chatbot backend for grounded answers about Tumelo's experience, projects, skills, education, certifications, and contact details.
```

Expected Tavus behavior:

```text
Use this tool for factual questions about Tumelo.
Do not invent facts.
Speak the backend response as the final answer.
```

### End a Tavus conversation

```bash
curl -X POST http://localhost:8000/api/tavus/conversations/end \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"tavus-conversation-id"}'
```

## Knowledge ingestion

The markdown knowledge source of truth lives under `app/knowledge/source/`.

Local CLI ingestion still runs directly in-process for development:

```bash
uv run python .\scripts\ingest_knowledge.py
```

The protected ingestion trigger now creates a `knowledge_ingestion_jobs` row and dispatches the actual work through the configured backend:

```env
INGESTION_BACKEND=local
MODAL_INGESTION_APP_NAME=production-chatbot-api
MODAL_INGESTION_FUNCTION_NAME=run_ingestion_job
```

Backend behavior:

- `local`: create the job and execute it in a local background thread so the HTTP response returns immediately.
- `modal`: create the job and dispatch the worker to Modal with `modal.Function.from_name(...).spawn(...)`.

Uploaded-file jobs include idempotency protection keyed off source identity, checksum, embedding config, and chunking config. That prevents accidental duplicate ingestion while still allowing re-ingestion after content or embedding changes.

Chunking is configurable through:

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

`CHUNK_SIZE` must be positive, `CHUNK_OVERLAP` must be zero or positive, and `CHUNK_OVERLAP` must be smaller than `CHUNK_SIZE`.

Embedding configuration is also explicit:

```env
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
KNOWLEDGE_EMBEDDING_DIMENSION=384
```

`KNOWLEDGE_EMBEDDING_DIMENSION` is the preferred setting name. The legacy `EMBEDDING_DIMENSION` alias remains supported for older environments.

If you change the embedding provider, model, or dimension, rebuild the knowledge index before running retrieval again. Retrieval validates both the stored collection metadata and the underlying pgvector column dimension, then fails loudly when the active embedding config does not match the indexed config.

### Supabase pgvector details

Production vector retrieval uses Supabase Postgres with pgvector through LangChain's `PGVector` tables:

- `knowledge_chunks` stores the source chunk text and chunk metadata used by the existing RAG flow.
- `langchain_pg_embedding.embedding` stores the actual `vector(...)` values used for similarity search.
- `langchain_pg_collection` stores per-collection embedding metadata so the app can reject stale indexes when the embedding config changes.

The production retrieval path assumes cosine distance and creates:

- a btree index on `langchain_pg_embedding.collection_id`
- an IVFFlat index on `langchain_pg_embedding.embedding` using `vector_cosine_ops` with an adaptive `lists` value up to `100`

That keeps the Supabase path simple while staying aligned with the current similarity query behavior.

Re-ingestion is required when any of these change:

- embedding provider
- embedding model
- embedding dimension
- chunking strategy
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- document parsing or cleaning logic that changes chunk text
- vector index settings in a way that materially changes retrieval behavior

For a clean rebuild:

1. Run `alembic upgrade head`.
2. If the vector dimension changed, update `KNOWLEDGE_EMBEDDING_DIMENSION` and let the migration recreate the pgvector storage shape.
3. Re-run `uv run python .\scripts\ingest_knowledge.py`.
4. Validate retrieval with a targeted smoke query or the opt-in Postgres integration test.

### Uploaded file storage providers

Local development can keep uploaded files in MinIO or the local filesystem:

```env
STORAGE_PROVIDER=minio
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=knowledge-files
MINIO_SECURE=false
```

Production can store uploaded files in Supabase Storage:

```env
STORAGE_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET=knowledge-files
```

When `STORAGE_PROVIDER=supabase`, uploaded file metadata is still stored in `knowledge_files`, including the provider, bucket, and object path used for ingestion. Keep the Supabase service role key on the backend only.

### Run ingestion from the CLI

```bash
uv run python .\scripts\ingest_knowledge.py
```

The CLI prints the chunking config used for that ingestion run.

### Trigger ingestion from a script

```bash
uv run python .\scripts\trigger_ingestion.py --source-type uploaded_file --source-id <file_id>
```

You can also queue a local-directory rebuild:

```bash
uv run python .\scripts\trigger_ingestion.py --source-type local_directory
```

### Trigger ingestion from the backend

Set an ingestion secret in `.env`:

```env
INGESTION_API_SECRET=dev-secret-change-me
INGESTION_BACKEND=local
```

Run the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Trigger the protected endpoint:

```bash
curl -X POST "http://localhost:8000/api/knowledge/ingest" \
  -H "x-ingestion-secret: dev-secret-change-me"
```

Example response:

```json
{
  "job_id": "0c7985f8-64a2-4c49-bc86-77f6111c1fd7",
  "status": "queued",
  "source_type": "local_directory",
  "file_id": null
}
```

If the same uploaded file has already completed with the same content and embedding/chunking config, the trigger returns `status: "skipped"` and does not dispatch a second ingestion run.

### Inspect ingestion jobs

Job state lives in `knowledge_ingestion_jobs`. Useful columns include:

- `status`
- `chunk_count`
- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `error_message`
- `started_at`
- `completed_at`

Example query:

```sql
select
  id,
  source_type,
  file_id,
  status,
  chunk_count,
  error_message,
  created_at,
  completed_at
from knowledge_ingestion_jobs
order by created_at desc
limit 20;
```

### Frontend handoff

This repository now includes a local Tavus test frontend under `frontend/`.

Run it with:

```bash
cd frontend
npm install
npm run dev
```

The Vite app runs at `http://localhost:5173` and uses:

```env
VITE_BACKEND_URL=http://localhost:8000
```

For Tavus local callback testing, the backend `.env` should set:

```env
PUBLIC_BACKEND_URL=<ngrok-or-cloudflare-url>
```

Tavus must call the public backend URL, not `localhost`.

The app follows this MVP flow:

1. Render a `Talk to Tumelo's AI Avatar` button.
2. Call `POST /api/tavus/conversations`.
3. Read `conversation_url` from the backend response.
4. Embed that URL in an iframe.
5. End the conversation with `POST /api/tavus/conversations/end`.

## Model experiments

Runtime prompt selection still loads versioned templates from `app/infrastructure/prompts/templates`.
If `prompt_version` is omitted in the API request, the backend falls back to `DEFAULT_PROMPT_VERSION`.
If `model_config_id` is omitted, the backend falls back to `DEFAULT_MODEL_CONFIG_ID`.

### OpenAI and OpenRouter provider configuration

The backend uses one OpenAI-compatible transport implementation, but it can keep separate provider credentials and base URLs loaded at the same time.
That means one running backend can send:

- `provider: "openai"` model configs to OpenAI directly
- `provider: "openrouter"` model configs to OpenRouter

Base environment variables:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL_CONFIG_ID=openai:gpt-4.1-mini
```

When `LLM_PROVIDER` and `LLM_MODEL` are set, they define the active default runtime model config and take precedence over `.env` defaults for that provider path.
The built-in model configs use `provider: "openai"` and continue to call OpenAI directly by default.
To evaluate OpenRouter-backed models in the same backend, add model configs with `provider: "openrouter"`:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3.5-sonnet
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<openrouter-api-key>
LLM_PROMPT_COST_PER_1M_TOKENS=3.0
LLM_COMPLETION_COST_PER_1M_TOKENS=15.0
DEFAULT_MODEL_CONFIG_ID=openrouter:anthropic/claude-3.5-sonnet
MODEL_CONFIGS_JSON=[
  {
    "config_id": "openrouter:anthropic/claude-3.5-sonnet",
    "provider": "openrouter",
    "model": "anthropic/claude-3.5-sonnet",
    "input_cost_per_1m_tokens": 3.0,
    "output_cost_per_1m_tokens": 15.0
  }
]
```

For backwards compatibility, the provider-specific variables still work on their own. The generic `LLM_*` values are the preferred single-model experiment path, while `MODEL_CONFIGS_JSON` remains the way to register additional named configs for comparisons.

### Embedding provider configuration

Retrieval defaults to the local Hugging Face embedding path:

```env
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
KNOWLEDGE_EMBEDDING_DIMENSION=384
```

OpenRouter-compatible embedding experiments can use:

```env
EMBEDDING_PROVIDER=openrouter
KNOWLEDGE_EMBEDDING_MODEL=openai/text-embedding-3-small
KNOWLEDGE_EMBEDDING_DIMENSION=1536
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

`KNOWLEDGE_EMBEDDING_DIMENSION` must match both the provider output and the pgvector storage dimension. If you change provider, model, or dimension, rebuild the knowledge index. If you change to a new vector dimension, run migrations before ingestion so the pgvector column type matches the new embedding size.

### Adding custom model configs

Built-in OpenAI model configs continue to work without extra setup.
For experiments, add extra model configs through `MODEL_CONFIGS_JSON` as a JSON array. The `provider` field must be either `openai` or `openrouter`:

```env
MODEL_CONFIGS_JSON=[
  {
    "config_id": "openrouter:anthropic/claude-3.5-sonnet",
    "provider": "openrouter",
    "model": "anthropic/claude-3.5-sonnet",
    "input_cost_per_1m_tokens": 3.0,
    "output_cost_per_1m_tokens": 15.0
  },
  {
    "config_id": "openrouter:openai/gpt-4.1-mini",
    "provider": "openrouter",
    "model": "openai/gpt-4.1-mini",
    "input_cost_per_1m_tokens": 0.0,
    "output_cost_per_1m_tokens": 0.0
  }
]
```

Set `DEFAULT_MODEL_CONFIG_ID` to one of those custom IDs, or pass a custom `model_config_id` per request or eval run.
This makes it possible to compare direct OpenAI models and OpenRouter-backed models without restarting the app to swap API keys.
If a model config ID is unknown, the backend fails with a clear validation error listing available IDs.

### Langfuse observability

Langfuse is optional request-level observability for chat, retrieval, and LLM execution. It stays disabled by default and complements, rather than replaces, MLflow or DagsHub experiment tracking.

Set:

```env
ENABLE_LANGFUSE_OBSERVABILITY=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=local
LANGFUSE_RELEASE=modal-v1
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_EXPORT_DEFAULT_LIMIT=100
```

- When `ENABLE_LANGFUSE_OBSERVABILITY=false`, the backend starts normally without Langfuse credentials.
- When `ENABLE_LANGFUSE_OBSERVABILITY=true`, both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are required and Langfuse initialization must succeed at startup.
- Retrieval traces log source names, chunk IDs, scores, and short content previews only; full retrieved documents are not sent by default.
- Runtime Langfuse failures are fail-open: chat responses still succeed, provider errors are logged safely, and the chat flow performs a best-effort `flush()` after each request for short-lived deployments such as Modal.
- Internal `chat_traces.id` remains the primary application trace key. When Langfuse exposes a provider trace ID, it is also stored on the internal trace as optional `external_trace_id` metadata and propagated as `langfuse_trace_id` for feedback/export workflows.
- To find a production request in Langfuse Cloud, search the trace list for the stored `external_trace_id` / `langfuse_trace_id` or start from the internal `chat_traces` row and follow that provider ID.
- Keep MLflow and DagsHub enabled separately for experiment runs, metrics, and eval comparisons.

### Exporting bad Langfuse traces into eval review queues

When production traces reveal weak answers, errors, slow responses, or negative feedback, export them into a JSONL review queue with:

```bash
python -m evals.langfuse.export_bad_langfuse_traces \
  --output evals/datasets/production_failures_review.jsonl \
  --score-name answer_quality \
  --max-score 0.6 \
  --from-date 2026-07-01 \
  --to-date 2026-07-06 \
  --limit 100
```

Use `--only-errors` for failed traces, `--detect-fallback-answer` for refusal-style answers, and `--append` to merge new rows without duplicating trace IDs.

The exporter writes a review dataset with empty expected fields by default so maintainers can inspect the production failure, fill `expected_facts`, `expected_answer_points`, and `expected_source_documents`, then promote the row into a scored eval dataset intentionally.

Detailed workflow notes live in `docs/evals/langfuse_trace_export.md`.

### Local-only MLflow tracking

Set:

```env
ENABLE_MLFLOW_TRACKING=true
ENABLE_DAGSHUB_TRACKING=false
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

When `MLFLOW_TRACKING_URI` is blank and DagsHub is disabled, MLflow falls back to a local file-backed store.
That local mode is optional and legacy for this repo. If you use it, MLflow will recreate local runtime output directories as needed.

If you prefer a local MLflow server and UI instead of the default file-backed store, start one first:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then point the eval runners at it:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
```

### DagsHub-backed remote tracking

Set:

```env
ENABLE_MLFLOW_TRACKING=true
ENABLE_DAGSHUB_TRACKING=true
DAGSHUB_REPO_OWNER=<your-dagshub-owner>
DAGSHUB_REPO_NAME=<your-dagshub-repo>
DAGSHUB_TOKEN=<your-dagshub-token>
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

The tracking helper initializes DagsHub before selecting the MLflow experiment, so the existing MLflow logging code continues to log params, metrics, and artifacts to the remote DagsHub-backed MLflow backend.
Leave `MLFLOW_TRACKING_URI` blank when using DagsHub so the DagsHub client can configure the tracking backend.

Authentication can be provided either with `DAGSHUB_TOKEN` in `.env` or with a prior `dagshub login` on the machine.

### Running eval workflows

When `ENABLE_MLFLOW_TRACKING=true`, the eval runners reuse the shared MLflow/DagsHub tracker in `app/infrastructure/tracking/`.
With DagsHub enabled, run metadata is logged to the remote backend while eval result files remain available under `evals/results/`.

Workflows that log to MLflow/DagsHub when tracking is enabled:

- `evals/runners/run_generation_eval.py`
- `evals/runners/run_rag_eval.py`
- `evals/runners/run_retrieval_eval.py`
- `evals/runners/run_retrieval_sweep.py`
- `scripts/run_chunking_experiment.py`
- `scripts/run_embedding_experiment.py`

Core tracked params are standardized where they apply:

- `embedding_model`
- `chunk_size`
- `chunk_overlap`
- `retriever_type`
- `top_k`
- `query_rewriting`
- `reranker`
- `llm_model`
- `prompt_version`

Core tracked metrics are standardized where they apply:

- `recall_at_k`
- `precision_at_k`
- `mrr`
- `faithfulness`
- `answer_relevance`
- `latency`
- `cost`

Local comparison workflows keep inspectable artifacts alongside MLflow:

- summary CSV and JSON
- ranked comparison markdown table
- per-run detailed result artifacts

Use local artifacts for quick inspection under `evals/results/`, and use MLflow or DagsHub to compare the same runs remotely by the standardized params and metrics above.

The model comparison workflow:

```bash
python -m evals.runners.run_model_eval \
  --models openai:gpt-4.1-mini openai:gpt-4.1 \
  --prompt-version v1_professional \
  --dataset evals/datasets/model_eval_dataset.jsonl \
  --experiment-name personal-chatbot-model-comparison
```

The fixed-context generation comparison workflow:

```bash
uv run python -m evals.runners.run_generation_eval \
  --dataset evals/datasets/generation_eval_dataset.jsonl \
  --prompt-version v1_professional
```

OpenAI example:

```bash
LLM_PROVIDER=openai \
LLM_MODEL=gpt-4.1-mini \
LLM_BASE_URL=https://api.openai.com/v1 \
uv run python -m evals.runners.run_generation_eval --prompt-version v1_professional
```

OpenRouter example:

```bash
LLM_PROVIDER=openrouter \
LLM_MODEL=anthropic/claude-3.5-sonnet \
LLM_BASE_URL=https://openrouter.ai/api/v1 \
LLM_PROMPT_COST_PER_1M_TOKENS=3.0 \
LLM_COMPLETION_COST_PER_1M_TOKENS=15.0 \
uv run python -m evals.runners.run_generation_eval --prompt-version v1_professional
```

This runner keeps retrieval fixed by loading context directly from the dataset, then logs:

- `llm_provider`
- `llm_model`
- `llm_base_url`
- quality and groundedness metrics
- latency average, p50, and p95
- prompt, completion, and total token counts where available
- estimated prompt, completion, and total cost where configured

When the active generation model uses `provider: "openrouter"` and the selected model config does not already include token pricing, `evals/runners/run_generation_eval.py` now looks up the model automatically with OpenRouter's single-model endpoint and uses `data.pricing.prompt` and `data.pricing.completion` to estimate cost. Those API values are documented by OpenRouter as USD per token and are converted to USD per 1M tokens inside the runner.

The RAG evaluation workflow:

```bash
python -m evals.runners.run_rag_eval \
  --model openai:gpt-4.1-mini \
  --prompt-version v1_professional \
  --run-name portfolio-rag-eval
```

This workflow logs retrieval plus generation params together, including `embedding_model`, `chunk_size`, `chunk_overlap`, `retriever_type`, `top_k`, `query_rewriting=false`, `reranker`, `llm_model`, and `prompt_version`. It also logs `recall_at_k`, `precision_at_k`, `mrr`, `faithfulness`, `answer_relevance`, `latency`, and `cost` where available.

The canonical RAG benchmark contract is documented in `evals/README.md`.

The retrieval-only baseline workflow:

```bash
uv run python -m evals.runners.run_retrieval_eval --config evals/configs/retrieval_baseline.json
```

The reranked retrieval workflow:

```bash
uv run python -m evals.runners.run_retrieval_eval --config evals/configs/retrieval_reranked_llm.json
```

The same runner still supports direct flag overrides when needed:

```bash
python -m evals.runners.run_retrieval_eval --k 5 --enable-reranker --reranker-type llm --reranker-initial-top-k 20
```

The retrieval artifact payload includes reranker config, before/after chunk order, context relevance, embedding config, and chunking config used for that run.

To compare multiple retrieval configurations in one command, use the sweep runner with a YAML config:

```bash
python -m evals.runners.run_retrieval_sweep --config evals/configs/retrieval_sweep.yaml
```

The protected backend exposes the same batch pattern over HTTP:

```bash
curl -X POST http://localhost:8000/api/evals/retrieval-sweeps \
  -H "Content-Type: application/json" \
  -H "x-eval-admin-token: <EVAL_ADMIN_TOKEN>" \
  -d '{
    "experiments": [
      {"name": "retrieval-baseline-k5", "retriever_type": "vector", "top_k": 5},
      {
        "name": "retrieval-reranked-k5-from20",
        "retriever_type": "vector",
        "top_k": 5,
        "reranker_enabled": true,
        "reranker_type": "llm",
        "reranker_initial_top_k": 20
      },
      {"name": "retrieval-keyword-k5", "retriever_type": "keyword", "top_k": 5}
    ]
  }'
```

The sample sweep config defines one run per experiment:

```yaml
experiments:
  - name: retrieval-baseline-k5
    retriever_type: vector
    top_k: 5

  - name: retrieval-reranked-k5-from10
    retriever_type: vector
    top_k: 5
    reranker_enabled: true
    reranker_type: llm
    reranker_initial_top_k: 10

  - name: retrieval-reranked-k5-from20
    retriever_type: vector
    top_k: 5
    reranker_enabled: true
    reranker_type: llm
    reranker_initial_top_k: 20

  - name: retrieval-keyword-k5
    retriever_type: keyword
    top_k: 5
```

Optional per-experiment overrides:

- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `chunk_size`
- `chunk_overlap`
- `reranker_enabled`
- `reranker_type`
- `reranker_model`
- `reranker_initial_top_k`

Each sweep experiment logs as its own MLflow or DagsHub-backed MLflow run and records:

- `retriever_type`
- `top_k`
- `reranker_enabled`
- `reranker_type`
- `reranker_initial_top_k`
- `reranker_final_top_k`
- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `dataset_path`
- `git_commit_sha`
- retrieval metrics including `mrr`, `recall_at_k`, `mean_precision_at_k`, `hit_at_k`, `context_relevance`, and query counts

Sweep artifacts are written under `evals/results/retrieval_sweeps/`. Each run gets its own artifact directory, and the sweep root also includes:

- `retrieval_sweep_summary.csv`
- `retrieval_sweep_summary.json`
- `retrieval_sweep_ranking.md`
- `sweep_manifest.json`

The summary artifacts are ranked by `recall_at_k`, then `mrr`, then `precision_at_k`, and the CLI prints the best configuration clearly.

To compare multiple chunking strategies without editing code:

```bash
python scripts/run_chunking_experiment.py --configs "300:50,500:100,800:150,1000:200"
```

This re-ingests the knowledge base for each chunk config, runs retrieval eval, logs one MLflow run per chunk setup, and writes local artifacts under `evals/results/chunking_experiments/`, including:

- `chunking_experiment_summary.csv`
- `chunking_experiment_summary.json`
- `chunking_experiment_ranking.md`
- per-config `results.json`, `results.csv`, and `config.json`

The summary output is ranked by `recall_at_k`, then `mrr`, then `precision_at_k`.

To compare multiple embedding setups against the same retrieval dataset:

```bash
python scripts/run_embedding_experiment.py --config evals/configs/retrieval_embedding_matrix.example.json
```

This workflow logs one MLflow run per embedding setup and writes local artifacts under `evals/results/embedding_experiments/`, including:

- `embedding_experiment_summary.csv`
- `embedding_experiment_summary.json`
- `embedding_experiment_ranking.md`
- per-model `results.json`, `results.csv`, and `config.json`

The prompt comparison workflow:

```bash
python scripts/compare_prompts.py \
  --prompt-version v1_professional \
  --prompt-version v2_warm_conversational \
  --experiment-name personal-chatbot-model-comparison
```

Artifacts are written under `evals/results/` and `evals/prompt_eval_results/`.
When `ENABLE_MLFLOW_TRACKING=true`, the runners log one MLflow run per evaluated unit, plus workflow-level summary artifacts for the comparison-style retrieval, chunking, and embedding workflows.

## Tests

```bash
pytest
```

Opt-in Postgres smoke coverage for the Supabase-compatible pgvector path:

```bash
RUN_DB_INTEGRATION_TESTS=true \
TEST_DATABASE_URL=postgresql+psycopg://... \
pytest tests/integration/test_supabase_pgvector_smoke.py
```

Targeted eval checks for the retrieval runners:

```bash
PYTHONPATH=. pytest tests/evals/test_retrieval_eval_runner.py tests/evals/test_retrieval_sweep_runner.py tests/test_embedding_experiment_script.py
```
