# Modal deployment

This backend can be deployed to Modal as an ASGI web service without changing the local `uvicorn` or Docker Compose workflow.

## Prerequisites

Install the deployment-only tooling:

```bash
uv sync --extra deploy
```

Authenticate the Modal CLI:

```bash
modal setup
```

## Required secrets

Create a Modal Secret named `production-chatbot-api-secrets`. Keep production values in Modal Secrets only; do not commit a production `.env` file.

Minimum required environment variables:

```env
APP_ENV=production
FRONTEND_ORIGIN=https://your-frontend-domain.com
INGESTION_API_SECRET=...
INGESTION_BACKEND=modal
MODAL_INGESTION_APP_NAME=production-chatbot-api
MODAL_INGESTION_FUNCTION_NAME=run_ingestion_job

DATABASE_URL=postgresql+psycopg://...pooler...:6543/postgres?sslmode=require
DATABASE_DIRECT_URL=postgresql+psycopg://...:5432/postgres?sslmode=require

VECTOR_STORE_PROVIDER=supabase_pgvector
STORAGE_PROVIDER=supabase

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET=knowledge-files

REDIS_URL=rediss://...
REDIS_TOKEN=...

LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=...

ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

ENABLE_MLFLOW_TRACKING=true
MLFLOW_TRACKING_URI=...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

Example:

```bash
modal secret create production-chatbot-api-secrets `
  APP_ENV=production `
  FRONTEND_ORIGIN=https://your-frontend-domain.com `
  INGESTION_API_SECRET=... `
  INGESTION_BACKEND=modal `
  MODAL_INGESTION_APP_NAME=production-chatbot-api `
  MODAL_INGESTION_FUNCTION_NAME=run_ingestion_job `
  DATABASE_URL=postgresql+psycopg://...pooler...:6543/postgres?sslmode=require `
  DATABASE_DIRECT_URL=postgresql+psycopg://...:5432/postgres?sslmode=require `
  VECTOR_STORE_PROVIDER=supabase_pgvector `
  STORAGE_PROVIDER=supabase `
  SUPABASE_URL=https://your-project.supabase.co `
  SUPABASE_SERVICE_ROLE_KEY=... `
  SUPABASE_STORAGE_BUCKET=knowledge-files `
  REDIS_URL=rediss://... `
  REDIS_TOKEN=... `
  LLM_PROVIDER=openai `
  LLM_MODEL=gpt-4.1-mini `
  LLM_API_KEY=...
```

## Deploy

Deploy the ASGI app and ingestion worker from the same Modal definition:

```bash
modal deploy modal_app.py
```

The deployed app name is `production-chatbot-api`. The same deploy publishes the `run_ingestion_job` worker function used by the API trigger.

## Smoke tests

Check liveness:

```bash
curl https://<modal-endpoint>/health
```

Check readiness:

```bash
curl https://<modal-endpoint>/ready
```

Readiness verifies:

- database connectivity
- Redis connectivity when response caching or rate limiting is enabled
- production config that was already validated during startup

Test chat:

```bash
curl -X POST https://<modal-endpoint>/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"What can you tell me about Tumelo?\"}"
```

Queue ingestion:

```bash
curl -X POST https://<modal-endpoint>/api/knowledge/ingest `
  -H "x-ingestion-secret: <ingestion-secret>" `
  -H "Content-Type: application/json" `
  -d "{\"source_type\":\"uploaded_file\",\"file_id\":\"<file_id>\"}"
```

Expected response:

```json
{
  "job_id": "0c7985f8-64a2-4c49-bc86-77f6111c1fd7",
  "status": "queued",
  "source_type": "uploaded_file",
  "file_id": "<file_id>"
}
```

You can also trigger the same path from the repo:

```bash
uv run python .\scripts\trigger_ingestion.py --source-type uploaded_file --source-id <file_id>
```

## Operational notes

- Local development is unchanged. Continue using `uvicorn main:app --reload` or `docker compose up`.
- The API returns immediately after writing a `knowledge_ingestion_jobs` row; ingestion work runs in the Modal worker.
- Modal runtime traffic should use `DATABASE_URL`. Do not run Alembic migrations automatically on startup.
- Run migrations separately with `DATABASE_DIRECT_URL`.
- `FRONTEND_ORIGIN` is required in production. Startup fails fast if it is missing.
- If Redis-backed features are enabled but `REDIS_URL` is missing or unreachable, `/ready` returns `503`.

## Inspect and retry jobs

Job visibility lives in Postgres. Inspect failures in `knowledge_ingestion_jobs`, for example:

```sql
select
  id,
  file_id,
  status,
  chunk_count,
  error_message,
  started_at,
  completed_at
from knowledge_ingestion_jobs
order by created_at desc
limit 20;
```

Retries reuse the same trigger path. If the previous run failed, a new `pending` job is created and dispatched. If the same source already completed with the same checksum and embedding/chunking config, the API returns `status: "skipped"` instead of re-ingesting it.

## Troubleshooting

- Supabase SSL or pooler issues: ensure `DATABASE_URL` includes `?sslmode=require` and uses the transaction pooler host and port `6543`.
- Migration connection issues: use the direct or session-mode URL for `DATABASE_DIRECT_URL`; do not point migrations at the runtime pooler unless Supabase explicitly documents that setup for your project.
- Missing secrets: if startup fails on Modal, verify `production-chatbot-api-secrets` exists and includes `FRONTEND_ORIGIN`, `DATABASE_URL`, `INGESTION_BACKEND=modal`, and the active LLM key.
- Redis readiness failures: if `/ready` reports `redis: "unavailable"`, verify the Upstash URL, token, and network reachability. If it reports `redis: "misconfigured"`, a Redis-backed feature is enabled without a usable `REDIS_URL`.
