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

LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=...

ENABLE_LANGFUSE_OBSERVABILITY=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=production

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
  ENABLE_REDIS=true `
  UPSTASH_REDIS_REST_URL=https://...upstash.io `
  UPSTASH_REDIS_REST_TOKEN=... `
  RATE_LIMIT_ENABLED=true `
  RATE_LIMIT_MAX_REQUESTS=20 `
  RATE_LIMIT_WINDOW_SECONDS=60 `
  EXACT_CACHE_ENABLED=true `
  EXACT_CACHE_TTL_SECONDS=300 `
  REQUEST_LOCK_ENABLED=true `
  REQUEST_LOCK_TTL_SECONDS=30 `
  LLM_PROVIDER=openai `
  LLM_MODEL=gpt-4.1-mini `
  LLM_API_KEY=...
```

## Deploy

Deploy the ASGI app and ingestion worker from the same Modal definition:

```bash
modal deploy -m modal_app
```

The deployed app name is `production-chatbot-api`. The same deploy publishes the `run_ingestion_job` worker function used by the API trigger.

## GitHub Actions CI/CD

Continuous integration and deployment are separate workflows:

- `.github/workflows/ci.yml` validates every pull request targeting `main` and every
  push to `main`. It installs the locked Python 3.12 environment, runs the repository's
  configured Ruff lint checks, and runs the pytest suite. The project does not
  currently configure a formatter or type checker, so CI does not invent
  repository-wide checks for either. CI has no Modal credentials, and optional
  external-database integration tests remain disabled.
- `.github/workflows/cd.yml` listens for completion of the workflow named `CI`. It
  deploys only when that CI run succeeded, originated from a push, and validated the
  `main` branch. CD checks out the exact commit SHA reported by the successful CI run
  before deploying it.

Pull requests, including pull requests from forks, cannot pass the CD workflow's
push-only condition or access its production credentials. CD uses the GitHub
`production` environment, deploys the importable `modal_app` module to the Modal
workspace `tumelokonaite` in its `main` environment, and calls the public `/health`
endpoint afterward.
Production deployments are serialized, and an in-progress deployment is never
cancelled by a newer commit.

### One-time Modal setup

1. Authenticate a trusted local Modal CLI with `modal setup` (or your existing
   authenticated profile), then verify that it targets the `tumelokonaite` workspace:

   ```bash
   modal profile current
   modal token info
   modal environment list
   ```

2. In the `tumelokonaite` Modal workspace, create a service user/token dedicated to
   GitHub Actions, with only the access needed to deploy into the `main` environment. Copy the
   token ID and secret when they are shown. A personal token can instead be created
   interactively with `modal token new`, but a dedicated CI/CD identity is preferred.
3. Ensure the `production-chatbot-api-secrets` runtime secret described above exists
   in the Modal `main` environment. Application runtime secrets stay in Modal;
   do not duplicate them in GitHub.

### One-time GitHub setup

In the repository's **Settings > Environments**, create an environment named
`production` and configure it as follows:

- Limit deployment branches to the protected `main` branch.
- Add environment secrets named `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
- Add an environment variable named `MODAL_HEALTHCHECK_URL` containing the complete
  public liveness URL, including `/health` (for example,
  `https://example--production-chatbot-api-fastapi-app.modal.run/health`). Do not put
  credentials or other secrets in this URL.
- Optionally add required reviewers when deployment should pause for manual approval
  after CI succeeds.

The GitHub environment and Modal environment intentionally have different names:
GitHub uses `production` for deployment protection and credentials, while Modal uses
`main`, where this project's app and runtime secret already live. The CD workflow sets
`MODAL_ENVIRONMENT=main` explicitly and never relies on a developer's default.

### Operation and troubleshooting

- Open the repository's **Actions** tab and select the failing `CI` or `CD` run to
  inspect the individual Ruff, pytest, deploy, or health-check step. A failed CI run
  never starts a production deployment; a failed Modal command or health request
  marks CD as failed.
- To redeploy a previous commit, find that commit's original `CI` push run in
  **Actions** and choose **Re-run all jobs**. A successful rerun emits a new completed
  CI event, which starts CD for the same validated commit. Confirm that redeploying old
  application code is safe for the current database schema first.
- To suspend automatic production deployment temporarily while retaining the CI
  workflow, add a required reviewer to the `production` environment and leave new
  deployments unapproved. Remove the temporary protection when deployment should
  resume. You can also disable only the `CD` workflow from its Actions page without
  affecting pull-request validation.
- If authentication fails, replace the two GitHub environment secrets with a newly
  issued Modal CI/CD token. Never print tokens in workflow commands or logs.
- If the smoke test fails after a successful deploy, verify that
  `MODAL_HEALTHCHECK_URL` is the current public `/health` URL and inspect the Modal app
  logs. The `/ready` endpoint is intentionally not used here because it also checks
  production dependencies and is better suited to deeper operational monitoring.

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
- If Redis-backed features are enabled but Upstash credentials are missing or unreachable, `/ready` returns `503`.

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
- Missing secrets: if startup fails on Modal, verify `production-chatbot-api-secrets` exists and includes `FRONTEND_ORIGIN`, `DATABASE_URL`, `INGESTION_BACKEND=modal`, the active LLM key, and both Langfuse keys when `ENABLE_LANGFUSE_OBSERVABILITY=true`.
- Langfuse behavior: when `ENABLE_LANGFUSE_OBSERVABILITY=false`, the app starts without Langfuse credentials. When it is `true`, startup fails fast if the Langfuse client cannot initialize, but runtime Langfuse API failures do not block chat responses.
- Redis readiness failures: if `/ready` reports `redis: "unavailable"`, verify the Upstash REST URL, token, and network reachability. If it reports `redis: "misconfigured"`, `ENABLE_REDIS=true` is set without usable Upstash credentials.
