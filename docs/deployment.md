# Deployment

## Supported topology

- A separately managed frontend (intended for Vercel) calls the backend from an allowed origin.
- Modal hosts the FastAPI ASGI function and a named ingestion worker function.
- Managed PostgreSQL with pgvector stores application, knowledge, trace, and optional evaluation data.
- Supabase Storage or another configured provider stores uploaded source bytes.
- Upstash Redis REST optionally provides request cache, locks, and rate limiting.
- Langfuse Cloud optionally receives operational traces.
- MLflow or DagsHub receives evaluation runs from developer/automation environments.
- GitHub Actions runs CI and deploys successful `main` commits to Modal.

This repository does not contain the frontend or its Vercel configuration.

## Prerequisites

- Python 3.12, `uv`, and a Modal account/token.
- A managed PostgreSQL database with pgvector support and direct/session access for migrations.
- A model-provider API key.
- A deployed frontend origin.
- Optional Supabase Storage, Upstash, Langfuse, and DagsHub projects.

Install deployment dependencies and authenticate:

```bash
uv sync --locked --extra deploy
uv run modal setup
```

## Production secrets

`modal_app.py` expects a Modal secret named `production-chatbot-api-secrets`. At minimum it should contain:

```env
APP_ENV=production
FRONTEND_ORIGIN=https://your-frontend.example.com
DATABASE_URL=postgresql+psycopg://user:password@pooler-host:6543/postgres?sslmode=require
DATABASE_DIRECT_URL=postgresql+psycopg://user:password@direct-host:5432/postgres?sslmode=require
VECTOR_STORE_PROVIDER=pgvector
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-provider-key
INGESTION_API_SECRET=your-long-random-secret
INGESTION_BACKEND=modal
MODAL_INGESTION_APP_NAME=production-chatbot-api
MODAL_INGESTION_FUNCTION_NAME=run_ingestion_job
```

Add embedding, storage, Upstash, and Langfuse credentials only when used. Consult [configuration](configuration.md) and the detailed [Modal reference](deployment/modal.md). Never commit the values or place them in GitHub variables visible to untrusted workflows.

Create/update the Modal secret through the Modal dashboard or CLI. When using the CLI, avoid shell history for secrets; prefer an approved secure input process.

## Database and storage preparation

1. Create a production database/project.
2. Enable pgvector; migrations also issue `CREATE EXTENSION IF NOT EXISTS vector` where appropriate.
3. Choose runtime/pooler `DATABASE_URL` and direct/session `DATABASE_DIRECT_URL` values.
4. Configure the uploaded-file storage bucket/provider.
5. Run migrations from a controlled environment with production URLs loaded securely:

   ```bash
   uv run alembic upgrade head
   uv run alembic current
   ```

Do not run application startup and schema migration concurrently. Apply migrations before deploying code that depends on them. For Supabase-specific URLs and storage setup, read [Supabase deployment](deployment/supabase.md).

## Deploy Modal

```bash
uv run modal deploy -m modal_app
```

The deployment creates:

- `fastapi_app`, an ASGI function with a 300-second timeout;
- `run_ingestion_job`, a 300-second background ingestion worker.

The image includes `app/`, `alembic/`, `evals/`, `alembic.ini`, and `main.py`, and pre-downloads the default Hugging Face embedding model.

Record the FastAPI URL printed by Modal. Verify:

```bash
curl --fail https://your-modal-url.example/health
curl --fail https://your-modal-url.example/ready
```

`/ready` checks PostgreSQL and, when enabled, Upstash Redis.

## Production ingestion

Trigger curated directory ingestion through the protected endpoint:

```bash
curl -X POST https://your-modal-url.example/api/knowledge/ingest \
  -H "Content-Type: application/json" \
  -H "x-ingestion-secret: your-ingestion-secret" \
  -d '{"source_type":"local_directory"}'
```

With `INGESTION_BACKEND=modal`, the API looks up and spawns `production-chatbot-api::run_ingestion_job`. Confirm the returned job in `knowledge_ingestion_jobs`. Uploaded files must be stored with the same `STORAGE_PROVIDER` later used by the worker.

Do not use the synchronous local ingestion command casually against production. See [knowledge ingestion](ingestion.md) for idempotency and re-indexing behavior.

## Frontend and CORS

Set `FRONTEND_ORIGIN` to exact origins, comma-separated when necessary:

```env
FRONTEND_ORIGIN=https://portfolio.example.com,https://preview.example.com
```

Avoid broad preview wildcards: the implementation accepts exact strings, not wildcard-origin patterns. The configured frontend should call the Modal `/chat` URL over HTTPS. Update the Modal secret and redeploy/restart after origin changes.

## GitHub Actions CI/CD

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`:

```bash
uv sync --locked
uv run ruff check .
uv run python -m pytest
```

`.github/workflows/cd.yml` runs only after a successful CI workflow for a push to `main`. Configure the GitHub `production` environment with:

- secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`;
- variable: `MODAL_HEALTHCHECK_URL`.

CD installs `--extra deploy`, runs `uv run modal deploy -m modal_app`, then retries the health URL. `MODAL_ENVIRONMENT` is set to `main` by the workflow.

The Modal application secret is managed in Modal, not copied into GitHub Actions.

## Production differences

- `APP_ENV=production` enforces frontend origin, database URL, and model API key.
- Use managed URLs and TLS; never use Compose hostnames or local credentials.
- Prefer a pooler/runtime URL for API traffic and a direct/session URL for migrations.
- Set `INGESTION_BACKEND=modal`.
- Configure explicit embedding model/dimension and re-ingest after controlled changes.
- Use a production object-storage provider; local filesystem storage is ephemeral on serverless workers.
- Enable optional Redis and Langfuse only after their credentials and privacy policy are ready.

## Rollback

1. Identify the last known-good Git commit and any schema compatibility constraints.
2. Redeploy that commit with `uv run modal deploy -m modal_app`.
3. Verify `/health`, `/ready`, a direct response, and a grounded portfolio response.
4. Do not downgrade the database blindly. Alembic downgrades can delete or transform data; write and rehearse an explicit rollback plan for each migration.
5. If the embedding/index configuration changed, restore the matching configuration and re-ingest the full approved source set.

GitHub Actions deploys commits, so a revert merged to `main` is the normal auditable rollback path.

## Secret rotation

Rotate provider, database, ingestion, Upstash, Langfuse, DagsHub, Modal, and Supabase keys in their source system, update the appropriate secret store, redeploy, and verify. During database credential rotation, update both runtime and direct URLs. Revoke old credentials only after the new deployment passes readiness and functional checks.

See [troubleshooting](troubleshooting.md) and the detailed [Modal reference](deployment/modal.md).
