# Supabase Postgres Deployment

Use Supabase Postgres for production while keeping local Docker Compose Postgres for development.

## Environment variables

Set these in production:

```env
APP_ENV=production
FRONTEND_ORIGIN=https://your-frontend.example.com
DATABASE_URL=postgresql+psycopg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require
DATABASE_DIRECT_URL=postgresql+psycopg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres?sslmode=require
```

Local development can keep both variables pointed at the local database:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot
DATABASE_DIRECT_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot
```

## Which URL to use

- `DATABASE_URL`: runtime application traffic. For Supabase, use the transaction-mode pooler URL on port `6543`.
- `DATABASE_DIRECT_URL`: Alembic migrations, admin scripts, and one-off maintenance work. Copy the session-mode or direct connection string from the Supabase Connect panel. In this project, the working setup used port `5432`.

If `DATABASE_DIRECT_URL` is not set, Alembic falls back to `DATABASE_URL`.

## Where to find the URLs

In the Supabase dashboard:

1. Open your project.
2. Open the `Connect` dialog for the database.
3. Copy the transaction-mode pooler connection string for `DATABASE_URL`.
4. Copy the session-mode or direct connection string for `DATABASE_DIRECT_URL`.

Keep both values secret and inject them through your deployment environment instead of committing them.
Do not hand-build the hostnames if you can avoid it. Copy the exact strings from Supabase and then change only the SQLAlchemy driver prefix to `postgresql+psycopg://` when needed.

## Supabase Storage setup

Use Supabase Storage for uploaded knowledge files only when you set:

```env
STORAGE_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET=knowledge-files
```

This is separate from:

```env
VECTOR_STORE_PROVIDER=supabase_pgvector
```

`VECTOR_STORE_PROVIDER` controls chunk and embedding persistence. `STORAGE_PROVIDER` controls the original uploaded file bytes used by the upload and ingestion flows.

Recommended bucket setup:

1. Open your Supabase project.
2. Go to `Storage`.
3. Create a bucket named `knowledge-files`.
4. Keep the bucket private.
5. Inject the service role key into the backend environment only.

This integration is server-side only. Do not expose the service role key to the frontend.

## SSL

Supabase production connections require SSL. Keep `?sslmode=require` on both production URLs. Do not disable SSL for production connections.

If your password contains special characters such as `!`, URL-encode them inside the connection string. For example, `!` becomes `%21`.

## Running migrations

When `DATABASE_DIRECT_URL` is already exported:

```bash
alembic upgrade head
```

You can also run migrations with an explicit direct URL:

```bash
DATABASE_DIRECT_URL="postgresql+psycopg://..." alembic upgrade head
```

If you use `APP_ENV=production` in your deployment shell:

```bash
APP_ENV=production alembic upgrade head
```

The migration history already enables `CREATE EXTENSION IF NOT EXISTS vector`, and the latest pgvector migration also backfills the retrieval indexes when the LangChain tables already exist.

## pgvector storage layout

This app keeps the production RAG data split across:

- `knowledge_chunks`: source chunk text plus chunk metadata for the chat and admin flows
- `langchain_pg_embedding`: pgvector embeddings stored in the `embedding` column
- `langchain_pg_collection`: collection-level embedding metadata used to detect stale indexes

The retrieval path uses cosine distance. After ingestion, the app ensures:

- `ix_langchain_pg_embedding_collection_id`
- `ix_langchain_pg_embedding_embedding_cosine_ivfflat`

The IVFFlat index uses `vector_cosine_ops` with `lists = 100`.

## Re-ingestion and rebuilds

Re-ingest the knowledge base after any of these changes:

1. the embedding provider changes
2. the embedding model changes
3. `KNOWLEDGE_EMBEDDING_DIMENSION` changes
4. `CHUNK_SIZE` or `CHUNK_OVERLAP` changes
5. document parsing, cleaning, or chunking logic changes
6. vector index settings change enough to require a rebuild

Recommended rebuild flow:

1. Run `alembic upgrade head`.
2. If the vector dimension changed, update `KNOWLEDGE_EMBEDDING_DIMENSION` before ingestion.
3. Re-run `uv run python .\scripts\ingest_knowledge.py`.
4. Validate retrieval with a targeted query or the opt-in integration smoke test.

Opt-in smoke test:

```bash
RUN_DB_INTEGRATION_TESTS=true TEST_DATABASE_URL="postgresql+psycopg://..." pytest tests/integration/test_supabase_pgvector_smoke.py
```

## Verifying connectivity

Run the backend with the pooled runtime URL:

```bash
DATABASE_URL="postgresql+psycopg://..." APP_ENV=production uvicorn app.main:app --reload
```

Then check readiness:

```bash
curl http://localhost:8000/ready
```

Expected success response:

```json
{
  "status": "ok",
  "database": "ok"
}
```

`GET /health` remains a liveness check and does not verify database connectivity.

## Troubleshooting

- If `alembic upgrade head` works but `/ready` returns `503`, the migration URL is valid and the runtime `DATABASE_URL` is still wrong or the app has not been restarted since the `.env` change.
- After changing `.env`, fully restart `uvicorn`. Hot reload does not guarantee that process-level environment changes are re-read.
- If you see `invalid interpolation syntax` from Alembic, the password is probably URL-encoded and contains `%`. This repo now escapes that correctly inside `alembic/env.py`.
- If you see `failed to resolve host`, the hostname in the connection string is wrong for your environment or DNS cannot resolve it. Recopy the exact connection string from the Supabase Connect panel instead of editing the hostname manually.
- PowerShell `curl` is an alias for `Invoke-WebRequest`, so a failing `/ready` check shows up as a command error even when the response body includes useful JSON.
