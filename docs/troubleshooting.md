# Troubleshooting

Start with the process logs, `docker compose ps`, `/health`, and `/ready`. Never paste secret-bearing URLs or tokens into issues.

## Application cannot connect to PostgreSQL

### Symptom

`/ready` returns `503`, migrations raise a connection error, or startup/chat persistence fails.

### Likely cause

PostgreSQL is stopped, the host/port is wrong for the execution mode, credentials/database do not match, or a managed database requires TLS/direct access.

### Diagnosis

```bash
docker compose ps db
docker compose logs db
docker compose exec db pg_isready -U postgres -d production_chatbot
```

Check whether the app runs on the host (`127.0.0.1:5434`) or in Compose (`db:5432`). Inspect `DATABASE_DIRECT_URL` separately when only migrations fail.

### Resolution

Start `db`, restore the correct mode-specific URL, and retry `uv run alembic upgrade head`. For managed services, use the provider's required TLS parameters and a direct/session URL for migrations.

## pgvector extension is missing

### Symptom

Migrations or retrieval fail with `type "vector" does not exist`, missing extension, operator, or index errors.

### Likely cause

The PostgreSQL server lacks pgvector or the database role cannot create the extension.

### Diagnosis

```bash
docker compose exec db psql -U postgres -d production_chatbot \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

### Resolution

Use the Compose `pgvector/pgvector` image or enable pgvector through the managed provider. With adequate privileges run `CREATE EXTENSION IF NOT EXISTS vector`, then migrate. Do not substitute a plain PostgreSQL image without installing pgvector.

## Migrations fail

### Symptom

`uv run alembic upgrade head` exits with connection, multiple-head, missing-revision, or permission errors.

### Likely cause

Wrong direct URL, incomplete checkout, schema drift, insufficient DDL permission, or divergent migration branches.

### Diagnosis

```bash
uv run alembic heads
uv run alembic current
uv run alembic history
```

The repository should report one head. Confirm the target database before changing it.

### Resolution

Use the session/direct database URL, restore missing revisions, merge migration heads in code when necessary, and grant migration privileges. Do not stamp or downgrade a production database merely to hide unexplained drift.

## No documents are returned

### Symptom

Portfolio chat falls back or retrieval evaluation returns empty source lists.

### Likely cause

Knowledge was never ingested, the wrong database/collection is selected, the similarity threshold is too high, or embedding configuration differs from the indexed data.

### Diagnosis

```bash
docker compose exec db psql -U postgres -d production_chatbot \
  -c "SELECT source, count(*) FROM knowledge_chunks GROUP BY source ORDER BY source;"
uv run python -m evals.runners.run_retrieval_eval --config evals/configs/retrieval_baseline.json
```

Check `KNOWLEDGE_COLLECTION_NAME`, provider/model/dimension, `RETRIEVER_TYPE`, and `RETRIEVAL_MIN_SIMILARITY`.

### Resolution

Run migrations and `uv run python scripts/ingest_knowledge.py` against the same database used by the API. Restore the index's embedding settings or perform a controlled full re-ingestion.

## Ingestion completes but retrieval is empty

### Symptom

The CLI reports chunks, and `knowledge_chunks` is populated, but vector search returns none.

### Likely cause

The vector collection is absent/stale, provider/model/dimension changed after ingestion, the embedding provider failed during collection replacement, or the threshold filters every match. Collection replacement currently deletes/recreates the collection before requesting replacement embeddings, so it is not atomic.

### Diagnosis

Compare `.env` used by ingestion and API. Run retrieval with a lower temporary threshold in an isolated local environment and inspect retrieval evaluation per-query results.

### Resolution

Align configuration, schema, provider access, and collection, then re-ingest all sources. Restore an evidence-based threshold after evaluation; do not permanently set it to zero just to mask an index mismatch. Treat a failed ingestion as requiring vector-count/retrieval verification even when chunk rows were committed.

## Embedding dimensions do not match

### Symptom

pgvector reports different vector dimensions, or embedding insert/search fails.

### Likely cause

`KNOWLEDGE_EMBEDDING_DIMENSION` does not match the model output or migrated vector schema/index.

### Diagnosis

Check provider documentation/model output, current environment, migration vector dimensions, and the configuration stored with ingestion jobs.

### Resolution

Use the model's actual dimension. Apply an intentional schema migration and fully rebuild affected vector data. Never mix 384-dimension Hugging Face vectors with 1536-dimension OpenAI vectors in the same configured collection.

## OpenAI-compatible provider rejects the request

### Symptom

Chat returns `502`, while logs show authentication, model-not-found, invalid base URL, or unsupported parameter errors.

### Likely cause

Provider/key/base URL are mixed, a registry ID lacks `provider:model`, the model name is invalid, or the compatible API does not support a supplied option.

### Diagnosis

Check `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, and requested `model_config_id`. Verify the same provider prefix for rewrite/reranker/judge models.

### Resolution

Use a consistent provider configuration and a supported model. For OpenRouter use its base URL and key. Restart the process after `.env` changes because settings are cached.

## Redis is unavailable

### Symptom

`/ready` reports Redis `unavailable`/`misconfigured`, cache misses increase, or rate limiting does not operate.

### Likely cause

`ENABLE_REDIS=true` without both Upstash REST values, invalid credentials/endpoint, or outbound network failure.

### Diagnosis

Confirm `UPSTASH_REDIS_REST_URL` uses HTTP(S), the token belongs to that database, and the process received both values. The local Redis container does not satisfy the Upstash REST client.

### Resolution

Correct Upstash settings or set `ENABLE_REDIS=false` to use no-op operational caching locally. Use `REDIS_URL=redis://localhost:6379/0` only for the separate legacy response-cache path.

## Langfuse tracing does not appear

### Symptom

Requests succeed but no Langfuse trace is visible.

### Likely cause

Tracing is disabled, sample rate excluded the request, wrong host/region/project keys, cached pre-change settings, or export connectivity failed.

### Diagnosis

Check startup logs, `ENABLE_LANGFUSE_OBSERVABILITY`, both keys, `LANGFUSE_BASE_URL`, environment filters, and sample rate. Inspect `chat_traces.observability_provider`/`external_trace_id` to distinguish internal persistence from external tracing.

### Resolution

Correct settings, restart/redeploy, use `LANGFUSE_SAMPLE_RATE=1.0` temporarily for diagnosis, and confirm outbound access. Rotate exposed keys rather than reusing them.

## MLflow or DagsHub authentication fails

### Symptom

An evaluation writes local artifacts but remote run logging raises a setup/authentication error.

### Likely cause

MLflow URI/credentials are wrong, DagsHub owner/repository/token are missing, or DagsHub was enabled without MLflow.

### Diagnosis

For DagsHub confirm both enable flags and `DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME`, and token. For ordinary MLflow confirm URI and optional username/password.

### Resolution

Set both flags for DagsHub. Do not set `MLFLOW_TRACKING_URI` as a substitute for DagsHub initialization. Re-run after authentication is fixed; retain local artifacts as evidence from the failed logging attempt.

## Evaluation dataset cannot be loaded

### Symptom

The runner reports invalid JSON, missing fields, duplicate IDs, boundary violations, or insufficient expected-source coverage.

### Likely cause

A JSONL row uses an invented schema, has a trailing/multiline object, references unknown sources, or lacks labels.

### Diagnosis

```bash
uv run python -m pytest tests/evals/test_portfolio_eval_dataset.py tests/evals/test_eval_dataset_boundaries.py
```

Compare rows with [evaluation](evaluation.md) and [evals/README.md](../evals/README.md).

### Resolution

Use one complete JSON object per line, stable unique IDs, actual source filenames, and the correct dataset-specific fields. Do not lower coverage validation to conceal unlabeled benchmark rows.

## CORS blocks the frontend

### Symptom

Browser requests fail with a CORS error while curl works.

### Likely cause

The exact scheme/host/port is absent from `FRONTEND_ORIGIN`, the API was not restarted, or the frontend calls a different backend URL.

### Diagnosis

Inspect the browser `Origin` header and compare it exactly with comma-separated configured origins. Check the preflight response and deployment environment.

### Resolution

Add the exact HTTPS production origin (and intentional preview origins), update the Modal secret, and redeploy. Wildcard preview patterns are not expanded by the implementation.

## Modal secrets are missing

### Symptom

Modal deploys but the function fails during import/startup or production validation names a missing value.

### Likely cause

The `production-chatbot-api-secrets` secret is absent from the deployment environment or lacks required production settings.

### Diagnosis

Inspect Modal function logs and secret attachment. Confirm `APP_ENV`, frontend origin, database URL, and model key, then feature-specific credentials.

### Resolution

Create/update the expected Modal secret, keep the name aligned with `modal_app.py`, and redeploy. Do not work around validation by committing production values.

## Rate limiting behaves unexpectedly

### Symptom

Requests are not limited, are limited in fixed bursts, or daily/concurrency values have no effect.

### Likely cause

Redis is disabled, `RATE_LIMIT_ENABLED=false`, actor keys differ, Upstash errors fail open, or granular `CHAT_RATE_LIMIT_*` settings are assumed to be active.

### Diagnosis

Check `ENABLE_REDIS`, Upstash connectivity, `RATE_LIMIT_ENABLED`, `RATE_LIMIT_MAX_REQUESTS`, and `RATE_LIMIT_WINDOW_SECONDS`. Review logs for failed checks.

### Resolution

Configure the primary fixed-window settings. The current concurrency/token/cost budget methods are no-ops; implement and test them before treating their parsed settings as policy controls.

## Cached answers appear stale

### Symptom

Chat returns an older answer after knowledge, prompt, model, or retrieval changes.

### Likely cause

An exact cache entry is still within its TTL or the knowledge-base version was not changed for a legacy cache.

### Diagnosis

Inspect response fields `response_cache_hit`, `response_cache_type`, and `response_cache_reason`. Check exact-cache TTL and which Redis path is enabled.

### Resolution

Temporarily disable/clear the relevant cache, reduce TTL during development, or increment `RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION` for the legacy path. Re-ingestion does not automatically purge every external cache key.

## Docker API cannot reach dependencies

### Symptom

Host commands work, but the Compose `api` service cannot connect to PostgreSQL, MinIO, or Redis.

### Likely cause

`.env` still uses host loopback addresses, which point back into the API container.

### Diagnosis

```bash
docker compose logs api
docker compose exec api env
```

### Resolution

Use `db:5432`, `minio:9000`, and `redis:6379` inside Compose. Restore host-facing values when returning to host Python mode.
