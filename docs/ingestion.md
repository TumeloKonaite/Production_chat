# Knowledge ingestion

"Knowledge ingestion" is the preferred term for validating, cleaning, chunking, embedding, and indexing approved portfolio content.

## Sources and formats

| Source | Supported formats | Location/storage |
|---|---|---|
| Curated repository knowledge | UTF-8 Markdown (`.md`) | `app/knowledge/source/` recursively |
| Uploaded knowledge | UTF-8 Markdown or text (`.md`, `.txt`) | MinIO, local filesystem, or Supabase Storage |

The upload API rejects empty files, unsupported extensions/content types, invalid UTF-8 at ingestion time, and files larger than `KNOWLEDGE_UPLOAD_MAX_BYTES` (10 MiB by default).

## Prerequisites

1. PostgreSQL is reachable and migrations are current.
2. pgvector is installed for vector retrieval.
3. `EMBEDDING_PROVIDER`, model, and dimension match the database/index configuration.
4. The embedding provider credential is configured when using OpenAI or OpenRouter.
5. The selected uploaded-file storage provider is reachable for uploaded sources.

## How chunking works

Ingestion normalizes Markdown whitespace, splits first on `#`, `##`, and `###` headings, then applies recursive character splitting. Defaults are `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=200`.

Each chunk stores:

- source and source type;
- resolved section heading;
- content;
- global and section-local chunk indexes;
- content type;
- source modification timestamp;
- uploaded-file ID, original filename, storage provider/bucket/path when applicable;
- database timestamps and generated chunk ID.

## Initial local ingestion

Start PostgreSQL, migrate, and run the synchronous command:

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python scripts/ingest_knowledge.py
```

Example output shape:

```text
Using chunk config: chunk_size=1000, chunk_overlap=200
Loaded 9 source documents
Ingested certifications.md: 1 chunks
...
Knowledge ingestion complete
```

The exact counts change with source content and chunk settings.

## Re-ingestion and duplicates

The synchronous command is safe to rerun for the same source names. For each source, the repository deletes previous `knowledge_chunks` rows and inserts the new chunks; the retrieval service replaces indexed vector entries. It does not append duplicate rows.

> Warning: vector replacement is collection-wide and is not atomic with embedding generation. The current service deletes/recreates the vector collection before requesting all replacement embeddings. If the provider call fails, `knowledge_chunks` can remain populated while the vector collection is empty. Always verify retrieval after ingestion and rerun the complete ingestion only after provider access is restored.

The asynchronous trigger path computes an idempotency key from:

- source type and identifier;
- source content checksum;
- embedding provider, model, and dimension;
- chunk size and overlap.

An active matching job is reused. A previously completed matching job creates a terminal `skipped` job. Change the source or relevant configuration to schedule new work.

`reset_existing_vectors=true` is required by the request schema when overriding embedding configuration, but the current worker's replacement behavior is source-based. Treat provider/dimension changes as a controlled full-index migration: update schema/configuration and re-ingest every approved source.

## Trigger local-directory ingestion as a job

The CLI trigger uses the configured `INGESTION_BACKEND` and prints the job ID:

```bash
uv run python scripts/trigger_ingestion.py --source-type local_directory
```

With `INGESTION_BACKEND=local`, the worker starts in a daemon thread. Because the CLI process exits immediately, use the synchronous `scripts/ingest_knowledge.py` command for reliable standalone local-directory ingestion. The local threaded backend is intended for triggers made by the running API process.

HTTP trigger:

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/ingest \
  -H "Content-Type: application/json" \
  -H "x-ingestion-secret: your-ingestion-secret" \
  -d '{"source_type":"local_directory"}'
```

The endpoint returns `202` and `status: queued`, or `200` and `status: skipped`. `INGESTION_API_SECRET` must be non-empty and match the header.

## Upload and ingest a file

Start MinIO when using the default storage provider:

```bash
docker compose up -d minio
```

Upload:

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/files \
  -F "file=@path/to/approved-knowledge.md;type=text/markdown"
```

Copy the returned `id`, then trigger it:

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/ingest \
  -H "Content-Type: application/json" \
  -H "x-ingestion-secret: your-ingestion-secret" \
  -d '{"source_type":"uploaded_file","file_id":"00000000-0000-0000-0000-000000000000"}'
```

Or use the trigger script:

```bash
uv run python scripts/trigger_ingestion.py \
  --source-type uploaded_file \
  --file-id 00000000-0000-0000-0000-000000000000
```

An uploaded file can be re-ingested from `uploaded`, `failed`, or `ingested` status. `ingesting` conflicts; `deleted` is gone. The current public API does not expose file deletion or ingestion-job status endpoints, so inspect the `knowledge_files` and `knowledge_ingestion_jobs` tables for job operation.

## Embedding experiments and forced re-indexing

The trigger accepts a complete override only:

```bash
uv run python scripts/trigger_ingestion.py \
  --source-type local_directory \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --embedding-dimension 1536 \
  --reset-existing-vectors
```

Do not run this against a 384-dimension database/index. Use the dedicated experiment scripts and an isolated database when comparing dimensions:

```bash
uv run python scripts/run_embedding_experiment.py --help
uv run python scripts/run_chunking_experiment.py --help
```

## Verify ingestion

Count stored chunks with PostgreSQL in Compose:

```bash
docker compose exec db psql -U postgres -d production_chatbot \
  -c "SELECT source, count(*) AS chunks FROM knowledge_chunks GROUP BY source ORDER BY source;"
```

Then run retrieval evaluation as an end-to-end index check:

```bash
uv run python -m evals.runners.run_retrieval_eval \
  --config evals/configs/retrieval_baseline.json
```

Finally start the API and ask a portfolio question. An empty retrieval result despite populated `knowledge_chunks` usually indicates vector collection, embedding, or similarity-threshold mismatch; see [troubleshooting](troubleshooting.md).

## Safe production ingestion

- Store `INGESTION_API_SECRET` and provider keys in the Modal secret.
- Use `INGESTION_BACKEND=modal` so the API spawns the deployed worker.
- Apply migrations before triggering jobs.
- Verify database, collection, storage provider, and embedding dimension before re-indexing.
- Upload only approved, non-sensitive content; chunks and trace previews may be observable.
- Rotate the ingestion secret after suspected exposure.
- Test embedding/chunk changes against an isolated database and eval suite first.

See [deployment](deployment.md) for Modal commands and [configuration](configuration.md) for provider settings.
