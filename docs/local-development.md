# Local development

## Supported modes

1. **Host Python with Dockerized dependencies (recommended).** Run FastAPI and commands through `uv`; run PostgreSQL/pgvector, Redis, and MinIO with Compose.
2. **Fully Dockerized.** The Compose API service works after container-facing service URLs are configured. Migrations and ingestion remain explicit steps.
3. **Managed services.** Replace local URLs with managed PostgreSQL/pgvector, storage, and optional Upstash values. Avoid pointing development commands at production data.

The frontend is a separate project. This repository only configures its allowed origin.

## Recommended setup

### 1. Clone the repository

```bash
git clone https://github.com/TumeloKonaite/Production_chat.git
cd Production_chat
```

### 2. Install Python dependencies

Python 3.12 is required. The lockfile and CI use `uv`:

```bash
uv sync --locked
```

You normally do not need to activate the environment because `uv run` uses `.venv` directly. If activation is useful:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Configure the application

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Add `LLM_API_KEY` for the configured provider. The defaults use local PostgreSQL, Hugging Face embeddings, and MinIO. Read [configuration](configuration.md) before changing embedding provider or dimension.

### 4. Start dependencies

```bash
docker compose up -d db redis minio
docker compose ps
```

Only PostgreSQL is necessary for the built-in directory ingestion and basic chat. MinIO is necessary for uploaded-file workflows. The local Redis container supports the optional Redis-protocol response cache; the primary operational cache/rate limiter expects Upstash REST credentials and does not use this container.

Optional pgAdmin:

```bash
docker compose up -d pgadmin
```

Open <http://127.0.0.1:5051>. Register the database host as `db`, port `5432`, when connecting from pgAdmin's container.

### 5. Apply migrations

```bash
uv run alembic upgrade head
uv run alembic current
```

The migration chain creates/enables pgvector where required. `DATABASE_DIRECT_URL` takes precedence for migrations.

### 6. Ingest portfolio knowledge

```bash
uv run python scripts/ingest_knowledge.py
```

Expected output lists the active chunk settings, loaded source count, each source's chunk count, and `Knowledge ingestion complete`.

### 7. Start FastAPI

```bash
uv run uvicorn main:app --reload
```

The application listens on <http://127.0.0.1:8000>. API documentation is at <http://127.0.0.1:8000/docs>.

### 8. Verify health and readiness

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`/health` proves the process is serving. `/ready` also checks PostgreSQL and checks Upstash only when `ENABLE_REDIS=true`.

### 9. Verify routing and generation

Portfolio question:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What projects has Tumelo built?"}'
```

Deterministic out-of-scope response (no model call):

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the weather today?"}'
```

Windows PowerShell equivalent:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat `
  -ContentType 'application/json' `
  -Body '{"message":"What projects has Tumelo built?"}'
```

## Fully Dockerized mode

Copy `.env.example`, then change host URLs to Compose service names:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/production_chatbot
MINIO_ENDPOINT=http://minio:9000
REDIS_URL=redis://redis:6379/0
```

Build and start:

```bash
docker compose up -d --build
docker compose exec api uv run alembic upgrade head
docker compose exec api uv run python scripts/ingest_knowledge.py
curl http://127.0.0.1:8000/ready
```

The API image runs `uvicorn main:app` without reload. Source edits require a rebuild; no source volume is mounted. The image does not apply migrations or ingest automatically.

To return to host mode, restore `127.0.0.1:5434`, `localhost:9000`, and `localhost:6379` in `.env`.

## Managed-service mode

Set the relevant database, vector-store, storage, Redis, and provider values from [configuration](configuration.md). Recommended safety practices:

- use an isolated development database/project;
- keep production secrets out of `.env` and shell history;
- set `APP_ENV=local` for local runs;
- verify `DATABASE_URL` before migrations or ingestion;
- use `DATABASE_DIRECT_URL` for migrations when a transaction pooler is unsuitable;
- re-ingest only after confirming the target collection and embedding dimension.

## Tests and quality checks

Run CI-equivalent checks:

```bash
uv run ruff check .
uv run python -m pytest
```

Run a focused test:

```bash
uv run python -m pytest tests/test_chat_routing.py
```

Database integration tests are skipped by default. Use a disposable database:

Linux/macOS:

```bash
RUN_DB_INTEGRATION_TESTS=true TEST_DATABASE_URL='postgresql+psycopg://user:password@host/database' uv run python -m pytest tests/integration
```

Windows PowerShell:

```powershell
$env:RUN_DB_INTEGRATION_TESTS='true'
$env:TEST_DATABASE_URL='postgresql+psycopg://user:password@host/database'
uv run python -m pytest tests/integration
```

## Frontend integration

Set the exact development origin, including port:

```env
FRONTEND_ORIGIN=http://localhost:5173
```

Multiple origins are comma-separated. The frontend sends `POST /chat`; `/api/chat` remains an undocumented compatibility alias. This repository has no frontend install or start command.

## Stop and reset

Stop containers while preserving data:

```bash
docker compose stop
```

Remove containers but preserve named volumes:

```bash
docker compose down
```

Reset local PostgreSQL, Redis, MinIO, and pgAdmin data:

```bash
docker compose down --volumes
```

> Warning: `--volumes` permanently deletes all data stored in this Compose project. Run migrations and ingestion again afterward.

To clear only Python/test caches, use the normal tool commands rather than deleting `.venv`. Re-run `uv sync --locked` when the lockfile changes.

## Operating-system notes

- PowerShell uses `Copy-Item` rather than `cp` in the documented Windows path and `Invoke-RestMethod` for JSON requests.
- Linux Docker installations may require membership in the `docker` group or `sudo`, depending on host policy.
- If PowerShell blocks activation, continue using `uv run`; activation is optional.
- Set a workspace-local `UV_CACHE_DIR` if the global uv cache is not writable.

See [troubleshooting](troubleshooting.md) for common failures.
