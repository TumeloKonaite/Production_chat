# Configuration reference

The application loads `.env` through `python-dotenv`; process environment values take precedence. Copy `.env.example` to `.env` for local development. Keep secrets empty in committed files and store production values in Modal/GitHub secret stores.

Definitions used below:

- **Local** means needed for the standard local workflow.
- **Conditional** means required only when its feature/provider is enabled.
- **Production** means enforced when `APP_ENV=production`.
- An empty default means no value is configured.

## Minimal local configuration

The checked-in defaults already target Docker Compose PostgreSQL, local Hugging Face embeddings, and MinIO. Add a model key for generated answers:

```env
APP_ENV=local
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5434/production_chatbot
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-openai-api-key
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
KNOWLEDGE_EMBEDDING_DIMENSION=384
```

`LLM_API_KEY` is not required for `/health`, migrations, or default Hugging Face ingestion. It is required when a request reaches an LLM. Direct greetings and out-of-scope responses do not call a model.

## Application and CORS

| Variable | Required | Default | Purpose | Example |
|---|---:|---|---|---|
| `APP_ENV` | Local | `local` | `local`, `test`, or `production`; activates production validation | `local` |
| `FRONTEND_ORIGIN` | Production | local defaults to `http://localhost:5173` | Comma-separated exact CORS origins | `https://portfolio.example.com` |

CORS allows `GET`, `POST`, and `OPTIONS`, permits all request headers, and disables credentials. A production startup fails if `FRONTEND_ORIGIN` is absent.

## PostgreSQL and pgvector

| Variable | Required | Default | Purpose | Example |
|---|---:|---|---|---|
| `DATABASE_URL` | Production | local Docker URL on port `5434` | Runtime SQLAlchemy URL | `postgresql+psycopg://user:password@host:6543/postgres?sslmode=require` |
| `DATABASE_DIRECT_URL` | Optional | empty | Direct/session URL for migrations and administrative ingestion | `postgresql+psycopg://user:password@host:5432/postgres?sslmode=require` |
| `VECTOR_STORE_PROVIDER` | Local | `pgvector` | `pgvector` or `supabase_pgvector` | `pgvector` |
| `KNOWLEDGE_COLLECTION_NAME` | Local | `personal_knowledge_base` | Vector collection and cache-version identity | `personal_knowledge_base` |

When `DATABASE_DIRECT_URL` is empty, migrations use `DATABASE_URL`. `supabase_pgvector` additionally requires `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

Docker Compose consumes these bootstrap-only variables; application settings do not:

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_DB` | `production_chatbot` | Database created by the container |
| `POSTGRES_USER` | `postgres` | Local database user |
| `POSTGRES_PASSWORD` | `postgres` | Local-only database password |
| `PGADMIN_DEFAULT_EMAIL` | `admin@local.dev` | pgAdmin local login |
| `PGADMIN_DEFAULT_PASSWORD` | `admin` | pgAdmin local login |
| `PGADMIN_PORT` | `5051` | Host port bound to loopback |

## Chat model providers

| Variable | Required | Default | Purpose | Example |
|---|---:|---|---|---|
| `LLM_PROVIDER` | Local | `openai` | `openai` or `openrouter` | `openrouter` |
| `LLM_MODEL` | Local | `gpt-4.1-mini` | Active provider model name | `meta-llama/llama-3.1-70b-instruct` |
| `LLM_BASE_URL` | Local | provider default | Active OpenAI-compatible API base | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Production/generated answers | empty | Active provider key; preferred generic setting | `your-provider-key` |
| `LLM_PROMPT_COST_PER_1M_TOKENS` | Optional | empty | Manual prompt/input cost override | `0.40` |
| `LLM_COMPLETION_COST_PER_1M_TOKENS` | Optional | empty | Manual completion/output cost override | `1.60` |
| `DEFAULT_MODEL_CONFIG_ID` | Optional | derived from provider/model | Default registry ID in `provider:model` form | `openai:gpt-4.1-mini` |
| `MODEL_CONFIGS_JSON` | Optional | empty | JSON object defining additional model configurations | `{"openai:custom":{"provider":"openai","model":"custom"}}` |

Compatibility/provider-specific variables remain supported:

| Variable | Default | Behavior |
|---|---|---|
| `OPENAI_API_KEY` | empty | Used when OpenAI is active and `LLM_API_KEY` is empty |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI default unless `LLM_BASE_URL` overrides it |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Legacy model fallback |
| `OPENROUTER_API_KEY` | empty | Used when OpenRouter is active and `LLM_API_KEY` is empty |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter default unless `LLM_BASE_URL` overrides it |

Do not mix a provider with another provider's base URL/key. All registry IDs use `provider:model`.

## Embeddings and chunking

| Variable | Required | Default | Purpose | Example |
|---|---:|---|---|---|
| `EMBEDDING_PROVIDER` | Local | `hf` | `hf`, `openai`, or `openrouter` | `openai` |
| `KNOWLEDGE_EMBEDDING_MODEL` | Local | `all-MiniLM-L6-v2` | Embedding model | `text-embedding-3-small` |
| `KNOWLEDGE_EMBEDDING_DIMENSION` | Local | `384` | Persisted vector dimension | `1536` |
| `CHUNK_SIZE` | Local | `1000` | Recursive character chunk size | `800` |
| `CHUNK_OVERLAP` | Local | `200` | Character overlap; must be smaller than size | `120` |

`EMBEDDING_DIMENSION` is a supported legacy alias for `KNOWLEDGE_EMBEDDING_DIMENSION`. Provider, model, dimension, database vector column, and stored index must agree. OpenAI/OpenRouter embeddings require the corresponding provider credential.

## Retrieval, prompts, rewriting, and reranking

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `DEFAULT_PROMPT_VERSION` | Local | `v1_professional` | Prompt template selected by default |
| `CONVERSATION_HISTORY_LIMIT` | Local | `10` | Recent messages used for context resolution |
| `RETRIEVER_TYPE` | Local | `vector` | `vector`, `keyword`, or `hybrid` |
| `RETRIEVAL_TOP_K` | Local | `5` | Final number of retrieved chunks |
| `RETRIEVAL_MIN_SIMILARITY` | Local | `0.55` | Minimum vector similarity |
| `DEFAULT_RETRIEVAL_CONFIG` | Local | `default` | Label logged with responses/evaluations |
| `ENABLE_QUERY_REWRITING` | Optional | `false` | Enable LLM retrieval-query rewriting |
| `QUERY_REWRITE_MODEL` | Conditional | `openai:gpt-4.1-mini` | Registry ID for rewriting |
| `QUERY_REWRITE_TEMPERATURE` | Optional | `0.0` | Rewrite sampling temperature |
| `QUERY_REWRITE_PROMPT_VERSION` | Optional | `v1` | Rewrite prompt label |
| `QUERY_REWRITE_TIMEOUT_SECONDS` | Optional | `10` | Rewrite timeout |
| `QUERY_REWRITE_MAX_TOKENS` | Optional | `128` | Rewrite output cap |
| `ENABLE_RERANKING` | Optional | `false` | Enable post-retrieval reranking |
| `RERANKER_TYPE` | Conditional | `none` | `none` or `llm` |
| `RERANKER_MODEL` | Conditional | `openai:gpt-4.1-mini` | Registry ID for LLM reranking |
| `RERANKER_INITIAL_TOP_K` | Optional | `20` | Candidates before reranking |
| `RERANKER_FINAL_TOP_K` | Optional | `5` | Results after reranking |

`PROMPT_VERSION` is a legacy alias used only when `DEFAULT_PROMPT_VERSION` is absent.

## Uploaded-file storage

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `STORAGE_PROVIDER` | Local | `minio` | `local`, `minio`, or `supabase` |
| `MINIO_ENDPOINT` | MinIO | `http://localhost:9000` | S3-compatible endpoint |
| `MINIO_ACCESS_KEY` | MinIO | `minioadmin` | Local development access key |
| `MINIO_SECRET_KEY` | MinIO | `minioadmin` | Local development secret |
| `MINIO_BUCKET` | MinIO | `knowledge-files` | Uploaded-file bucket |
| `MINIO_SECURE` | MinIO | `false` | Use TLS for MinIO |
| `LOCAL_STORAGE_PATH` | Local storage | `.storage` | Root for filesystem uploads |
| `KNOWLEDGE_UPLOAD_MAX_BYTES` | Local | `10485760` | Upload size limit (10 MiB) |
| `SUPABASE_URL` | Supabase | empty | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase | empty | Server-only service role credential |
| `SUPABASE_STORAGE_BUCKET` | Supabase | `knowledge-files` | Supabase Storage bucket |

Never expose the service-role key to a browser. Uploaded file metadata remains in PostgreSQL regardless of byte-storage provider.

## Ingestion and protected APIs

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `INGESTION_API_SECRET` | HTTP ingestion | empty | Value expected in `x-ingestion-secret` |
| `INGESTION_BACKEND` | Local | `local` | `local` daemon thread or `modal` worker |
| `MODAL_INGESTION_APP_NAME` | Modal backend | `production-chatbot-api` | Deployed Modal app lookup name |
| `MODAL_INGESTION_FUNCTION_NAME` | Modal backend | `run_ingestion_job` | Worker function lookup name |
| `EVAL_ADMIN_TOKEN` | Evaluation HTTP APIs | empty | Bearer token protecting evaluation endpoints |

The synchronous `scripts/ingest_knowledge.py` command does not use `INGESTION_API_SECRET`.

## Operational Redis cache, lock, and rate limit

The current chat dependency path uses Upstash Redis REST:

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `ENABLE_REDIS` | Optional | `false` | Activate the Upstash REST client and readiness check |
| `UPSTASH_REDIS_REST_URL` | Conditional | empty | Upstash REST endpoint |
| `UPSTASH_REDIS_REST_TOKEN` | Conditional | empty | Upstash REST bearer token |
| `RATE_LIMIT_ENABLED` | Optional | `true` | Fixed-window limiter (only active with Redis) |
| `RATE_LIMIT_MAX_REQUESTS` | Optional | `20` | Requests allowed per window |
| `RATE_LIMIT_WINDOW_SECONDS` | Optional | `60` | Fixed-window size |
| `EXACT_CACHE_ENABLED` | Optional | `true` | Exact chat-response cache |
| `EXACT_CACHE_TTL_SECONDS` | Optional | `300` | Exact cache TTL |
| `REQUEST_LOCK_ENABLED` | Optional | `true` | Duplicate in-flight request lock |
| `REQUEST_LOCK_TTL_SECONDS` | Optional | `30` | Lock TTL |

Both Upstash values are required when `ENABLE_REDIS=true`. If an Upstash rate-limit operation fails, the current limiter logs and allows the request.

## Legacy response-cache configuration

These settings configure the separate Redis response-cache service:

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `ENABLE_RESPONSE_CACHE` | Optional | `false` | Enable the response-cache provider |
| `RESPONSE_CACHE_PROVIDER` | Conditional | `redis` | Only `redis` is supported |
| `REDIS_URL` | Conditional | local `redis://localhost:6379/0` | Redis protocol URL |
| `REDIS_TOKEN` | Optional | empty | Injected as URL password when needed |
| `ENABLE_EXACT_RESPONSE_CACHE` | Optional | `true` | Legacy exact lookup |
| `ENABLE_SEMANTIC_RESPONSE_CACHE` | Optional | `false` | Legacy semantic lookup |
| `RESPONSE_CACHE_TTL_SECONDS` | Optional | `604800` | Entry TTL |
| `RESPONSE_CACHE_EXACT_PREFIX` | Optional | `chat:exact` | Exact key prefix |
| `RESPONSE_CACHE_SEMANTIC_INDEX` | Optional | `chat_semantic_cache` | Semantic index name |
| `RESPONSE_CACHE_DISTANCE_THRESHOLD` | Optional | `0.10` | Maximum semantic distance (`0`–`2`) |
| `RESPONSE_CACHE_MAX_RESULTS` | Optional | `3` | Semantic candidates |
| `RESPONSE_CACHE_STORE_PRIVATE_SESSIONS` | Optional | `false` | Permit private-session storage |
| `RESPONSE_CACHE_KNOWLEDGE_BASE_VERSION` | Optional | collection name | Cache invalidation/version label |

`ENABLE_EXACT_RESPONSE_CACHE` is also a legacy alias for the primary `EXACT_CACHE_ENABLED` flag when the latter is absent.

The granular `ENABLE_RATE_LIMITING`, `RATE_LIMITING_FAIL_OPEN`, and `CHAT_RATE_LIMIT_*` settings are parsed for compatibility, but the current service enforces only `RATE_LIMIT_ENABLED`, `RATE_LIMIT_MAX_REQUESTS`, and `RATE_LIMIT_WINDOW_SECONDS`. Concurrency and token/cost budget methods are currently no-ops; do not rely on the granular values for production policy.

## Langfuse

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `ENABLE_LANGFUSE_OBSERVABILITY` | Optional | `false` | Enable live request traces |
| `LANGFUSE_PUBLIC_KEY` | Conditional | empty | Project public key |
| `LANGFUSE_SECRET_KEY` | Conditional | empty | Project secret key |
| `LANGFUSE_BASE_URL` | Optional | `https://cloud.langfuse.com` | Langfuse host |
| `LANGFUSE_ENVIRONMENT` | Optional | current `APP_ENV` | Environment tag |
| `LANGFUSE_RELEASE` | Optional | empty | Release tag |
| `LANGFUSE_SAMPLE_RATE` | Optional | `1.0` | Trace fraction from `0` to `1` |
| `LANGFUSE_EXPORT_DEFAULT_LIMIT` | Optional | `100` | Default bad-trace export limit |

`ENABLE_LANGFUSE` is a legacy alias. If tracing is enabled, both keys are required at startup.

## MLflow and DagsHub

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `ENABLE_MLFLOW_TRACKING` | Optional | `false` | Activate the MLflow logging API |
| `MLFLOW_TRACKING_URI` | Local/remote MLflow | empty | Tracking server or local file URI |
| `MLFLOW_TRACKING_USERNAME` | Conditional | empty | Tracking basic-auth username |
| `MLFLOW_TRACKING_PASSWORD` | Conditional | empty | Tracking basic-auth password |
| `MLFLOW_EXPERIMENT_NAME` | Optional | `personal-chatbot-model-comparison` | Default experiment |
| `ENABLE_DAGSHUB_TRACKING` | Optional | `false` | Initialize DagsHub's MLflow integration |
| `DAGSHUB_REPO_OWNER` | DagsHub | empty | DagsHub owner |
| `DAGSHUB_REPO_NAME` | DagsHub | empty | DagsHub repository |
| `DAGSHUB_TOKEN` | Optional | empty | Copied to `DAGSHUB_USER_TOKEN` when needed |

DagsHub does not use `MLFLOW_TRACKING_URI` in this implementation. With `ENABLE_DAGSHUB_TRACKING=true`, `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)` selects the backend. `ENABLE_MLFLOW_TRACKING` must also be true because MLflow remains the logging API.

## Evaluation, feedback, and tests

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `ENABLE_PRODUCTION_FEEDBACK_EXPORT` | Optional | `false` | Permit feedback export from production data |
| `ALLOW_RAW_PRODUCTION_TEXT_IN_EVALS` | Optional | `false` | Permit raw production text in exports; privacy-sensitive |
| `RUN_DB_INTEGRATION_TESTS` | Optional | `false` | Opt into live database integration tests |
| `TEST_DATABASE_URL` | Conditional | empty | Isolated database used by integration tests |

## Production requirements

At minimum, a production Modal secret must provide:

```env
APP_ENV=production
FRONTEND_ORIGIN=https://your-frontend.example.com
DATABASE_URL=postgresql+psycopg://user:password@pooler-host:6543/postgres?sslmode=require
DATABASE_DIRECT_URL=postgresql+psycopg://user:password@direct-host:5432/postgres?sslmode=require
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-provider-key
INGESTION_API_SECRET=generate-a-long-random-secret
INGESTION_BACKEND=modal
```

Add only the optional provider credentials you use. See [deployment](deployment.md) and keep `.env.example` as the canonical variable-name inventory.
