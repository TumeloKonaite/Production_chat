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
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_PROMPT_VERSION=v1_professional
CONVERSATION_HISTORY_LIMIT=10
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SIMILARITY=0.55
DEFAULT_RETRIEVAL_CONFIG=default
ENABLE_MLFLOW_TRACKING=false
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
ENABLE_DAGSHUB_TRACKING=false
DAGSHUB_REPO_OWNER=
DAGSHUB_REPO_NAME=
DAGSHUB_TOKEN=
```

`OPENAI_MODEL` is still supported as a legacy fallback, but `DEFAULT_MODEL_CONFIG_ID` is the preferred way to select the runtime model.

## Local setup

Install dependencies:

```bash
uv sync
```

Run the database migration:

```bash
alembic upgrade head
```

If you want the local Postgres instance from `docker-compose.yml`, start it first so the database is listening on `127.0.0.1:5434`:

```bash
docker compose up -d db
```

If you also want pgAdmin for local database inspection, start both services:

```bash
docker compose up -d db pgadmin
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

## Docker Compose

```bash
docker compose up --build
```

`docker compose up -d` starts the full local stack, including `db` and `pgadmin`. If you only want the database and admin UI, use `docker compose up -d db pgadmin`. The pgAdmin data directory is persisted in the `pgadmin_data` volume so saved server registrations survive container restarts.

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

The markdown knowledge source of truth lives under `app/knowledge/source/`. Both the local CLI script and the protected admin API call the same backend ingestion service.

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
EMBEDDING_DIMENSION=384
```

If you change the embedding provider, model, or dimension, rebuild the knowledge index before running retrieval again. Retrieval now validates the stored index metadata and fails loudly when the active embedding config does not match the indexed config.

### Run ingestion from the CLI

```bash
uv run python .\scripts\ingest_knowledge.py
```

The CLI prints the chunking config used for that ingestion run.

### Trigger ingestion from the backend

Set an ingestion secret in `.env`:

```env
INGESTION_API_SECRET=dev-secret-change-me
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
  "status": "ok",
  "documents_loaded": 9,
  "results": [
    {
      "source": "profile.md",
      "chunk_count": 3
    }
  ]
}
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
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL_CONFIG_ID=openai:gpt-4.1-mini
```

The built-in model configs use `provider: "openai"` and continue to call OpenAI directly by default.
To evaluate OpenRouter-backed models in the same backend, add model configs with `provider: "openrouter"`:

```env
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

`LLM_BASE_URL` is still accepted as a fallback alias for `OPENAI_BASE_URL`, but only for the OpenAI provider path. OpenRouter uses `OPENROUTER_BASE_URL`.

### Embedding provider configuration

Retrieval defaults to the local Hugging Face embedding path:

```env
EMBEDDING_PROVIDER=hf
KNOWLEDGE_EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

OpenRouter-compatible embedding experiments can use:

```env
EMBEDDING_PROVIDER=openrouter
KNOWLEDGE_EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSION=1536
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

`EMBEDDING_DIMENSION` must match both the provider output and the pgvector storage dimension. If you change provider, model, or dimension, rebuild the knowledge index. If you change to a new vector dimension, you may also need a database migration before ingestion can succeed.

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

### Local-only MLflow tracking

Set:

```env
ENABLE_MLFLOW_TRACKING=true
ENABLE_DAGSHUB_TRACKING=false
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

When `MLFLOW_TRACKING_URI` is blank, MLflow uses the local file-backed store and writes run metadata under `mlruns/` and artifacts under `mlartifacts/`.
Those directories are local-only outputs and are gitignored.

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

The model comparison workflow:

```bash
python evals/run_model_eval.py \
  --models openai:gpt-4.1-mini openai:gpt-4.1 \
  --prompt-version v1_professional \
  --dataset evals/datasets/model_eval_dataset.jsonl \
  --experiment-name personal-chatbot-model-comparison
```

The RAG evaluation workflow:

```bash
python evals/run_rag_eval.py \
  --model openai:gpt-4.1-mini \
  --prompt-version v1_professional \
  --run-name portfolio-rag-eval
```

The canonical RAG benchmark contract is documented in `evals/README.md`.

The retrieval-only baseline workflow:

```bash
python evals/run_retrieval_eval.py --k 5
```

The retrieval artifact payload includes the embedding provider, embedding model, embedding dimension, and chunking config used for that run.

To compare multiple retrieval configurations in one command, use the sweep runner with a YAML config:

```bash
python evals/run_retrieval_sweep.py --config evals/configs/retrieval_sweep.yaml
```

The protected backend exposes the same batch pattern over HTTP:

```bash
curl -X POST http://localhost:8000/api/evals/retrieval-sweeps \
  -H "Content-Type: application/json" \
  -H "x-eval-admin-token: <EVAL_ADMIN_TOKEN>" \
  -d '{
    "experiments": [
      {"name": "retrieval-vector-k3", "retriever_type": "vector", "top_k": 3},
      {"name": "retrieval-keyword-k5", "retriever_type": "keyword", "top_k": 5}
    ]
  }'
```

The sample sweep config defines one run per experiment:

```yaml
experiments:
  - name: retrieval-vector-k3
    retriever_type: vector
    top_k: 3

  - name: retrieval-vector-k5
    retriever_type: vector
    top_k: 5

  - name: retrieval-vector-k10
    retriever_type: vector
    top_k: 10

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

Each sweep experiment logs as its own MLflow or DagsHub-backed MLflow run and records:

- `retriever_type`
- `top_k`
- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `dataset_path`
- `git_commit_sha`
- retrieval metrics including `mrr`, `recall_at_k`, `mean_precision_at_k`, `hit_at_k`, and query counts

Sweep artifacts are written under `evals/results/retrieval_sweeps/`. Each run gets its own artifact directory, and the sweep root also includes a manifest plus comparison JSON and CSV outputs.

To compare multiple chunking strategies without editing code:

```bash
python scripts/run_chunking_experiment.py --configs "300:50,500:100,800:150,1000:200"
```

This re-ingests the knowledge base for each chunk config, runs retrieval eval, writes per-run artifacts, and saves comparison outputs under `evals/results/chunking_experiments/`.

The prompt comparison workflow:

```bash
python scripts/compare_prompts.py \
  --prompt-version v1_professional \
  --prompt-version v2_warm_conversational \
  --experiment-name personal-chatbot-model-comparison
```

Artifacts are written under `evals/results/` and `evals/prompt_eval_results/`.
When `ENABLE_MLFLOW_TRACKING=true`, the runners log one MLflow run per evaluated unit plus the generated JSON and summary artifacts.

## Tests

```bash
pytest
```

Targeted eval checks for the retrieval runners:

```bash
PYTHONPATH=. pytest tests/evals/test_retrieval_eval_runner.py tests/evals/test_retrieval_sweep_runner.py tests/test_embedding_experiment_script.py
```
