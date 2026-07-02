# Production Chatbot

Simple FastAPI backend for a personal website chatbot. The frontend sends a message to `POST /chat`, the backend calls the configured LLM with a server-side API key, stores production chat metadata in PostgreSQL, and can run offline model comparisons with MLflow-backed eval artifacts.

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
  datasets/
    personal_chatbot_eval_set.jsonl
  results/
  run_model_eval.py
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
TAVUS_API_KEY=
TAVUS_BASE_URL=https://tavusapi.com
TAVUS_FACE_ID=
TAVUS_PAL_ID=
PUBLIC_BACKEND_URL=
TAVUS_TOOL_SECRET=
INGESTION_API_SECRET=
DEFAULT_MODEL_CONFIG_ID=openai:gpt-4.1-mini
OPENAI_MODEL=gpt-4.1-mini
DEFAULT_PROMPT_VERSION=v1_professional
CONVERSATION_HISTORY_LIMIT=10
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SIMILARITY=0.55
DEFAULT_RETRIEVAL_CONFIG=default
ENABLE_MLFLOW_TRACKING=false
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=personal-chatbot-model-comparison
```

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

### Run ingestion from the CLI

```bash
uv run python .\scripts\ingest_knowledge.py
```

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

Run the model comparison workflow:

```bash
mlflow server --host 0.0.0.0 --port 5000
python evals/run_model_eval.py \
  --models openai:gpt-4.1-mini openai:gpt-4.1 \
  --prompt-version v1_professional \
  --dataset evals/datasets/personal_chatbot_eval_set.jsonl \
  --experiment-name personal-chatbot-model-comparison
```

Artifacts are written under `evals/results/`. When `ENABLE_MLFLOW_TRACKING=true` and `MLFLOW_TRACKING_URI` is reachable, the runner logs one MLflow run per model plus JSON and summary artifacts.

## Tests

```bash
pytest
```
