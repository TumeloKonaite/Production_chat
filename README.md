# Production Chatbot

Simple FastAPI backend for a personal website chatbot. The frontend sends a message to `POST /chat`, the backend calls the configured LLM with a server-side API key, stores production chat metadata in PostgreSQL, and can run offline model comparisons with MLflow-backed eval artifacts.

## Project structure

```text
app/
  main.py
  config.py
  api/
    chat.py
    dependencies/
      chat_dependencies.py
      common_dependencies.py
    schema.py
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
