# Production Chatbot

Simple FastAPI backend for a personal website chatbot. The frontend sends a message to `POST /chat`, the backend calls the LLM with a server-side API key, stores the conversation in PostgreSQL, and returns a JSON assistant reply.

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
    prompts/
      prompt_loader.py
      templates/
        v1_professional.md
        v2_warm_conversational.md
  repositories/
    chat_repository.py
    db/
      base.py
      models.py
      session.py
  services/
    chat/
      errors.py
      service.py
    llm/
      errors.py
      service.py
    chat/
      prompting.py
evals/
  prompt_eval_questions.jsonl
scripts/
  compare_prompts.py
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
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5433/production_chatbot
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DEFAULT_PROMPT_VERSION=v1_professional
CONVERSATION_HISTORY_LIMIT=10
RETRIEVAL_TOP_K=5
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=portfolio-chatbot-prompt-experiments
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

If you want the local Postgres instance from `docker-compose.yml`, start it first so the database is listening on `127.0.0.1:5433`:

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
  "prompt_version": "v2_warm_conversational",
  "latency_ms": 842,
  "token_usage": {
    "input_tokens": 1200,
    "output_tokens": 180,
    "total_tokens": 1380
  }
}
```

## Prompt experiments

Runtime prompt selection loads versioned templates from `app/infrastructure/prompts/templates`.
If `prompt_version` is omitted in the API request, the backend falls back to `DEFAULT_PROMPT_VERSION`.

Run the prompt comparison workflow:

```bash
mlflow ui
python scripts/compare_prompts.py
```

Artifacts are written under `evals/prompt_eval_results/` and logged to the MLflow experiment named by `MLFLOW_EXPERIMENT_NAME`.

## Tests

```bash
pytest
```
