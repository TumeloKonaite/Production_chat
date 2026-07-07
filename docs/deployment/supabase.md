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
