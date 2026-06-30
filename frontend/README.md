# Frontend Test App

Local Tavus test UI for this FastAPI backend.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs at `http://localhost:5173`.

## Environment

Create `frontend/.env` from `.env.example` and set:

```env
VITE_BACKEND_URL=http://localhost:8000
```

The frontend only calls the backend Tavus endpoints:

- `POST /api/tavus/conversations`
- `POST /api/tavus/conversations/end`

It does not use the Tavus API key directly.

## Local Tavus callback note

When testing Tavus against a local backend, the backend `.env` should set:

```env
PUBLIC_BACKEND_URL=<ngrok-or-cloudflare-url>
```

Tavus must call the public backend URL, not `localhost`.
