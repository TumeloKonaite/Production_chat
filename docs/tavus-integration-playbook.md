# Tavus Integration Playbook

## Purpose

This document explains exactly how Tavus was introduced into this FastAPI chatbot project so the same pattern can be reproduced in other projects that already have a text-only backend.

The core architectural rule is:

> Tavus is the video and voice interface layer.  
> Your backend remains the source of truth for retrieval, prompting, model selection, logging, evaluation, and factual control.

This is the pattern you want to reuse.

---

## 1. Integration strategy

### 1.1 What Tavus should do

Tavus should handle:

- video avatar rendering
- voice input/output
- live conversation UI
- turn-taking
- persona / PAL hosting

### 1.2 What your backend should keep doing

Your backend should continue to own:

- knowledge base retrieval
- RAG orchestration
- LLM prompting
- prompt versioning
- model configuration
- response persistence
- telemetry and evaluation
- access control around Tavus credentials

### 1.3 Why this pattern matters

If you move answer generation into Tavus, you lose the structure and observability you already built for text chat. The reusable pattern is therefore:

```text
Frontend -> your backend -> Tavus conversation creation
Tavus -> your backend tool endpoint -> existing chat service
Backend -> Tavus -> spoken answer
```

That is the exact pattern implemented in this repository.

---

## 2. Minimum backend capabilities your old project must already have

Before reusing this pattern in another project, the existing backend should already have:

- a text chat or answer-generation service
- a place to inject metadata such as channel/source
- environment-based configuration
- HTTP routing for new Tavus endpoints
- logging or persistence for requests

In this repository, the existing answer engine is [app/services/chat/service.py](C:/Users/l/Documents/production_chatbot/app/services/chat/service.py:1).

If another project already has a function conceptually similar to:

```python
answer = await chat_service.generate_answer(...)
```

then Tavus can usually be added without changing the core intelligence path.

---

## 3. Files in this repo that are directly reusable

These are the main implementation pieces you can copy or adapt.

### 3.1 Configuration

- [app/config.py](C:/Users/l/Documents/production_chatbot/app/config.py:1)

Reusable settings added for Tavus:

- `TAVUS_API_KEY`
- `TAVUS_BASE_URL`
- `TAVUS_FACE_ID`
- `TAVUS_PAL_ID`
- `PUBLIC_BACKEND_URL`
- `TAVUS_TOOL_SECRET`

### 3.2 Tavus HTTP client

- [app/infrastructure/tavus/client.py](C:/Users/l/Documents/production_chatbot/app/infrastructure/tavus/client.py:1)

This contains:

- Tavus conversation creation
- Tavus conversation termination
- Tavus API authentication with `x-api-key`

### 3.3 Tavus service layer

- [app/services/tavus/service.py](C:/Users/l/Documents/production_chatbot/app/services/tavus/service.py:1)

This contains:

- orchestration for conversation creation
- PAL / face validation
- conversational context generation
- mapping Tavus conversation IDs back to backend conversations

### 3.4 Tavus API routes

- [app/api/tavus/routes.py](C:/Users/l/Documents/production_chatbot/app/api/tavus/routes.py:1)
- [app/api/tavus/schemas.py](C:/Users/l/Documents/production_chatbot/app/api/tavus/schemas.py:1)

This contains:

- `POST /api/tavus/conversations`
- `POST /api/tavus/tools/ask-tumelo`
- `POST /api/tavus/conversations/end`

### 3.5 Dependency injection

- [app/api/dependencies/tavus_dependencies.py](C:/Users/l/Documents/production_chatbot/app/api/dependencies/tavus_dependencies.py:1)

This wires:

- settings
- Tavus client
- Tavus service
- repository access

### 3.6 Frontend test harness

- [frontend/src/App.tsx](C:/Users/l/Documents/production_chatbot/frontend/src/App.tsx:1)
- [frontend/README.md](C:/Users/l/Documents/production_chatbot/frontend/README.md:1)

Use this when you want a quick local frontend to test Tavus without touching an existing production frontend.

---

## 4. Reusable rollout plan for another project

Use the following sequence in future projects.

### Step 1. Identify the existing text answer engine

Find the current backend entry point that takes a user question and returns the final answer.

Typical examples:

- `generate_reply(message, ...)`
- `chat(question, conversation_id, ...)`
- `run_rag_pipeline(query, ...)`
- `ask_assistant(prompt, ...)`

Your Tavus tool endpoint should call that exact path. Do not duplicate its logic.

### Step 2. Add Tavus configuration to env/settings

Add these env vars to the project:

```env
TAVUS_API_KEY=
TAVUS_BASE_URL=https://tavusapi.com
TAVUS_FACE_ID=
TAVUS_PAL_ID=
PUBLIC_BACKEND_URL=
TAVUS_TOOL_SECRET=
```

Implementation reference:

- [app/config.py](C:/Users/l/Documents/production_chatbot/app/config.py:1)

Important considerations:

- `PUBLIC_BACKEND_URL` must be the public tunnel/domain Tavus can reach.
- `TAVUS_TOOL_SECRET` must be an arbitrary shared secret string.
- `TAVUS_TOOL_SECRET` is not a URL.
- `TAVUS_PAL_ID` chooses the persona.
- `TAVUS_FACE_ID` chooses the avatar face.

### Step 3. Add a Tavus infrastructure client

Create a client dedicated to Tavus API calls.

Reusable behavior from this repo:

- create conversation
- end conversation
- attach Tavus API key via `x-api-key`
- centralize timeout and error handling

Implementation reference:

- [app/infrastructure/tavus/client.py](C:/Users/l/Documents/production_chatbot/app/infrastructure/tavus/client.py:1)

Recommended client responsibilities:

- no business logic
- only HTTP transport and Tavus response parsing
- raise backend-specific exceptions on failure

### Step 4. Add a Tavus service layer

Do not let the route call Tavus directly. Add a service layer that:

- validates required env vars
- constructs `conversational_context`
- extracts `conversation_id` and `conversation_url`
- optionally links Tavus conversation IDs to backend conversation records

Implementation reference:

- [app/services/tavus/service.py](C:/Users/l/Documents/production_chatbot/app/services/tavus/service.py:1)

This layer is where project-specific adaptation belongs.

For example:

- if your old project has conversations already, link them
- if it does not, store Tavus IDs however that project tracks sessions

### Step 5. Add Tavus API routes

At minimum, add these routes:

```text
POST /api/tavus/conversations
POST /api/tavus/tools/ask-<domain-subject>
POST /api/tavus/conversations/end
```

Implementation reference:

- [app/api/tavus/routes.py](C:/Users/l/Documents/production_chatbot/app/api/tavus/routes.py:1)

Recommended route responsibilities:

- validate request payload
- validate tool secret
- delegate to service layer
- delegate answer generation to existing chat service

### Step 6. Add a tool endpoint that calls the existing text engine

This is the most important backend endpoint.

In this repository, Tavus calls:

```text
POST /api/tavus/tools/ask-tumelo
```

The route then calls the existing backend answer engine:

```python
answer = await chat_service.generate_answer(
    user_message=request.message,
    conversation_id=request.tavus_conversation_id,
    channel="tavus_video",
    metadata={
        "visitor_name": request.visitor_name or "Website visitor",
        "source": "tavus_tool_call",
        "tavus_conversation_id": request.tavus_conversation_id,
    },
)
```

Design rule:

- Tavus tool route should be thin.
- Existing chat service should remain the brain.

### Step 7. Make sure the existing chat service can accept Tavus metadata

If your old project only supports pure text inputs, add minimal extension points for:

- `channel`
- `metadata`
- external conversation identifiers

In this repo, the chat service was extended to support:

- `channel="tavus_video"`
- metadata persistence
- mapping non-UUID Tavus conversation identifiers to backend conversations

Implementation reference:

- [app/services/chat/service.py](C:/Users/l/Documents/production_chatbot/app/services/chat/service.py:1)
- [app/repositories/chat_repository.py](C:/Users/l/Documents/production_chatbot/app/repositories/chat_repository.py:1)
- [app/repositories/models/message.py](C:/Users/l/Documents/production_chatbot/app/repositories/models/message.py:1)

If another project already persists message metadata, you may not need these changes.

### Step 8. Add route registration and CORS

Wire the Tavus router into the main FastAPI app.

Implementation reference:

- [app/main.py](C:/Users/l/Documents/production_chatbot/app/main.py:1)

For local frontend testing, allow the dev frontend origin only:

```text
http://localhost:5173
```

Do not open CORS globally unless the project already does that safely.

### Step 9. Add a lightweight frontend test harness

If the old project has no frontend or a production frontend is inconvenient to change, build a small Vite or React test app that:

- enters visitor name
- starts a Tavus conversation through your backend
- embeds `conversation_url`
- ends the conversation through your backend

Implementation reference:

- [frontend/src/App.tsx](C:/Users/l/Documents/production_chatbot/frontend/src/App.tsx:1)

This is not mandatory for production integration, but it speeds up debugging dramatically.

---

## 5. Tavus dashboard / PAL configuration pattern

This is where most integration failures happen.

### 5.1 What must be configured in Tavus

For the PAL you actually use, configure:

- the correct PAL/persona ID
- the backend tool
- the public tool URL
- the secret header
- instructions that force the PAL to use the tool for factual questions

### 5.2 The correct tool endpoint pattern

For this repo, the tool endpoint is:

```text
POST https://<public-backend-url>/api/tavus/tools/ask-tumelo
```

Headers:

```text
x-tavus-tool-secret: <TAVUS_TOOL_SECRET>
```

Request body shape:

```json
{
  "message": "{user_input}",
  "tavus_conversation_id": "tavus-session",
  "visitor_name": "Website visitor"
}
```

Adapt the route name and content fields for your future projects.

### 5.3 The PAL must not be generic

A common failure mode is using a default Tavus PAL such as a sales/demo persona. In that case Tavus may start the call correctly but never behave like your domain assistant.

Always confirm:

- `TAVUS_PAL_ID` points to the exact intended PAL
- the PAL has the backend tool configured
- the PAL instructions tell it when to use the tool

### 5.4 Recommended PAL instruction pattern

Use wording like:

```text
Use the backend tool for any factual question about the person, product, company, portfolio, projects, experience, skills, education, certifications, or contact information.
Do not answer these questions from memory.
Do not invent facts.
Speak the backend tool response as the final answer.
```

This same structure can be adapted to:

- HR assistant
- doctor assistant
- legal document explainer
- portfolio proxy
- customer support assistant

---

## 6. Local development and public callback pattern

When testing locally, the browser and Tavus do not reach the backend the same way.

### 6.1 Frontend-to-backend path

The local frontend should usually call:

```text
http://localhost:8000
```

In this repo:

```env
VITE_BACKEND_URL=http://localhost:8000
```

### 6.2 Tavus-to-backend path

Tavus cloud services cannot call `localhost`.

So your backend must expose a public URL via ngrok or another tunnel:

```env
PUBLIC_BACKEND_URL=https://<ngrok-or-cloudflare-url>
```

### 6.3 Reusable testing rule

Split the paths like this:

- browser -> `http://localhost:8000`
- Tavus cloud callback -> `https://<public-url>`

Never try to make Tavus call `http://localhost:8000`.

---

## 7. Exact mistakes encountered in this project and how to avoid them next time

These were the practical failure modes discovered during this integration.

### 7.1 Wrong Tavus tool URL

Incorrect:

```text
/api/tavus/tools
```

Correct:

```text
/api/tavus/tools/ask-tumelo
```

Lesson:

- Always copy the full route, not just the route prefix.

### 7.2 Wrong tool secret value

Incorrect:

```text
TAVUS_TOOL_SECRET=https://public-url/api/tavus/tools/ask-tumelo
```

Correct:

```text
TAVUS_TOOL_SECRET=tumelo-tavus-secret-2026
```

Lesson:

- `TAVUS_TOOL_SECRET` is a shared secret token, not a URL.

### 7.3 Wrong header name in Tavus

Incorrect:

```text
X-API-Key
```

Correct:

```text
x-tavus-tool-secret
```

Lesson:

- Tavus dashboard header config must match the FastAPI route exactly.

### 7.4 Wrong prompt version

The old-style alias `v1` failed because the available template names were:

- `v1_professional`
- `v2_warm_conversational`

This repo was updated so aliases normalize correctly.

Implementation reference:

- [app/infrastructure/prompts/prompt_loader.py](C:/Users/l/Documents/production_chatbot/app/infrastructure/prompts/prompt_loader.py:1)

Lesson:

- Keep canonical prompt names in env.
- Add compatibility aliases when migrating older projects.

### 7.5 Tavus not interpolating body variables as expected

At one point, the Tavus tool request reached the backend but with bad content or validation issues.

Lesson:

- If Tavus tool calls fail, inspect the actual incoming request via ngrok or your reverse proxy logs.
- Do not assume `{user_input}` or other variables are being populated as expected without checking.

### 7.6 Timeout too short

A 2-second tool timeout was too aggressive for a RAG + LLM backend.

Lesson:

- use 20 to 30 seconds for Tavus tool timeouts
- especially when retrieval and generation happen on demand

---

## 8. Debugging workflow to reuse in future projects

Whenever Tavus integration fails, debug in this order.

### 8.1 Check conversation creation

Verify the frontend/backend call:

```text
POST /api/tavus/conversations
```

Expected:

- `200 OK`
- Tavus `conversation_id`
- Tavus `conversation_url`

### 8.2 Check Tavus can reach your public backend

Test:

```text
GET https://<public-url>/health
```

Expected:

- `200 OK`

### 8.3 Check tool endpoint manually

Send a manual request:

```bash
curl -X POST https://<public-url>/api/tavus/tools/<tool-name> \
  -H "Content-Type: application/json" \
  -H "x-tavus-tool-secret: <secret>" \
  -d "{\"message\":\"Test question\",\"tavus_conversation_id\":\"manual-test\",\"visitor_name\":\"Test User\"}"
```

Expected:

- `200 OK`
- a valid answer payload

### 8.4 Check whether Tavus called the backend at all

If your backend logs show no request:

- Tavus tool is not configured
- PAL is wrong
- Tavus is not choosing the tool

If the backend logs show a request:

- inspect the exact status code
- inspect the response body

### 8.5 Use ngrok inspector

When using ngrok:

```text
http://127.0.0.1:4040
```

Inspect:

- request headers
- request JSON body
- response body

This was the fastest way to distinguish:

- missing tool call
- wrong header
- bad payload
- backend prompt/config error

### 8.6 Start new Tavus conversations after config changes

Do this after changing:

- PAL config
- tool config
- backend env
- prompt settings

Reason:

- old sessions often reflect old configuration state

---

## 9. Recommended reusable implementation recipe

If you are integrating Tavus into a new text-only project, use this sequence.

### Backend

1. Add Tavus env vars to settings.
2. Add Tavus HTTP client.
3. Add Tavus service layer.
4. Add Tavus routes:
   - create conversation
   - tool callback
   - end conversation
5. Make the tool callback call the existing answer engine.
6. Add `channel` and metadata support if missing.
7. Add narrow local CORS if using a dev frontend.
8. Add tests for:
   - Tavus client
   - secret validation
   - tool callback
   - route registration

### Tavus

1. Create/select the correct PAL.
2. Create/select the correct face.
3. Configure the backend tool.
4. Use the public backend URL.
5. Use `x-tavus-tool-secret`.
6. Set timeout to 20–30 seconds.
7. Add instructions that force factual questions through the tool.

### Frontend

1. Create a local test UI.
2. Call backend conversation start.
3. Embed `conversation_url`.
4. End conversation through backend.
5. Keep Tavus API credentials out of frontend code.

---

## 10. What to customize in future projects

These parts are project-specific and should be renamed or adjusted.

### 10.1 Tool route name

Current:

```text
/api/tavus/tools/ask-tumelo
```

Future examples:

- `/api/tavus/tools/ask-product-assistant`
- `/api/tavus/tools/ask-doctor-backend`
- `/api/tavus/tools/ask-legal-assistant`

### 10.2 Tool metadata

Current metadata includes:

- `visitor_name`
- `source=tavus_tool_call`
- `tavus_conversation_id`

Adapt this to your domain if your analytics model is richer.

### 10.3 Conversational context

Current implementation explicitly tells Tavus:

- use the backend tool
- do not invent facts
- speak the backend response

Adjust that context for the future domain, but keep those three principles.

### 10.4 Conversation linking

This repo links Tavus conversation IDs to backend conversation rows. Another project may use:

- Redis session IDs
- JWT-bound conversation keys
- tenant-aware identifiers
- customer CRM IDs

Adapt the persistence model, not the overall pattern.

---

## 11. Reusable checklist for future Tavus projects

Before first test:

- backend has Tavus env vars
- public tunnel works
- Tavus client can create conversations
- Tavus PAL ID is correct
- Tavus face ID is correct
- tool URL is exact
- tool secret header is exact
- tool timeout is long enough
- backend prompt default is valid
- frontend only calls backend, never Tavus directly

When debugging:

- no backend hit means Tavus config problem
- `401` means tool secret/header mismatch
- `422` means request schema mismatch
- `400` often means business validation failure
- `502` often means upstream LLM/Tavus/provider error

After fixes:

- restart backend
- create a brand new Tavus conversation
- test again

---

## 12. Suggested document storage options

For future reference, keep this content in one of these forms:

- `docs/tavus-integration-playbook.md`
- internal engineering wiki page
- project template repository
- LaTeX design note or appendix for implementation reports

If converting to LaTeX later, the current section structure maps cleanly to:

- architecture
- implementation
- environment setup
- testing
- failure modes
- debugging checklist

---

## 13. Summary

To reproduce Tavus integration in future text-only projects:

1. Do not replace your backend intelligence layer.
2. Add Tavus as a conversation transport and avatar interface.
3. Create Tavus conversations from your backend.
4. Expose a protected Tavus tool endpoint from your backend.
5. Route Tavus factual questions into the existing text backend.
6. Keep Tavus credentials server-side only.
7. Use a public callback URL for Tavus cloud requests.
8. Debug with real request inspection, not assumptions.

That is the exact integration pattern implemented in this repository.
