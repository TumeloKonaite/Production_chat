# Internal Trace Schema

This backend now stores first-party request traces in Postgres before any vendor-specific observability integration.

## Purpose

The trace schema captures the lifecycle of a chatbot request in a vendor-neutral format:

- one `chat_traces` row per chatbot request
- many `chat_trace_steps` rows for ordered lifecycle events inside that request

The intent is to keep enough structure for future export to Langfuse, MLflow, or custom analytics without coupling the application to any one platform today.

## Tables

### `chat_traces`

Stores one high-level record for a single chatbot request.

Important columns:

- `conversation_id`, `request_id`, `session_id`: request correlation
- `input_text`, `output_text`: request and final answer text
- `status`, `error_message`: final request outcome
- `llm_provider`, `llm_model`, `prompt_version`: model execution metadata
- `observability_provider`, `external_trace_id`: optional vendor trace correlation fields
- `retriever_type`, `embedding_provider`, `embedding_model`: retrieval and embedding metadata
- `input_tokens`, `output_tokens`, `total_tokens`, `estimated_cost_usd`, `latency_ms`: performance and cost data
- `metadata`: flexible JSON for route, channel, Langfuse back-references, and additional safe request context

Indexed columns:

- `conversation_id`
- `created_at`
- `status`
- `llm_model`

### `chat_trace_steps`

Stores ordered step-level events for a trace.

Important columns:

- `trace_id`: parent request trace
- `step_index`: sequence order within the trace
- `step_type`: lifecycle event type
- `status`: step outcome
- `input_payload`, `output_payload`, `metadata`: flexible JSON payloads
- `latency_ms`, `error_message`, `started_at`, `completed_at`: timing and failure detail

Indexed columns:

- `trace_id`
- `step_type`

## Current Chat Write Path

`ChatService` records the following steps for normal web and Tavus chat requests:

1. `request_received`
2. `retrieval_started`
3. `retrieval_completed`
4. `prompt_built`
5. `llm_call_started` when an LLM call is used
6. `llm_call_completed` when an LLM call succeeds
7. `response_generated`
8. `error` when the request fails after tracing has started

The top-level trace is marked `success` or `error` at the end of the request.

## Safety Notes

- Trace writes run through a dedicated SQLAlchemy session so trace rollback does not interfere with the main chat persistence flow.
- If trace persistence fails, the chatbot request still continues.
- The schema avoids storing secrets by recording normalized error messages instead of raw upstream provider exceptions.
- Prompt capture is intentionally truncated to a preview to avoid storing oversized payloads in this first iteration.

## Future Extensions

This schema is intended to support later additions such as:

- external observability exporters
- prompt/version registries
- feedback and quality signals
- eval links
- redaction pipelines
- async export jobs
