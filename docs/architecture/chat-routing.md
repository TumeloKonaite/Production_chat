# Chat routing and grounding

The chat endpoint is a focused interface to Tumelo's approved portfolio knowledge. It does not
provide general-purpose question answering.

## Request flow

```text
validation and rate limiting
  -> exact-cache lookup
  -> load limited recent conversation history
  -> conditionally resolve an obvious contextual follow-up
  -> deterministic portfolio-scope routing
     -> DIRECT_RESPONSE: deterministic response, no retrieval or generation
     -> PORTFOLIO_KNOWLEDGE: approved retrieval, evidence check, grounded generation
  -> persist the assistant response and routing metadata
```

Exact-cache eligibility and lookup remain ahead of routing. Semantic caching and LLM-based
routing/query rewriting are not part of this flow.

## Routes

`DIRECT_RESPONSE` has four response kinds:

- `GREETING` and `ACKNOWLEDGEMENT` handle conversational messages.
- `CLARIFICATION` handles follow-ups whose subject cannot be resolved safely.
- `OUT_OF_SCOPE` redirects unrelated requests to Tumelo's background and projects. When a known
  technical topic is present, the redirect suggests asking how Tumelo used that topic.

These branches do not embed, search, retrieve curated projects, or call the LLM.

`PORTFOLIO_KNOWLEDGE` has two retrieval modes:

- `HYBRID` uses the configured approved-knowledge retriever for background, experience,
  education, skills, technology use, and project details.
- `PROJECT_OVERVIEW` selects the first curated chunk for each project section in `projects.md` for
  broad project and portfolio-listing questions.

Both modes require at least one relevant retrieved chunk before generation. Empty retrieval
returns a deterministic grounded-not-found response; it never falls through to pretrained model
knowledge.

## Context resolution

Context resolution runs only for messages with deterministic dependence signals such as `it`,
`that project`, `which one`, ordinal project references, or `tell me more`. It uses the limited
recent conversation window and performs a single resolve-and-route pass. If the recent exchange
does not clearly concern Tumelo's portfolio, the request becomes `DIRECT_RESPONSE / CLARIFICATION`.

## Observability

For each uncached request, assistant-message metadata and completed trace metadata include a
`routing` object with:

- route, reason code, resolved query, retrieval mode, or direct-response kind;
- context-resolution attempted/succeeded flags;
- retrieval attempted/result count/used flags;
- generation and grounded-not-found flags;
- routing latency.

Retrieval and generation latencies, token usage, cache details, trace IDs, and existing retrieval
logs continue to use their established fields.
