# Documentation

The root [README](../README.md) is the quick-start entry point. These guides are the authoritative homes for operational and technical detail.

| Guide | Purpose |
|---|---|
| [Architecture](architecture.md) | Understand product scope, request flow, persistence, integrations, and topology |
| [Configuration](configuration.md) | Configure every implemented subsystem and provider |
| [Local development](local-development.md) | Install, run, verify, stop, and reset the application locally |
| [Knowledge ingestion](ingestion.md) | Load curated or uploaded knowledge safely |
| [Evaluation](evaluation.md) | Run retrieval, generation, and end-to-end RAG experiments |
| [Observability](observability.md) | Operate internal traces, Langfuse, logging, MLflow, and DagsHub |
| [Deployment](deployment.md) | Deploy the backend and ingestion worker to Modal |
| [Troubleshooting](troubleshooting.md) | Diagnose common configuration and runtime failures |
| [Contributing](contributing.md) | Follow code, test, migration, and documentation expectations |

Additional implementation-specific references remain available:

- [Chat routing details](architecture/chat-routing.md)
- [Internal trace schema](observability/internal_trace_schema.md)
- [Modal reference](deployment/modal.md)
- [Supabase reference](deployment/supabase.md)
- [Langfuse trace export](evals/langfuse_trace_export.md)
- [Evaluation dataset reference](../evals/README.md)

When a command or variable changes, update its authoritative guide, the root README only if the quick-start changes, and `.env.example` if configuration changes.
