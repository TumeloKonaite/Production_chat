# Contributing

## Development workflow

1. Create a focused branch from the current default branch. Prefer descriptive names such as `docs/configuration-reference` or `fix/retrieval-timeout`.
2. Follow [local development](local-development.md) to install dependencies, migrate, ingest, and run the API.
3. Make the smallest cohesive change; keep unrelated refactors separate.
4. Add or update tests and evaluation examples appropriate to the behavior.
5. Run required checks and review the diff before opening a pull request.

## Coding standards

- Target Python 3.12 and preserve the service/repository/infrastructure boundaries already used by the codebase.
- Prefer explicit types, small functions, and domain-specific errors.
- Keep API input/output models strict and avoid leaking provider/database errors to clients.
- Use `uv` and commit `uv.lock` when dependency resolution changes.
- Do not introduce secrets, real user data, or production endpoints with embedded credentials.

## Formatting, linting, typing, and tests

The repository's enforced static check is Ruff:

```bash
uv run ruff check .
```

Run all tests:

```bash
uv run python -m pytest
```

There is no configured formatter or standalone type checker in `pyproject.toml`; do not claim CI runs one. Keep annotations consistent and add a type-checking tool only in a dedicated, agreed change.

Use focused tests during development, then run the full suite. Database integration tests are opt-in and must target an isolated disposable database.

## Database migrations

- Create an Alembic revision for every schema change.
- Do not edit a migration that may already have been applied outside your branch.
- Keep one coherent head; verify with `uv run alembic heads`.
- Test `uv run alembic upgrade head` from the previous schema state.
- Review destructive operations explicitly and document deployment/rollback implications.
- Update models, repositories, tests, and documentation together.

## Environment-variable changes

When adding, renaming, aliasing, or removing a variable:

1. Update `app/config.py` and configuration tests.
2. Update `.env.example` with a safe default or empty secret.
3. Update [configuration](configuration.md), including required conditions and defaults.
4. Update deployment docs/secrets if production operation changes.
5. Remove stale names only after compatibility implications are understood.

Never put real keys in examples. `.env.example` is the canonical name inventory; `docs/configuration.md` is the canonical explanation.

## Evaluation expectations

Run the lowest-cost evaluation that measures the changed subsystem:

- retrieval/index changes: retrieval smoke/baseline;
- prompt/model changes: fixed-context generation;
- cross-cutting RAG changes: end-to-end RAG or a matrix suite;
- routing changes: routing unit tests, plus unsupported/portfolio examples where appropriate.

Document dataset version, model/provider, prompt, retrieval settings, and material result changes in the pull request. Never silently weaken expected sources/facts to make a regression pass.

## Documentation expectations

Update documentation whenever a change affects:

- environment variables or defaults;
- installation/startup commands or supported Python/dependency versions;
- API routes, request/response behavior, or authentication headers;
- routing or scope behavior;
- ingestion sources, chunking, replacement, or worker operation;
- evaluation schemas, commands, metrics, or output paths;
- observability, caching, rate limiting, storage, or tracking integrations;
- migrations, deployment, CI/CD, health checks, or rollback.

Put detailed instructions in the focused guide and keep only the shortest onboarding path in the root README. Verify every new command against `--help`, source, CI, or an actual local run. Use relative repository links and keep Mermaid syntax GitHub-compatible.

## Commit guidance

Use concise imperative commits that describe one logical change, for example:

```text
docs: add verified ingestion workflow
fix: preserve retrieval source metadata
test: cover out-of-scope routing
```

Do not rewrite or discard another contributor's uncommitted work. Avoid generated evaluation artifacts, caches, local databases, `.env`, and credentials in commits.

## Pull-request checklist

- [ ] The change is focused and its behavior is explained.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run python -m pytest` passes.
- [ ] Relevant migrations were tested and the Alembic graph has one head.
- [ ] Relevant evaluation smoke/baseline was run or its omission is explained.
- [ ] `.env.example` and configuration docs match settings code.
- [ ] README/focused guides reflect command, API, workflow, or deployment changes.
- [ ] New links and Mermaid diagrams render correctly on GitHub.
- [ ] No secrets, private production data, or credential-bearing URLs are present.
- [ ] The PR includes verification evidence and any known operational/rollback risk.
