# Working on taco-shell-rag

This is a GPU-only, evidence-first RAG starter kit. Keep integration-specific
logic and workflows in adapters, outside the core. Do not commit credentials
or deployment-local configuration.

## Boundaries

- Core contracts and services must not import MCP, Qdrant, HTTP clients or model libraries.
- Keep model roles, data sources and execution backends independently selectable.
  Defaults and presets are conveniences, not hidden dependencies or a custom-model allowlist.
  Expose resolved choices and reject incompatible settings instead of silently changing models.
- Adapters implement the core protocols. Wire implementations at the application entrypoint.
- GPU inference requires CUDA. Do not add a CPU deployment or silent device fallback.
- Search returns source evidence, timestamps and limitations. The caller owns the answer.
- Indexing is distinct from live API observation; do not silently index live tool results.
- Preserve stable source/document identities. Advance checkpoints only after publication.
- A failed or incomplete scan must not delete previously indexed documents.
- Never treat a missing, failed or truncated retrieval as proof of absence.
- This first release has one trust boundary per deployment, not per-user document ACLs.

## Code and writing

Use ordinary names and direct control flow. Add an abstraction when an actual
adapter or test needs it. Comments should explain a constraint or decision,
not translate the next line. Retain input validation and meaningful error handling.
Use paired Korean and English comments for non-obvious invariants, safety
boundaries and recovery decisions. Both versions must mean the same thing.
Do not annotate every line.

Keep third-party attribution and licenses. Never invent test results or performance claims.

## Validation

Run `uv run pytest` and `uv run ruff check .`. Contract tests use fake inference;
passing them is not a claim that CUDA, real models or GPU containers were tested.
For deployment changes, validate Compose configuration and run the installation
doctor. Do not use a production GPU server without explicit authorization.

## Agent entrypoints

Read the `taco-install` skill for installation, recovery or MCP setup. Codex
discovers it under .agents/skills; Claude Code under .claude/skills. Keep both
installation instructions aligned with docs/install.md and the actual scripts.

Do not read unrelated .env files, change Git identity, force-push, delete user
data, install system packages or restart unrelated services without permission.
