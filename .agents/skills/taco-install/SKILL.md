---
name: taco-install
description: Install, connect or diagnose this repository's GPU RAG stack and optional generation providers. Use for first-time setup, GPU/container failures, model changes and MCP client configuration, not ordinary source-code edits.
---

# Install taco-shell-rag

Read [the installation guide](../../../docs/install.md) before executing commands.
For model selection or replacement, also read [model configuration](../../../docs/models.md).
Resolve paths from the repository root; never assume the current directory is the root.

Start by inspecting the host and existing deployment. This runtime requires Linux,
an NVIDIA GPU, its driver, Docker Compose/Buildx and NVIDIA Container Toolkit. A Mac can
run contract tests or act as an MCP client, but is not a supported inference host.
Run `python3 scripts/bootstrap.py --check`; distinguish host checks from actual
container CUDA access and successful model inference. Never substitute CPU inference.
Reuse working image-provided drivers/toolkit packages; do not force past package holds.
For native package installation, select Python 3.12 explicitly with uv. The project
supports Python 3.11 through 3.13; its pinned Torch 2.8 CUDA wheels do not support
Python 3.14. The base Docker images already select Python 3.12.

Keep existing settings and data. Ask for the source directories or API contract,
the intended GPU and deployment ports when unknown. Use `.env.example` for names
and explanations; credentials belong in the local `.env`, not the conversation.
Run `python3 scripts/bootstrap.py --write-env` only when local configuration is
authorized; it must not overwrite an existing file. Never source `.env` as shell code.

Select embedding, reranking and generation independently. Inference model choices
belong in models.toml; .env selects its path, credentials and host GPU allocation.
Inspect presets or resolve a profile with `taco models` before loading weights.
Support compatible custom identities instead of forcing a preset. Distinguish
configuration inspection from runtime doctor/search results and real GPU tests.
For split processes, verify both URLs and GPU mappings; indexing must not depend
on reranker availability. Do not start roles that the user has disabled.

Explain and obtain approval before system-package installation, Docker daemon
changes, image/model downloads, starting GPU processes or changing an existing
deployment. Do not infer this authority from permission to diagnose. Stop on a
missing prerequisite or a shared-GPU conflict; show the next concrete action.
Never delete volumes, caches or source documents as a repair shortcut.

Validate Compose before startup. Once authorized, build and start the base stack,
wait for inference readiness, run the container doctor, sync the example source,
and verify both search and the exact-version document tool through MCP. A health
check alone is not end-to-end verification. Report the commands actually run,
the observed result, and anything skipped; do not invent successful CUDA checks.

Generation is optional and separate from embedding/reranking. Claude CLI ambient
authentication, Anthropic API keys and local OpenAI-compatible endpoints are
different adapters. Do not interchange their credentials. Ask before external
model calls. Agent mode requires working tool calling; select `rag` explicitly
for text-only models instead of silently downgrading an agent failure.

For an embedding-model/revision/encoding-option change, use a new collection and
catalog and reindex. This includes prompt, precision and length changes and the
initial migration from the legacy option-free fingerprint. Do not erase the old
pair. A reranker-only or generator-only change does not require re-embedding.
Check the runtime's tool parser, model revision and available VRAM
before starting a self-hosted generator. Do not promise that every model fits.

Bind MCP to loopback by default and use SSH forwarding for a remote GPU host.
Configure the client with its own secret storage. Each deployment shares all
configured sources; bearer authentication is not document-level authorization.
Finish with source freshness, search evidence and remaining limitations.
