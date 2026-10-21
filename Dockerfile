FROM golang:1.26.1-bookworm AS go-build
WORKDIR /build
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -mod=readonly -trimpath -o /out/mcp-http ./cmd/mcp-http \
    && CGO_ENABLED=0 go build -mod=readonly -trimpath -o /out/mcp-stdio ./cmd/mcp-stdio

FROM python:3.12-slim AS indexer
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --link-mode=copy
COPY src ./src
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --link-mode=copy \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /config/data && chown app:app /config/data && chmod 700 /config/data
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=go-build /out/mcp-stdio /usr/local/bin/mcp-stdio
USER app
ENTRYPOINT ["taco"]

FROM debian:bookworm-slim AS mcp
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /config/data && chown app:app /config/data && chmod 700 /config/data
COPY --from=go-build /out/mcp-http /out/mcp-stdio /usr/local/bin/
COPY LICENSE /usr/share/doc/taco-shell-rag/LICENSE
WORKDIR /config
USER app
ENTRYPOINT ["mcp-http"]
