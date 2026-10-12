FROM python:3.12-slim
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
USER app
ENTRYPOINT ["taco"]
