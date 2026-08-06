FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /config/data && chown app:app /config/data && chmod 700 /config/data
ENV PATH="/app/.venv/bin:$PATH"
USER app
ENTRYPOINT ["taco"]
