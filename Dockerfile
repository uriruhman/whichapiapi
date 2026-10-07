# Which API API — CLI + MCP/REST server. Local-first: keys come from the environment, data lives in /data.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    WHICHAPIAPI_HOME=/data WHICHAPIAPI_SUITE_ROOTS=/suites

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev && useradd --create-home --uid 10001 app && mkdir -p /data /suites && chown app /data

USER app
VOLUME ["/data"]
EXPOSE 8765
ENTRYPOINT ["whichapiapi"]
# HTTP on 0.0.0.0 refuses to start without WHICHAPIAPI_MCP_TOKEN (see SECURITY.md)
CMD ["mcp", "--http", "--host", "0.0.0.0", "--port", "8765"]
