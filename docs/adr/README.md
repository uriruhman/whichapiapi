# Architecture Decision Records

Format: [MADR](https://adr.github.io/madr/) light. One decision per file, never edited after `Accepted`
except to mark it `Superseded by ADR-NNNN`.

| # | Title | Status |
|---|---|---|
| [0001](0001-python-core.md) | Python core, uv, src layout | Accepted |
| [0002](0002-own-eval-runner-promptfoo-format.md) | Own eval runner with a promptfoo-compatible suite format | Accepted |
| [0003](0003-openai-sdk-transport-litellm-optional.md) | `openai` SDK as default transport; LiteLLM optional | Accepted |
| [0004](0004-sqlite-first-sqlalchemy-core.md) | SQLite first via SQLAlchemy Core, Postgres by URL | Accepted |
| [0005](0005-price-data-sources.md) | Price/catalog sources: models.dev, OpenRouter, LiteLLM JSON, new-api, InferIndex | Accepted |
| [0006](0006-fastmcp-surface.md) | FastMCP for the MCP surface | Accepted |
| [0007](0007-byo-channels-local-only.md) | BYO channels are local-only and carry a risk profile | Accepted |
| [0008](0008-budget-ledger-measured-cost.md) | Hard budget ledger and measured cost | Accepted |
