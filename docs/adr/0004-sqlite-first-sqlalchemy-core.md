# ADR-0004: SQLite first via SQLAlchemy Core, Postgres by URL

- Status: Accepted · 2026-09-28

## Decision
Local-first: `~/.local/share/whichapiapi/whichapiapi.db` (override with `WHICHAPIAPI_DB_URL`). SQLAlchemy Core (no ORM),
JSON columns for nested Offer parts, append-only `offer_versions` for history. Postgres for hosted/self-hosted
via the same URL. Migrations with Alembic once the schema stabilises (Phase 2).

## Consequences
+ Zero-setup CLI and MCP. + Same code for hosted.
− JSON columns are less queryable; acceptable at our volume (~10k offers).
