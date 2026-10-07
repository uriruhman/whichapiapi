# ADR-0006: FastMCP for the MCP surface

- Status: Accepted · 2026-09-28

## Decision
Use **FastMCP** (PrefectHQ/fastmcp, Apache-2.0) for the MCP server: stdio for local agents, streamable HTTP for
hosted. Tools map 1:1 to `whichapiapi.core` functions: `find_offers`, `estimate_cost`, `run_eval`, `recommend`,
`coverage`. Resources expose the capability taxonomy.

## Consequences
+ Minimal code, typed tool schemas from Python signatures. − Tied to FastMCP's release cadence (acceptable).
