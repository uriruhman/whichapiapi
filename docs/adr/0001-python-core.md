# ADR-0001: Python core, uv, src layout

- Status: Accepted · 2026-09-28

## Context
The ecosystem we glue together is mostly Python (LiteLLM data, FastMCP, jiwer, Inspect, DeepEval, pandas-style
analysis). promptfoo and the n8n node are TypeScript.

## Decision
Core in **Python 3.12**, managed with **uv**, `src/whichapiapi` layout, Pydantic v2 models, Typer CLI, Ruff + pytest.
TypeScript only where the host demands it (n8n community node, Phase 5), as a thin client of the REST API.

## Consequences
+ Direct access to the Python AI tooling, one language for core logic.
− promptfoo cannot be imported as a library → ADR-0002.
