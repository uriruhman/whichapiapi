# Security model

Which API API runs evaluation suites against paid APIs with **your** keys. Its trust boundaries:

## Suites are code

A suite can run Python (`python` assertions, `file://checks.py:fn`), name which env var holds a channel key and where
that key is sent (`base_url`), and choose which local files go into prompts or get uploaded as audio. Treat a suite
from someone else like a Makefile: read it before running it. What the tool enforces:

- `file://` references and STT audio paths must resolve **inside the suite's directory** (no `../`, no absolute paths
  outside it).
- Prompt templates render in Jinja2's `SandboxedEnvironment` (no attribute access to Python internals).
- Provider error text is stripped of API keys (`transports.base.redact`) before it is cached or written to reports.
- Paid runs refuse to exceed the run cap, the suite budget, the global cap (`WHICHAPIAPI_BUDGET_TOTAL`) and a
  channel's `min_balance` floor.

Not enforced: `python` assertions run in-process with your privileges, and a suite may point `key_env` at any env var.
Both are fine for your own suites and are why an MCP client may only run suites under trusted roots (below).

## MCP server

- **stdio** (default): only the client that launched it can talk to it.
- **`--http`**: listens on `127.0.0.1` by default. Binding any other address requires `WHICHAPIAPI_MCP_TOKEN`; clients
  then send `Authorization: Bearer <token>`.
- Suite paths passed by MCP clients must lie under `WHICHAPIAPI_SUITE_ROOTS` (`os.pathsep`-separated; default: the
  server's working directory).
- The changedetection.io webhook (`/webhooks/changedetection`) is not covered by the bearer auth; set
  `WHICHAPIAPI_WEBHOOK_TOKEN` (falls back to the MCP token) and call it with `?token=...`. Bodies over 64 KiB are
  rejected. It only records notifications and never spends money.

## Local data

`~/.local/share/whichapiapi/` (created `0700`) holds the offer catalog, cached model outputs, the spend ledger and
learned prices. It never stores keys. Keys come from the environment only.

## Known gaps

- Regex assertions use Python's backtracking engine: a hostile pattern can burn CPU.
- The global budget cap is checked per process; two concurrent runs can each spend up to it.
- new-api's call-log endpoint takes the key as a `?key=` query parameter (its API); it only ever goes to the channel
  that issued the key, over HTTPS.

Report issues privately to the repository owner.
