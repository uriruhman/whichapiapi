# Public router API

OpenAI-compatible endpoint for people the owner invites. Served by `whichapiapi serve-api` (127.0.0.1:8766) behind
nginx + Let's Encrypt at **https://whichapiapi.sluice.stream** (base URL with or without `/v1`). The docs page for
guests is `GET /` (Russian, `surfaces/api_docs.html`). MCP and the REST catalog are *not* exposed here.

| Route (also under `/v1`) | What |
|---|---|
| `POST /chat/completions` | `model`: `auto` · `auto:<task>` · `auto:<task>@<preset>` (guests); any model id (owner) |
| `GET /models` | routing policies |
| `GET /usage` | the caller's limit / spend / remaining |
| `POST /messages` | Anthropic Messages protocol (Claude Code: `ANTHROPIC_BASE_URL=https://whichapiapi.sluice.stream`, `ANTHROPIC_AUTH_TOKEN=<key>`, `ANTHROPIC_MODEL=auto:coding`); `x-api-key` auth also accepted. Concrete Claude model names from a guest's config are routed as `auto`. |
| `POST /messages/count_tokens` | local o200k estimate |
| `POST /responses` | OpenAI Responses protocol (Codex: provider with `wire_api = "responses"`), stateless (no `previous_response_id`) |

Response headers: `x-whichapiapi-route`, `-task`, `-tried`, `-trail` (every attempt: `route=status:class/ms`),
`-cost-usd`, `-repaired` (think tags / text tool calls / double-encoded args fixed), `-note`.
The same three protocol routes are on the internal HTTP server (172.19.0.1:8765, owner token).

## Keys

```bash
whichapiapi keys add alice --limit 0.20   # prints wa-… once; only a SHA-256 is stored
whichapiapi keys list                     # limit, spent, calls, last use
whichapiapi keys limit alice 0.50
whichapiapi keys revoke alice
```

Store: `$WHICHAPIAPI_HOME/api-keys.json` (0600, flock). The owner's `WHICHAPIAPI_MCP_TOKEN` also works, unlimited.

## Guards for guest keys

- `auto` policies only; answer capped at 4096 tokens; 20 requests/min (`WHICHAPIAPI_GUEST_RATE_PER_MIN`).
- Spend = list price of the answering route × learned real/list ratio (unpriced route: $0.01/call); HTTP 402 at the
  limit. Streams get `stream_options.include_usage` so they are charged too.
- HTTP 503 when the primary custom channel's balance is under `WHICHAPIAPI_GUEST_MIN_BALANCE` (default $0.40), so
  guests can't drain a key that production traffic also uses.
- Activity log records model, task, key name, tokens and cost — never message content.

## Ops

`~/.config/systemd/user/whichapiapi-api.service`; nginx site via `sudo agent-nginx-site add whichapiapi.sluice.stream 8766`.

## Tester keys (default role)

`keys add <name> --limit 0.20` makes a **tester** key: router + hosted MCP at `https://whichapiapi.sluice.stream/mcp`
(read-only tools + `route_chat` + `my_usage` + `report_feedback` tagged `tester:<name>`), and **full traces** of
everything it sends and gets back in `$WHICHAPIAPI_HOME/traces/<name>/*.jsonl` (0700/0600; view with
`whichapiapi traces [name] [--full]`; chat rows feed `audit traces-suite` directly). `--role guest` = router only,
no content logged. Testers are told about tracing on the docs page. The code stays private: testers use the hosted
server, nothing is published.

Codex (`~/.codex/config.toml`, shared by CLI, IDE extension and desktop app):

```toml
[mcp_servers.whichapiapi]
url = "https://whichapiapi.sluice.stream/mcp"
http_headers = { "Authorization" = "Bearer wa-…" }
```
