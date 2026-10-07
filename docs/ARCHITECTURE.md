# Architecture

> Status: living document. Decisions with trade-offs live in [`adr/`](adr/). The product vision is
> summarised in the [README](../README.md).

## 1. Shape of the system

One core library and thin surfaces around it. Every surface (CLI, MCP, REST, n8n node, Telegram) calls the
same Python API in `whichapiapi.core`; no surface contains business logic.

```
                    ┌────────────────────────── Surfaces ──────────────────────────┐
                    │  CLI (Typer) · MCP (FastMCP) · REST (Starlette routes on the MCP HTTP server) · n8n node · TG  │
                    └───────────────────────────────┬──────────────────────────────┘
                                                    │  whichapiapi.core (public Python API)
      ┌──────────────┬──────────────┬───────────────┼───────────────┬──────────────────┐
      ▼              ▼              ▼               ▼               ▼                  ▼
  Catalog        Offers          Cost engine     Evaluator        Selector         Implementer
  (capabilities, (who sells it,  (real price for (runs suites on  (hard filters,   (parse → capability
   coverage map)  on what terms)  a load profile) candidate offers) Pareto, fallback) graph → plan → diff)
      ▲              ▲                                 │
      │              │                                 ▼
      │        Adapters (plugins)               Transports (openai-compatible, native SDKs, HTTP)
      │        models.dev · OpenRouter ·        + Budget ledger (hard cap, measured spend)
      │        LiteLLM JSON · new-api ·         + Result cache (content-addressed)
      │        InferIndex · user YAML
      └──────── Store: SQLite (default) / Postgres, append-only history of every field
```

## 2. Modules (`src/whichapiapi/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `schema/` | Pydantic models: `Capability`, `Offer`, `Price`, `Conditions`, `Billing`, `Access`, `Provenance`, `LoadProfile`, `EvalSuite`, `EvalResult`, `Recommendation`. The **only** shared vocabulary. | pydantic |
| `adapters/` | One file per source. Each implements `Adapter.fetch() -> Iterable[Offer]` (or capabilities). Registered via entry points so third parties can ship adapters. Exception: benchmark adapters (`lmarena.py`, `open_asr.py`, `swebench.py`, `bfcl.py`, `artificial_analysis.py`) return `BenchmarkScore`s (schema/benchmark.py), not `Offer`s — they feed `core.pull_benchmarks` (cache `bench:{model_key}`, one entry per `source/board`), not `store.upsert_offers`. Same shape for `litellm_conditions.py` (`model_key -> Conditions`-shaped dict; HTTP JSON, never the `litellm` package — ADR-0003), feeding `core.pull_conditions`. `inferindex.py` is a third exception: `cheapest(model)` is a live, on-demand, rate-limited lookup (no bulk `fetch`/cache) used only by `core.cross_check_price` for informational price sanity/promo checks against the closest competitor. | schema, httpx |
| `store/` | Persist offers with history (`offer_versions`), eval runs, results cache, spend ledger. SQLite via SQLAlchemy Core; Postgres by URL. | schema |
| `cost/` | `effective_cost(offer, profile, toggles) -> CostBreakdown`: cache hits, batch share, off-peak share, top-up fee, FX, credit expiry, tiered context pricing. Pure functions. | schema |
| `evaluator/` | Load promptfoo-compatible suites, render prompts, call transports, score with assertions (deterministic + LLM judge), aggregate. Budget-guarded, cached, resumable. `canary.py` diffs a run's `ProviderSummary`s against a stored baseline (pass rate/score/latency drop) — pure functions, no I/O; `core.arun_canary` owns the run + baseline cache read/write. `assertions.py`'s `wer` type (Phase 3, `jiwer`) scores `speech.stt`; capability-agnostic, no transport changes needed. | schema, transports, store |
| `transports/` | How to call a channel: `openai_compat` (default, `openai` SDK with `base_url`), `mock` (tests/offline), `tavily` (`search.web`, selected by `Channel.protocol`), later native SDKs and other non-LLM HTTP APIs. Every call returns output + usage + latency + measured cost if the channel exposes it. | openai, httpx |
| `selector/` | Hard constraints (`constraints.py`: region, data retention, payment method, KYC, uptime, BYO; unknown → warn or exclude in strict mode) → Pareto front over quality / *real* cost / latency → primary + fallback from a **different** vendor, with reasons. Consumes `ProviderSummary` from the evaluator (documented exception to the layering rule). | schema, cost, evaluator models |
| `implementer/` | `n8n.py` (pure: n8n JSON → AI calls, models, fallbacks, schedule, leaked keys), `audit.py` (joins with the catalog/benchmarks → findings + Markdown). Phase 5: plan as a diff, apply via n8n API with rollback. | core, cost, schema, store |
| `surfaces/` | `cli.py`, `mcp.py`, `api.py`. Thin. `mcp.py` also registers one plain HTTP route via FastMCP's `@mcp.custom_route` (`/webhooks/changedetection`, served only in `--http` mode) — the one place a surface does more than call straight into `core`, since a webhook needs request/response handling `core` shouldn't own. | core |
| `core.py` | Public façade: `find_offers`, `compare_prices`, `estimate_cost`, `learned_prices`, `pull_offers`, `pull_benchmarks`, `model_benchmarks`, `benchmark_boards`, `leaderboard`, `pull_conditions`, `conditions_priors`, `cross_check_price`, `eval_plan`, `run_eval`/`arun_eval`, `run_canary`/`arun_canary`, `ingest_price_change_notification`, `price_change_alerts`, `taxonomy` (later `audit`). Returns plain JSON-able data. | modules |
| `schema/naming.py` | `model_key()` normalises model ids across channels (strips vendor prefix and full-date suffixes only). | — |

Rule: modules only import "downwards" in the table above. `schema` imports nothing from the project.

## 3. Core data model

`Offer` is the key entity: not "a model" but **one concrete way to buy a capability**.

```
Capability  (id: "llm.chat", io schema, default scorers)
   1 ── * Offer (capability, product/model, channel, channel_type, price, conditions, billing, access,
                 reliability, provenance, visibility: public | user)
                    │
                    └── * OfferVersion (append-only; every field change keeps the old value + source + time)
EvalSuite  (cases, assertions, judge config)   1 ── * EvalRun (suite hash, offers, budget, status)
EvalRun    1 ── * CaseResult (offer, case, output, scores, tokens, latency, cost_computed, cost_measured)
Recommendation (suite/run, profile, constraints → primary, fallback, pareto set, rationale)
```

Field-level provenance: every non-trivial field can carry `{source, verified_at, confidence}`. Stale data is
marked, not hidden. See `schema/offer.py`.

`channel_type`: `official | aggregator | authorized_reseller | user_added`. Offers with
`visibility=user` (BYO channels) live only in the user's local store and are never exported.

## 4. Key flows

### 4.1 Evaluate on my tasks (Phase 0)

```
suite.yaml (promptfoo subset) + candidate offers
  → plan: cases × offers, estimate cost from token estimate × price, refuse if over budget
  → for each (case, offer): cache lookup → transport.call → assertions → judge (if any)
  → ledger: computed cost always; measured cost when the channel exposes usage (new-api /api/usage/token)
  → aggregate per offer: pass rate, mean score, p50/p95 latency, cost per case, cost per *successful* case
  → Pareto front → report (terminal table + JSON + Markdown)
```

Budget guard is **pre-flight and in-flight**: the run stops before the call that would cross the cap.

### 4.2 Recommend (Phase 1–2)

`task description + constraints + load profile` → candidate offers from store (hard filters) →
cost engine per offer → optional quick eval → selector → primary + fallback + "what each constraint costs you".

### 4.3 Audit an automation (Phase 4)

`n8n workflow JSON` → extract nodes calling paid APIs → map to capabilities → current offer →
compare with alternatives → report: overpaying, no fallback, data used for training.

## 5. Cross-cutting concerns

- **Secrets:** never stored by us. Read from env / user keychain at call time. BYOK only. Nothing secret in
  repo, logs, cache keys or reports.
- **Money:** agent never buys, tops up or creates paid accounts. Every eval run has a hard budget; the ledger
  persists across runs.
- **Privacy:** suites can be marked `contains_pii: true`; then offers whose conditions include
  "training on your data" are excluded (public) or warned (BYO).
- **Determinism & cost:** results cache keyed by `hash(offer_id, model params, rendered prompt)`; re-runs are free.
- **Licensing:** adapters declare the license and attribution string of their source; reports print attributions.
- **Observability:** structured logs (stdlib `logging` + JSON formatter), run ids on every line.

## 6. Extension points

1. **Adapter** — new data source: one module + entry point `whichapiapi.adapters`.
2. **Transport** — new way to call a capability (native SDK, STT HTTP API…): entry point `whichapiapi.transports`.
3. **Assertion / scorer** — new check type: entry point `whichapiapi.assertions`.
4. **Offer YAML** — community-submitted offers in `data/offers/*.yaml`, validated in CI.

## 7. Deployment

- Library + CLI: `uv tool install whichapiapi` / `pipx`.
- MCP: `whichapiapi mcp` (stdio) or streamable HTTP.
- Self-host: `docker compose up` (API + Postgres) — Phase "release hardening", not started.
- changedetection.io hook: no docker-compose service yet — run changedetection.io separately and point its
  webhook at `whichapiapi mcp --http`'s `/webhooks/changedetection` (recipe in README). Landed Phase 2.

## 8. Non-goals (for now)

Per-prompt routing (RouteLLM-style), a proxy/gateway in the request path, reselling, grey key markets.
