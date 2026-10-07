# Prior art and reuse map

> "Assemble the plane from existing parts." This file lists what already exists, what we reuse and how,
> and what we deliberately do **not** reuse. Checked 2026-09-28; stars and activity are from the GitHub API that day.

Legend for **How**: `lib` = Python dependency · `data` = we read its published data (HTTP/JSON) ·
`svc` = external service we call · `format` = we adopt its file format · `ref` = inspiration only.

## 1. Catalog and price data (LLM)

| Project | ★ / license / last push | What it gives | How | Notes |
|---|---|---|---|---|
| [models.dev](https://github.com/anomalyco/models.dev) (ex `sst/models.dev`) | 7.0k · MIT · 2026-09-28 | `https://models.dev/api.json`: **225 providers, ~8,250 provider×model rows** with input/output/cache prices, context, modalities, tool/JSON support | `data` | **Primary seed for Offers.** Each provider×model row maps 1:1 to an `Offer`. |
| [OpenRouter API](https://openrouter.ai/docs) | — | `/api/v1/models` (prices), `/models/{id}/endpoints` (per-upstream price, uptime, quantization, data policy) | `data` | Uptime + per-endpoint data policy, no key needed for listing. |
| [LiteLLM](https://github.com/BerriAI/litellm) | 59.7k · MIT-ish · 2026-09-28 | `model_prices_and_context_window.json` (batch, cache, tiered prices) | `data` (raw JSON), optional `lib` | We read the JSON, not the package, by default. See ADR-0003 (supply-chain incident on PyPI in 2026). |
| [Portkey-AI/models](https://github.com/Portkey-AI/models) | 140 · MIT · 2026-09-28 | `configs.portkey.ai/pricing/{provider}.json` | `data` | Cross-check source for prices. |
| [InferIndex](https://github.com/InferIndex/inferindex-docs) | docs MIT, backend private | Cheapest reseller per model across 70+ sources, price history, promo detection, flex/batch tiers, MCP | `svc` (cross-check) + **competitor** | Closest competitor on "price with conditions". LLM-only, no evals on user tasks, no non-LLM categories, no implementation. Public API, no auth, 60 rpm. |
| [new-api](https://github.com/QuantumNous/new-api) | 49k · AGPL-3.0 | The engine behind hundreds of resellers (a reseller is one). Public `/api/pricing` exposes `model_ratio`, `completion_ratio`, `cache_ratio`, `group_ratio` per channel group; `/api/status` exposes top-up rate, currency, min top-up. | `data` (HTTP only, **no code reuse**, AGPL) | **One adapter covers every new-api/one-api reseller.** Verified formula: `USD per 1M input = model_ratio × 2 × group_ratio`. Per-call log `/api/log/token?key=` → exact charge, routed price group and ratio for every call (the cumulative `/api/usage/token/` counter lags). |
| [tokencost](https://github.com/AgentOps-AI/tokencost) | 2.0k · MIT · last push 2025-09 | Token counting + price table | `ref` | Stale; LiteLLM JSON is fresher. |
| [simonw/llm-prices](https://github.com/simonw/llm-prices) | 184 · none · 2026-09-22 | Hand-curated price history | `ref` | No license → do not ingest. |
| [Artificial Analysis API](https://artificialanalysis.ai/documentation) | — | Quality/speed/price indices | `svc` (Phase 2+, needs key, attribution required) | |
| [LMArena dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) | CC BY 4.0 | Arena Elo | `data` (Phase 2+) | Attribution required. |
| [which-llm](https://github.com/ariobarin/which-llm) | 3 · MIT · 2026-09-27 | Agent skill, daily AA+OpenRouter snapshot | `ref` + competitor | Public benchmarks only. |
| [llm-pricing-mcp-server](https://github.com/skakumanu/llm-pricing-mcp-server) | 0 · MIT | MCP with router/benchmark features | `ref` | Tiny, unproven. |

## 2. Evaluation

| Project | ★ / license | How | Decision |
|---|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | 25.5k · MIT · Node | `format` | We adopt a **compatible subset of its YAML** (prompts / providers / tests / assert types) so a user can also run `npx promptfoo eval` on the same file. We do not embed the Node runtime. ADR-0002. |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 2.9k · MIT | `ref` | Great scorer/solver model, but heavy for 10–50-case user suites. Candidate backend later. |
| [DeepEval](https://github.com/confident-ai/deepeval) | 18.5k · Apache-2.0 | `ref` | G-Eval rubric prompts inspire our judge prompt. Too much telemetry/cloud coupling for the core. |
| [jiwer](https://github.com/jitsi/jiwer) | 929 · Apache-2.0 | `lib` (Phase 3) | WER/CER for `speech.stt`. |
| [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) | Apache-2.0 | `ref`/`data` | Text normalizer for WER; public STT priors. |
| [Picovoice STT benchmark](https://github.com/Picovoice/speech-to-text-benchmark) | Apache-2.0 | `ref` | Provider adapters for cloud STT. |
| [tavily-search-evals](https://github.com/tavily-ai/tavily-search-evals) | MIT | `ref` (Phase 3) | Multi-provider search harness (Tavily/Exa/Brave/Serper/Perplexity). |
| [openbenchmarks-labs](https://github.com/openbenchmarks-labs/factual-lookup-company-news-search) | MIT | `ref`/`data` | "Cost per 1,000 correct answers" metric — same idea as our cost-per-success. |

## 3. Routing / selection

| Project | How | Notes |
|---|---|---|
| [RouteLLM](https://github.com/lm-sys/RouteLLM) (inactive since 2024) | `ref` | Per-prompt routing is out of scope; we pick per task node. |
| [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) | `ref` | Same. |
| Pareto front | own ~30 lines | Too small to justify a dependency (`paretoset` exists if needed). |

## 4. Surfaces and plumbing

| Project | ★ / license | How |
|---|---|---|
| [FastMCP](https://github.com/PrefectHQ/fastmcp) | 27.9k · Apache-2.0 | `lib` — MCP server surface (Phase 1). |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 24.4k · MIT | transitive via FastMCP. |
| [openai-python](https://github.com/openai/openai-python) | Apache-2.0 | `lib` — transport for every OpenAI-compatible channel (OpenRouter, new-api resellers, most providers). |
| Typer, Pydantic v2, httpx, FastAPI, SQLAlchemy | permissive | `lib` |
| [changedetection.io](https://github.com/dgtlmoon/changedetection.io) | 34.6k · Apache-2.0 | `svc` (self-hosted, Phase 2) — watches pricing pages, webhooks into our ingest. |

## 5. Catalogs for non-LLM capabilities (Phase 3+)

| Source | License | How |
|---|---|---|
| [APIs.guru openapi-directory](https://github.com/APIs-guru/openapi-directory) | CC0 | `data` — OpenAPI specs → capability I/O schemas. |
| [public-apis](https://github.com/public-apis/public-apis) | MIT | `data` — long-tail discovery. |
| [MCP Registry](https://github.com/modelcontextprotocol/registry) | — | `data` — `registry.modelcontextprotocol.io/v0/servers`. |
| [Mobility Database catalogs](https://github.com/MobilityData/mobility-database-catalogs) | Apache-2.0 | `data` — GTFS/GTFS-RT feeds per city → `transit.*` coverage map. |
| Apify Store, Smithery, Glama | ToS | `svc` — link-out only until licensing is clear. |

## 6. Automation / implementation (Phase 4+)

| Project | ★ / license | How |
|---|---|---|
| [n8n-mcp](https://github.com/czlonkowski/n8n-mcp) | 23k · MIT | `svc`/`data` — node catalog + workflow validation. Already available as an MCP server in the author's environment. |
| n8n public REST API | — | `svc` — create/update workflows with rollback (local instance at `localhost:5678`). |

## 7. Positioning after research

The "price with conditions" slice is **not empty** (InferIndex, launched 2026). Our differentiation stays:
1. evaluation on the **user's own tasks** (nobody above does it end to end),
2. **all capability categories**, with honest coverage maps,
3. **measured** cost (read back from the channel's own billing), not only computed,
4. **implementation** into the automation (n8n diff + rollback),
5. BYO channels with a risk profile.

InferIndex and models.dev become **inputs**, not rivals, for the price layer.

## Artificial Analysis Optima (checked 2026-09-28)

[artificialanalysis.ai/optima](https://artificialanalysis.ai/optima): hosted custom benchmarks — describe the work,
attach examples, a build agent drafts tasks + rubrics, models are scored on score / cost per task / time per task
(rubric or pairwise judges; own HTTP agent can compete; raw token cost + $0.002/criterion standard judge).
It overlaps our Phase 0 core (eval on *your own* tasks) but is a hosted, model-level comparison. Not found on the
page: provider/reseller channel comparison, real billed cost, price groups/conditions, hard constraints (region,
retention, payment), non-LLM capabilities, applying the choice to an automation, self-hosting/BYO channels.
Treat as the closest rival for the eval step, and Artificial Analysis's API as a quality-prior input.

Full competitor comparison (2026-09-29): [COMPETITORS.md](COMPETITORS.md).
