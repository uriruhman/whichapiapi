# Competitors (2026-09-29)

Sources: vendors' own pages and docs (links below), gathered via Tributary + web search. "?" = not stated publicly.

## Head to head

| | **Which API API** | **Artificial Analysis Optima** | **OpenRouter (Auto Router)** | Braintrust | Not Diamond / RouteLLM | Eden AI | InferIndex / BenchLM / pricepertoken |
|---|---|---|---|---|---|---|---|
| What it is | Pick & wire the best API for an automation task | Custom benchmark on your workload | Model gateway + router | Evals + observability + gateway | Trained prompt routers | Gateway for LLM + non-LLM APIs | Price / benchmark comparison sites |
| Price catalog | ✅ ~10k offers, 5 catalogs + models.dev (225 providers) | list prices of evaluated models | ✅ own 400+ models | ✅ via gateway | – | ✅ own providers | ✅ |
| **Real billed cost incl. resellers** | ✅ per-call log attribution, learned real/list multipliers, reseller price groups | ? (cost per task, list-price based as far as stated) | own prices + 5.5 % credit fee | – | – | own prices + 5.5 % fee | list only |
| Public benchmarks | ✅ 6 sources, 121 boards (AA, LMArena, Open ASR, SWE-bench, BFCL, OpenRouter usage) | ✅ own indices | ✅ Benchmarks page | – | – | – | ✅ (BenchLM: 486 evals) |
| **Evals on your own data** | ✅ promptfoo-style suites from a description, n8n executions or traces (Langfuse, Phoenix, OTel); LLM judge + pairwise, WER, keyword/JSON checks; auto-refresh on new models; runs dashboard | ✅ **one click, upload / traces / "describe it"**, rubric + pairwise judge | – | ✅ on real traffic | ✅ train a router on your prompts | – | – |
| Cost per task incl. thinking tokens | ✅ per variant, thinking estimated from AA timings | ✅ cost + time per task | – | ✅ | partial | – | – |
| Recommendation + fallback from another vendor | ✅ primary + fallback, Pareto | ? | ✅ runtime fallback array | partial | ✅ routing | ✅ fallback | – |
| Runtime routing per request | ✅ OpenAI-compatible `/v1/chat/completions` with `auto` / `auto:<task>@<preset>`, cross-vendor fallback | – | ✅ ~30 task types × 7-day community spend, 5 cost tiers | ✅ | ✅ | ✅ | – |
| Non-LLM (STT, TTS, search, OCR, translation, image) | ✅ catalogs + real STT/search evals | images in benchmarks; speech leaderboards | partial | – | – | ✅ | – |
| **Audit & fix automations (n8n, Make, code)** | ✅ audit + node → eval suite + apply retries/fallbacks with backup & rollback | – | – | – | – | – | – |
| Silent model substitution canary | ✅ | – | – | partial (online scores) | – | – | – |
| Agent interfaces | MCP, REST, CLI, OpenAI-compatible router | "Optima skill" for coding agents | API, OpenAI-compatible | SDK, API | SDK | API | web |
| Hosting / privacy | local-first, BYOK, AGPL-3.0 | SaaS | SaaS, ZDR option | SaaS / hybrid | SaaS / OSS (RouteLLM) | SaaS | web |

Also adjacent: LiteLLM, Portkey, Vercel AI Gateway, Merge Gateway, Cloudflare / Kong / TrueFoundry AI gateways
(routing + fallback, no evals on your data); Azure Foundry model router, Amazon Bedrock prompt routing (single
cloud); Martian, Unify (routers); promptfoo (OSS evals — our suite format is compatible); Langfuse, Arize (traces).

## Where we are ahead

1. **What you actually pay.** Real per-call charges from the channel's own log, learned real/list multipliers, reseller
   price groups and which group really answers — nobody else models grey/reseller channels.
2. **Automations, not just models.** Audit of n8n / Make / code, node → eval suite, and applying the fix (retry,
   fallback on another channel) with diff, backup, verification, git sync and rollback.
3. **Beyond chat.** The same evaluator and selector for STT, search, TTS, OCR, translation, images.
4. **Aggregator of aggregators.** models.dev (225 providers) + OpenRouter + DeepInfra + HF router + Eden AI + your
   own resellers, with benchmarks from five sources on one scale.
5. **Local, BYOK, open.** Your keys never leave the machine; MCP/REST for any agent; AGPL.

## Where they are ahead — and what to build

| Gap | Who | Plan |
|---|---|---|
| "Describe the use case → benchmark built for you" | Optima | ✅ MCP `create_suite` (the agent drafts prompt/cases/criteria; candidates auto-picked among models your keys can call) + `audit n8n-suite` |
| Import real traffic as a benchmark | Optima (Arize, Braintrust, Langfuse), Braintrust | ✅ n8n executions (`--from-executions`) + `audit traces-suite` / MCP `suite_from_traces`: Langfuse (API v2 or export), Arize Phoenix / OpenInference and OTel GenAI spans (OTLP/JSON), JSON lines (Braintrust/Helicone reshaped); production answers become the judge's reference, production model found on your channels vs today's best. Langfuse API path tested with a mock only (no live instance) |
| Pairwise judging | Optima | ✅ `eval pairwise` (both orders; position-biased verdicts become ties) |
| Leaderboard auto-refresh when new models ship | Optima | ✅ `eval refresh` / MCP `refresh_suite`: today's best models for the suite's task that your keys can buy → added to the suite → re-run (old results cached, so only new models cost) under the budget; exit 3 + activity event when the primary changes; timer recipe in OPERATIONS |
| Polished hosted UI | Optima, OpenRouter | ✅ `whichapiapi runs-page`: one self-contained page of your own runs (cost/quality Pareto, per-provider table with real vs list cost, case explorer with judge reasons), rebuilt daily next to the model picker; served locally, no SaaS |
| Runtime per-request routing | OpenRouter, Not Diamond, Braintrust | ✅ own OpenAI-compatible router `/v1/chat/completions` (`auto`, `auto:<task>@<preset>`): task detected per request, benchmark + cost-per-task choice, fallback to another vendor, route in response headers; plus OpenRouter `models` list |
| Usage signal ("what people actually use") | OpenRouter rankings | ✅ 12 OpenRouter usage boards (`openrouter_usage/<category>`: programming, roleplay, translation, legal, finance, …; public `/api/v1/models?category=`, refreshed daily) shown on model cards and in the picker as "топ OpenRouter: programming #1"; deliberately kept out of the quality score (popularity ≠ quality) |

## Sources

- Optima: https://artificialanalysis.ai/articles/optima
- OpenRouter Auto Router: https://openrouter.ai/docs/guides/routing/routers/auto-router ·
  https://openrouter.ai/blog/insights/model-routing/ · provider routing: https://openrouter.ai/docs/guides/routing/provider-selection
- Router landscape: https://www.braintrust.dev/articles/best-llm-routers-2026 · https://www.edenai.co/post/best-llm-routers ·
  https://www.digitalocean.com/resources/articles/best-llm-routers · https://entelligence.ai/blogs/9-best-llm-routers-and-model-routing-tools-in-2026
- BenchLM: https://benchlm.ai/benchmarks
