# ADR-0005: Price and catalog sources

- Status: Accepted · 2026-09-28

## Decision (LLM, in priority order)
1. **models.dev** `api.json` — seed of provider×model offers (225 providers, ~8k rows, MIT).
2. **OpenRouter** `/api/v1/models` + `/endpoints` — aggregator prices, uptime, per-upstream data policy.
3. **new-api adapter** — any reseller running new-api/one-api (`/api/pricing`, `/api/status`): model ratios × group
   ratios → one Offer per (model, group); top-up rate, currency, min top-up → `billing`.
   Formula verified on a reseller 2026-09-28: `usd_per_1M_input = model_ratio × 2 × group_ratio`,
   `output = input × completion_ratio`, `cache_read = input × cache_ratio`.
4. **LiteLLM JSON** — batch/cache/tiered prices where others lack them.
5. **InferIndex** — cross-check and promo detection (external, no auth, rate-limited).

Conflicts: keep all values with provenance; the Offer's effective price is the most recent `official` source,
falling back by confidence. Never silently average.

## Consequences
+ No scraping of our own in Phase 0–2. − Dependent on third-party freshness → provenance + staleness flags.
