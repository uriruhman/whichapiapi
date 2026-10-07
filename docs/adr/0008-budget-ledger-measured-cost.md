# ADR-0008: Hard budget ledger and measured cost

- Status: Accepted · 2026-09-28

## Context
Evals spend the user's money. Price lists lie (group ratios, promos, hidden fees).

## Decision
- Every run has a hard cap (`--budget`, default $0.25) plus a persistent global cap (`WHICHAPIAPI_BUDGET_TOTAL`).
- Pre-flight estimate (tokens × price, conservative ×1.5) must fit; in-flight, the runner stops before the call
  that would cross the cap. Unknown price → refuse unless `--allow-unpriced` with a per-call cap.
- **Measured cost**: when a channel exposes its own balance (new-api `/api/usage/token/`, OpenRouter
  `/api/v1/key`), read it before/after the run and store the delta next to the computed cost. The gap between
  computed and measured is itself a data point on the channel.

## Consequences
+ Users can trust us with a funded key. + Unique "real price" evidence.
− Extra HTTP calls; measured delta is per-run, not per-call, when calls run concurrently.

## Addendum 2026-09-28 — per-call logs beat balance deltas

First real run showed the cumulative token counter on new-api **does not move** for minutes (batch updates),
so before/after deltas read $0. new-api also exposes `GET /api/log/token?key=<key>&start_timestamp=…` — a
per-call log with the charged quota, the **price group the call was routed to** and its `group_ratio`.

- The runner now matches each fresh call to a log line by (prompt, completion) tokens and stores the real charge
  and route on the call (also in the results cache, so cached cases keep their real cost).
- After each provider block the budget is re-based on the real charge.
- `whichapiapi eval learn` reads the key's call history and stores per-model multipliers (`max` for planning,
  `mean` for reporting) so the pre-flight plan uses realistic prices before any spend.

Observed on the owner's a reseller key: the reseller routes every call dynamically (`token_mode: price`);
gpt-6-luna was charged 5.9 %–90 % of list (mean 51.5 %, 6 groups), gpt-6-sol 3.7 %–5.9 %.
