# ADR-0007: BYO channels are local-only and carry a risk profile

- Status: Accepted · 2026-09-28

## Decision
- Public catalog: only `official | aggregator | authorized_reseller`.
- Users may add any channel (`channel_type: user_added`, `verified: false`) manually or by pointing at an
  OpenAI-compatible endpoint / new-api instance. Stored with `visibility: user` in the local store only; exports
  and public APIs filter them out by construction (query layer, not UI).
- A risk profile is shown next to the price (unknown retention, key revocation risk, no SLA, upstream group
  semantics unknown). Warn, don't block. PII suites → warning by default.
- Canary checks detect model substitution on such channels (Phase 2).

## Why
Resellers (e.g. new-api groups priced at 4–90 % of list) are how many non-US users actually buy access. Ignoring
them makes the tool useless for them; promoting them publicly is a trust risk. Local-only resolves both.
