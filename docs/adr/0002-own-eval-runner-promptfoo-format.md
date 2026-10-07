# ADR-0002: Own eval runner with a promptfoo-compatible suite format

- Status: Accepted · 2026-09-28

## Context
The concept says "take the eval engine from promptfoo". promptfoo is Node; running it as a subprocess would
hide cost accounting, caching and budget enforcement from us — and those are the product. Python frameworks
(Inspect AI, DeepEval) are heavier than a 10–50-case user suite needs and couple to their own telemetry/cloud.

## Decision
- Suite files use a **subset of promptfoo's YAML schema**: `description`, `prompts`, `providers`, `defaultTest`,
  `tests[].vars`, `tests[].assert[]` with types `equals`, `contains`, `icontains`, `not-contains`, `regex`,
  `is-json`, `contains-json`, `javascript` (ignored with warning), `python`, `llm-rubric`, `cost`, `latency`,
  plus `file://` references. Unknown fields are preserved and ignored.
- `wer` (word error rate via `jiwer`, Phase 3) is our own addition beyond promptfoo's types — non-LLM
  capabilities like `speech.stt` need it and promptfoo has no native equivalent.
- Our extensions live under the `x-whichapiapi` key (budget, judge, PII flag, load profile), which promptfoo ignores.
- The runner is ours: ~500 lines, async, cached, budget-guarded.
- `whichapiapi eval export-promptfoo` writes a file promptfoo can run unchanged.

## Consequences
+ Users keep a portable format; we own cost/budget/measurement.
− We re-implement a small set of assertions. Mitigation: keep the set small, add via entry points.
