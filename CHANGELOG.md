# Changelog

## v0.4.0 — 2026-09-29

- **Apply fixes to n8n (Phase 5)**: `apply n8n-retries` / `n8n-fallbacks` / `n8n-rollback` — reviewed diff, backup,
  PUT, read-back verification, git-copy sync; fallback = an OpenAI-compatible model node on another channel wired as
  the chain's second model. Verified live (18 chains on the owner's instance).
- **Benchmarks from your own data**: MCP `create_suite` (describe the task → suite; candidates auto-picked among
  models your keys can call via `suggest_candidates`), `audit n8n-suite --from-executions`, and
  `audit traces-suite` / MCP `suite_from_traces` for Langfuse (API v2 or export), Arize Phoenix / OpenInference and
  OpenTelemetry GenAI spans, JSON lines — production answers become the judge's reference.
- **Pairwise judging**: `eval pairwise`, both orders, position-biased verdicts become ties, judge cost measured.
- **Leaderboard auto-refresh**: `eval refresh` / MCP `refresh_suite` add today's best models for the suite's task
  (`x-whichapiapi.task`), re-run (cache keeps old models free), exit 3 when the recommended primary changes.
- **Per-request router**: OpenAI-compatible `/v1/chat/completions` + `/v1/models` on the HTTP server; `auto`,
  `auto:<task>`, `auto:<task>@<preset>`; deterministic task detection, cross-vendor fallback, route in headers.
- **Runs dashboard**: `runs-page` — your runs' cost/quality Pareto, per-provider table, case explorer.
- **Usage signal**: OpenRouter usage rankings for 12 categories as boards, shown on model cards and in the picker
  (never blended into quality); LMArena agent arena parsed.
- **Ops**: activity log for MCP/REST/CLI (`whichapiapi activity`), `report_feedback` inbox for agents in other
  projects.

## v0.3.0 — 2026-09-29

- **Model picker**: one card per model variant (reasoning effort included); quality = task-specific blend of
  benchmark percentiles, shrunk toward the median when data is partial; cost per task counts estimated thinking
  tokens; 14 presets; `models rank|presets|export|page`, MCP `rank_models`, REST `/api/rank`; interactive page.
- **Benchmarks**: generic score store; LMArena (all arenas and categories, via parquet), Open ASR Leaderboard,
  SWE-bench, BFCL, Artificial Analysis (LLM indices, per-benchmark scores, speed, media arenas).
- **Catalogs**: DeepInfra, Hugging Face router, Eden AI (non-LLM) adapters; capability inference (STT, TTS,
  embeddings, image, video); `offers pull all` pulls five catalogs concurrently.
- **Non-LLM**: OpenAI-compatible and Deepgram speech-to-text transports, per-minute costing, WER with Whisper's
  normalizers; LLM-judged search relevance.
- **Audit**: n8n (file or `--docker`), Make blueprints, source code; findings for overpay, better value, missing
  fallback, training-data risk, deprecated/unknown/runtime models, hardcoded keys; node → eval suite generator.
- **Release hardening**: security pass (sandboxed templates, suite-dir confinement, key redaction, MCP/REST token,
  webhook secret), read-only REST API, Dockerfile, daily refresh timer recipe, plans subtract cached calls.
- **Fixes**: per-request fees counted per call; unknown prices no longer pass price caps; re-pulled offers refresh
  `verified_at`; failed benchmark pulls keep previous scores; model keys unify dot/dash versions and Claude word order.

## v0.1.0 — 2026-09-28

MCP server, core façade, OpenRouter adapter, hard constraints, model-name normalization.

## v0.0.1 — 2026-09-28

LLM evaluator: promptfoo-compatible suites, measured cost from reseller call logs, judge, Pareto selector, reports.
