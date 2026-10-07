# Model selection: how a model variant is scored

Code: `src/whichapiapi/selector/cards.py` (pure), `core.model_cards` / `core.rank_models` / `core.export_model_data`.
Surfaces: `whichapiapi models rank|presets|export`, MCP `rank_models`, the interactive picker page (same formulas in JS).

## Unit of choice: a model *variant*

"Claude Opus 5.5 (Max Effort)" and "(High Effort)" are the same weights with very different cost and latency per task.
Cards come from Artificial Analysis rows, one per variant; benchmark boards from other sources attach by model key
(`schema.naming.model_key`, with leaderboard effort suffixes `-max`/`-high` merged in, exact name first).

## Quality for a task

1. Every benchmark component (AA intelligence/coding/math indices, HLE, long-context, IFBench, tau2, Terminal-Bench,
   BFCL, SWE-bench Verified, LMArena overall/coding/math/hard/russian/creative/webdev/vision/document) becomes a
   **0–100 percentile** over all models it covers. An Elo, an index and a pass rate are then comparable.
2. A task blends components with fixed weights (`TASKS`), e.g. coding = AA coding 35 % + SWE-bench 20 % + arena
   coding 20 % + webdev 15 % + Terminal-Bench 10 %.
3. Missing components are dropped and the weights renormalized; `coverage` = share of the blend that had data.
4. **Shrinkage:** quality = coverage × blend + (1 − coverage) × 50. A model measured on one benchmark can't outrank one
   that is good on all of them by luck. Candidates need coverage ≥ 0.3.

Mixing independent kinds of evidence — measured indices (AA), human preference votes (LMArena), agentic task suites
(SWE-bench, BFCL, tau2) — dampens any one benchmark's contamination, style bias or saturation.

## Cost per task (the "thinks too much" problem)

    thinking ≈ (time to first answer token − time to first token) × output tokens/s     (streamed reasoning)
             ≈ (time to first token − 1 s) × output tokens/s                         (hidden reasoning, only for
                                                                                        variants that say they reason)
    cost per task = in tokens × $in + (thinking + answer tokens) × $out
    seconds per task = time to first answer token + answer tokens / tokens/s

- A long first token on a non-reasoning model is queueing, not thinking (it would otherwise invent ~1.3k tokens for
  Mercury 2.5).
- Variants without a speed measurement get the median thinking of measured variants of the same effort class
  (max ≈ 7k, xhigh ≈ 2k, high ≈ 1.4k, medium ≈ 0.9k, low ≈ 0.2k, non-reasoning 0) and are marked `imputed`.
- Caveat: AA's speed test caps streamed reasoning near 2k tokens, so on hard tasks real thinking is higher. The
  estimate ranks variants correctly relative to each other; for absolute budgets run `eval run` on your own inputs.
- Price = the lower of AA's list price and the cheapest trusted catalog offer ("buy at"): first-party catalogs,
  vendors' own APIs, large inference providers and the user's own channels. Batch/free routes are excluded.
- Token shapes per task (`PROFILES`): chat 1500→400, classify 800→20, summarize 8000→500, RAG 12000→600, agent step
  6000→300, code 4000→1500, long document 60000→1000.

## Utility and presets

    score = 100 × (wq·quality/100 + wc·cheapness + ws·speed) / (wq + wc + ws)

Cheapness and speed are log-scaled across the candidates that pass the hard filters (a 10× gap counts the same at
any level). Hard filters: min quality, min tokens/s, max time to answer, max cost, vision, open weights.

Presets (`PRESETS`): best, optimal, value, fast (≥150 tok/s), ultrafast (≥400 tok/s), realtime (<1.5 s to answer),
coding, math, agents, russian, writing, vision, long_context, open.

## Other modalities

Image generation/editing, TTS and video: Artificial Analysis arena Elo; speech-to-text: Open ASR WER (lower is
better) and per-minute prices from the catalog. Shown as leaderboards; no per-task cost model yet.

## Freshness

`whichapiapi-refresh.timer` (systemd --user, daily 04:30 UTC) pulls catalogs, benchmarks and conditions and writes
`~/.local/share/whichapiapi/model-data.json`. Rankings in the CLI/MCP are always current; the published picker page
is a snapshot until it is republished with the new data file.

## Known limits

- Benchmark scores are the source's, on the source's tasks. Validate the top picks on your own inputs (`eval run`).
- AA covers speed for ~160 of ~680 variants; LMArena and AA name variants differently (matched by key + suffixes).
- Percentiles are relative to the current model set: a score of 90 means "better than 90 % of listed variants".
