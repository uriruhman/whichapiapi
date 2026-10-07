# Model integrity: is the model behind a name the one we pay for?

Grey resellers (a reseller and other new-api relays) route each call to the cheapest upstream "group". The risks:
a cheaper sibling served under the expensive name (Luna as Sol), a Chinese model behind a Western name, a hidden
system prompt that changes behaviour and is billed to us, faked usage. `whichapiapi integrity run` checks this
cheaply and on a schedule; the router checks hidden prompts passively on every live call.

## What runs

`whichapiapi integrity run [examples/integrity/watch.yaml] [--notify]` — ~20 short calls per watched model
(~$0.002 per model on a reseller, read back exactly from its call log by request id). MCP: `run_integrity` (asks
first), `integrity_status` (read-only). Weekly timer `whichapiapi-integrity.timer` (a random moment within each week — owner, 2026-10-07: daily is
more than needed; routing differs by time of day) with `--notify`: alerts and status changes go to the owner's Telegram (`tg-owner`).
Baselines follow the canary policy: first run records, a clean run refreshes, a run with alerts does not.

| Check | How | Catches | Cost |
|---|---|---|---|
| tokenizer | billed `prompt_tokens` of "Reply OK." vs "Reply OK." + 2 calibration texts (EN/RU/ZH/code/rare words); the difference cancels any hidden prompt; compared with local tokenizers (OpenAI o200k via tiktoken; DeepSeek, Qwen, GLM, MiMo via their `tokenizer.json`) | vendor swap, esp. a Chinese model behind a Western name; Claude/Gemini (no public tokenizer) must match their baseline and no local one | 3 calls, output 16 tokens |
| hidden_prompt | billed prompt tokens of "Reply OK." minus its real size; plus a persona question ("what kind of assistant are you configured to be?") for channels that hide those tokens | a relay injecting its own system prompt | 1 call |
| identity | "which company trained you" | Chinese vendor answering under a Western name (alert); other mismatches warn (models misreport) | 1 call |
| censorship | two China-sensitive questions (Tiananmen 1989, Winnie the Pooh); alert only if a Western model deflects both | Chinese-hosted substitute | 2 calls |
| fingerprint | random number ×4, random animal ×4, reasoning tokens on a fixed puzzle ×2 → distance (JSD + log reasoning ratio) to the model's own baseline and to its cheaper `sibling` measured in the same run | same-vendor downgrade (Sol→Luna, Opus→Haiku) | 10 calls (+ the sibling's) |
| metadata | id prefix (`resp_`/`chatcmpl-`/`msg_`), `system_fingerprint`, usage fields | backend change | free |

Passive (free): the router compares each plain-text OpenAI-family call's billed prompt tokens with a local count →
`route_health` shows the median hidden-prompt tokens per route.

## Measured on a reseller (2026-10-07)

- Tokenizer deltas on calibration text 1: o200k 90, DeepSeek 92, Qwen3 94, GLM 87, MiMo 94; billed: gpt-6.1-sol,
  gpt-6-sol, gpt-6-luna 90 (exact), gemini-3.8-flash 91, deepseek-v4.1-flash 92 (exact), glm-5.3-flash 87 (exact),
  mimo-v2.6-flash 94 (exact), qwen3.8-flash 87 (not Qwen3's 94 — a newer tokenizer, so qwen mismatches only warn),
  claude-opus-5-5 / claude-sonnet-5-5 139, claude-haiku-4.5 120 (pre-4.7 Claude tokenizer). A Chinese model behind
  a Claude name would bill ≤ 94 → caught.
- Hidden prompts: `group-a*` groups (gpt-6-sol, gpt-6.1-sol) bill ~4.4k extra prompt tokens (3840 cached) and the
  model calls itself "a pragmatic, rigorous coding assistant"; `group-e1` (Opus) does not bill extra tokens but
  answers "I'm Claude Code, Anthropic's CLI" → a hidden Claude Code prompt.
- Sol vs Luna: same tokenizer, similar answers ("otter", digit strings starting 5831049276…), both solve hard
  arithmetic; Sol reasons ~2× more efficiently (1/97 digit: Sol 141–202 vs Luna 310–600 reasoning tokens). Second run:
  Sol's distance to its baseline 0.12 vs to Luna 0.30; Sonnet 0.33 vs Haiku 0.66.
- One false alarm on the first run (Sonnet hedged once on Tiananmen, answered factually 2/2 on retry) → censorship now
  alerts only when every sensitive question is deflected.

## Not done (and why)

- **Watermarks.** Per vendor statements found in research, Google's SynthID-Text is used by Gemini and licensed to
  Anthropic for Claude; OpenAI shelved text watermarks. SynthID-Text changes token sampling (no hidden Unicode), and
  detection needs the vendor's secret key; Anthropic's detection API is a private preview, Google's SynthID Detector
  portal doesn't verify text. If the owner gets preview access, it would be the strongest check for Claude — add a
  `watermark` check that sends sampled answers to that API. (Claim not independently verified by us.)
- **Anthropic `count_tokens`.** Free but needs an Anthropic API key; would turn Claude's tokenizer check from
  baseline-only into exact. Add when a key exists (`ANTHROPIC_API_KEY`).
- **Statistical equality test vs a trusted reference** (Model Equality Testing, MMD over ~20 prompts × 10 samples;
  IRIS random-generation features). Needs a trusted channel for the same model (official API or OpenRouter key) —
  our fingerprint is the cheap version of this against the model's own past and its sibling.
- **LLMmap** (8 queries, open-source classifier): its trained model targets older model sets; our probes cover the
  same idea for the models we actually buy.
- **Long-context needle / max-output limits**: catches context truncation; expensive, so not scheduled.
- **TEE attestation**: only possible when the provider offers it; resellers don't.

## Research notes (2026-10-07, via Tributary web search)

Chinese relay community checkers (claude-detector: 19 probes incl. `msg_`/`toolu_` ids, prompt caching, count_tokens
audit; RelayRadar: LLMmap-based 8 queries ~4k tokens, "Hi" token-inflation test; API-CHECK; apiranking cache checks);
papers: Model Equality Testing (MMD, 77% median power at 10 samples/prompt), RUT (rank-based, local reference
needed), IRIS (179 visible-string features, AUROC ≥ 0.95 in one query for sequence probes), PALACE (token-count
auditing); greedy decoding is not reproducible across providers (< 5% exact match), so exact-match tests are useless.
OpenRouter Auto Exacto re-scores providers every ~5 min on tool-call validity and benchmarks.
