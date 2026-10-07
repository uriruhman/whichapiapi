# Roadmap

Build bottom-up. Each phase ends with something that works end to end and can be shown.
**Rule: do not start the next phase until the current one's exit criteria pass.**

| Phase | Scope | Exit criteria |
|---|---|---|
| **0 — Evaluator (LLM)** | Suite format, openai-compatible + mock transports, deterministic assertions, LLM judge, cache, budget ledger with measured cost, Pareto report. First real suite: a production Telegram channel's post writer (10 cases). | `whichapiapi eval run <suite>` produces a report "on your tasks the best offer is X, N % cheaper than your current Y" within budget; tests green in CI. |
| **1 — MCP server** | FastMCP surface over the core: `find_offers`, `estimate_cost`, `run_eval`, `recommend`. Offers seeded from models.dev + OpenRouter + new-api. | An agent (Claude Code) can ask "which model for X under $Y, EU only" and get a sourced answer; first release tag. |
| **2 — Offers & cost engine** | Quality priors (LMArena CC BY 4.0, Artificial Analysis) so price-only search stops surfacing weak models; conditions as toggles (batch, cache, off-peak, data sharing), billing (top-up fee, FX, expiry), load profiles, history, InferIndex/LiteLLM cross-check, changedetection.io hook, canary for silent degradation. | Same model shows different effective monthly cost per profile, with every number traceable to a source. |
| **3 — Non-LLM** | `search.web` (relevance via judge) and `speech.stt` (WER via jiwer). | One suite per category runs through the same evaluator and selector. |
| **4 — Implementer: audit** | Import n8n JSON → capability graph → audit report. | Audit of a real workflow flags overpay / missing fallback / training-data leaks. |
| **5 — Implementer: apply** | Plan-as-diff, apply via n8n API with rollback, voice input, Telegram alerts, n8n community node. | "Describe by voice → reviewed diff → applied workflow" demo. |
| **6+ — Long tail** | Regional coverage map (transit etc.), community adapters/offers via PR. | Coverage map published with honest statuses. |
