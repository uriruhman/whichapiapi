# ADR-0003: `openai` SDK as the default transport; LiteLLM optional

- Status: Accepted · 2026-09-28

## Context
Almost every channel we compare (OpenRouter, new-api/one-api resellers, DeepSeek, Groq, Together, Fireworks,
Mistral, Gemini's OpenAI endpoint…) speaks the OpenAI Chat Completions protocol. LiteLLM normalises many more,
but is a very large dependency and its PyPI releases 1.82.7/1.82.8 were **compromised on 2026-03-24**
(credential stealer via `.pth`; see BerriAI/litellm#24518). Our users hand us API keys; supply-chain surface matters.

## Decision
- Default transport: `openai` SDK with a per-channel `base_url`.
- LiteLLM is an **optional extra** (`whichapiapi[litellm]`) for native-protocol providers, pinned with hashes in `uv.lock`.
- LiteLLM's *price JSON* is consumed as data over HTTP, which needs no package at all.

## Consequences
+ Small, auditable dependency set. + No secret-stealing surface from a mega-package by default.
− Native-only providers need the extra or a small custom transport.
