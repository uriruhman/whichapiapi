"""Any OpenAI-compatible Chat Completions endpoint (OpenRouter, new-api resellers, most providers). ADR-0003."""

from __future__ import annotations

import time
from typing import Any

import openai
from openai import AsyncOpenAI

from whichapiapi.transports.base import CallResult, Usage, elapsed_ms, redact

# promptfoo-style config keys → Chat Completions params
_PASSTHROUGH = {
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "response_format",
    "reasoning_effort",
    "seed",
    "stop",
    "extra_body",
}


class OpenAICompatTransport:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0, max_retries: int = 1):
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    async def call(self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> CallResult:
        params = {k: v for k, v in config.items() if k in _PASSTHROUGH}
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(model=model, messages=messages, **params)
        except openai.APIError as e:
            return CallResult(
                error=redact(f"{type(e).__name__}: {str(e)[:300]}", self.client.api_key),
                latency_ms=elapsed_ms(t0),
            )
        latency = elapsed_ms(t0)
        if not resp.choices:
            return CallResult(error="empty choices", latency_ms=latency)
        msg = resp.choices[0].message
        u = resp.usage
        usage = Usage()
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cdetails = getattr(u, "completion_tokens_details", None)
            usage = Usage(
                input_tokens=u.prompt_tokens or 0,
                output_tokens=u.completion_tokens or 0,
                cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                reasoning_tokens=(getattr(cdetails, "reasoning_tokens", 0) or 0) if cdetails else 0,
            )
        return CallResult(
            output=msg.content or "",
            usage=usage,
            latency_ms=latency,
            model_reported=resp.model,
            error=None if msg.content else f"empty content (finish_reason={resp.choices[0].finish_reason})",
        )
