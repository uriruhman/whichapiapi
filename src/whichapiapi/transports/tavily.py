"""Tavily Search API (`search.web`). POST {base_url}/search, Bearer auth — verified against the live API 2026-09-28.

The provider "model" is the search depth (`basic` = 1 credit, `advanced` = 2). Usage is per request, not per token, so
cost comes from `Price.per_request`. Errors are prefixed like the OpenAI transport's so the runner's transient-error
handling (never cache a 429/5xx/connection failure) applies unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from whichapiapi.transports.base import CallResult, elapsed_ms, http_error_kind, redact

_PASSTHROUGH = {"max_results", "topic", "time_range", "include_domains", "exclude_domains", "country"}


class TavilyTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=timeout)

    async def call(self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> CallResult:
        query = next((str(m["content"]) for m in reversed(messages) if m.get("role") == "user"), "")
        body = {"query": query, "search_depth": model} | {
            k: v for k, v in config.items() if k in _PASSTHROUGH
        }
        t0 = time.perf_counter()
        try:
            r = await self.client.post(
                f"{self.base_url}/search", json=body, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        except httpx.HTTPError as e:
            return CallResult(error=f"APIConnectionError: {type(e).__name__}", latency_ms=elapsed_ms(t0))
        latency = elapsed_ms(t0)
        if r.status_code != 200:
            kind = http_error_kind(r.status_code)
            return CallResult(
                error=redact(f"{kind}: HTTP {r.status_code} {r.text[:200]}", self.api_key), latency_ms=latency
            )
        try:
            data = r.json()
            results = data["results"]
        except (ValueError, KeyError, TypeError):
            return CallResult(error="APIStatusError: unexpected response shape", latency_ms=latency)
        lines = [
            f"{i}. {x.get('title', '')} - {x.get('url', '')}\n   {x.get('content', '')}"
            for i, x in enumerate(results, 1)
        ]
        return CallResult(
            output="\n".join(lines),
            latency_ms=latency,
            raw={"response_time": data.get("response_time"), "n_results": len(results)},
            error=None if lines else "empty results",
        )
