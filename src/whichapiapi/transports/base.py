"""Transport = how to call one channel. Returns output + usage + latency; never raises on API errors."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

_KEYLIKE = re.compile(r"\b((?:sk|tvly|gsk|pk|rk)[-_])[A-Za-z0-9_\-]{12,}")


def redact(text: str, *secrets: str) -> str:
    """Strip API keys from provider error text before it is cached or written to a report."""
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "***")
    return _KEYLIKE.sub(r"\1***", text)


def elapsed_ms(t0: float) -> float:
    """Milliseconds since `t0` (a `time.perf_counter()` reading), rounded for reports."""
    return round((time.perf_counter() - t0) * 1000, 1)


def http_error_kind(status: int) -> str:
    """OpenAI-SDK-style error name for an HTTP status, so the runner's transient-error handling (never cache a
    429/5xx) works the same for every transport."""
    return "RateLimitError" if status == 429 else "InternalServerError" if status >= 500 else "APIStatusError"


def audio_file(messages: list[dict[str, Any]], base_dir: Path) -> Path | str:
    """The audio path in the last user message, resolved inside `base_dir`; an error string if it escapes the
    suite directory (a suite must not upload arbitrary local files) or doesn't exist."""
    rel = str(next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")).strip()
    path = (base_dir / rel).resolve()
    if not path.is_relative_to(base_dir.resolve()):
        return f"audio path outside the suite directory: {rel}"
    return path if path.is_file() else f"audio file not found: {rel}"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    units: float = 0.0  # billable non-token units priced by `Price.per_unit`, e.g. audio minutes


class CallResult(BaseModel):
    output: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None
    model_reported: str | None = None  # what the API says answered (substitution detection)
    cached: bool = False  # served from our results cache
    raw: dict[str, Any] | None = None
    measured_cost: float | None = None  # what the channel actually charged (from its call log)
    route: str | None = None  # where the channel routed the call, e.g. "group-a x0.037"
    cache_key: str | None = None


class Transport(Protocol):
    async def call(
        self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]
    ) -> CallResult: ...
