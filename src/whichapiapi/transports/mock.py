"""Deterministic offline transport for tests and dry runs.

Model names select behaviour: `echo` (returns last user message), `const:<text>`, `fail`.
Token usage is estimated from text length so cost logic is exercised.
"""

from __future__ import annotations

from typing import Any

from whichapiapi.transports.base import CallResult, Usage


class MockTransport:
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def call(self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> CallResult:
        self.calls.append((model, messages))
        prompt_text = "".join(str(m.get("content", "")) for m in messages)
        if model == "fail":
            return CallResult(error="mock failure")
        if model in self.responses:
            out = self.responses[model]
        elif model.startswith("const:"):
            out = model.removeprefix("const:")
        else:
            out = str(messages[-1].get("content", ""))
        return CallResult(
            output=out,
            usage=Usage(input_tokens=max(1, len(prompt_text) // 4), output_tokens=max(1, len(out) // 4)),
            latency_ms=1.0,
            model_reported=model,
        )
