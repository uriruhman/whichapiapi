"""Deterministic checks for the canary suite: exact/near-exact answers only, no LLM judge (Phase 2 canary —
judge variance would obscure a real degradation signal, and the whole point is to run this often and cheaply).

promptfoo-compatible python assertions: `fn(output, context) -> {"pass", "score", "reason"}`.
"""

from __future__ import annotations

import json


def _r(ok: bool, reason: str = "") -> dict:
    return {"pass": ok, "score": 1.0 if ok else 0.0, "reason": reason}


def matches(output: str, context) -> dict:
    """`context["vars"]["expected"]`: a JSON literal (compared parsed) or plain text (compared
    case-insensitively, trimmed of surrounding whitespace and a single trailing period)."""
    expected = str(context["vars"].get("expected", ""))
    out = output.strip()
    if expected.strip().startswith(("{", "[")):
        try:
            ok = json.loads(out) == json.loads(expected)
        except json.JSONDecodeError:
            return _r(False, f"not valid JSON: {out!r}")
        return _r(ok, "" if ok else f"expected {expected!r}, got {out!r}")
    norm = lambda s: s.strip().rstrip(".").casefold()  # noqa: E731
    ok = norm(out) == norm(expected)
    return _r(ok, "" if ok else f"expected {expected!r}, got {out!r}")
