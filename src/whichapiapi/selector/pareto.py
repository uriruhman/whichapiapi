"""Pareto front + primary/fallback recommendation (ARCHITECTURE §2, selector)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from whichapiapi.evaluator.runner import ProviderSummary

# model family -> company, so a fallback is really another vendor (gpt-4o-transcribe and whisper-1 are both OpenAI)
_FAMILY_VENDOR = {
    "gpt": "openai", "o": "openai", "whisper": "openai", "dall": "openai", "tts": "openai", "text": "openai",
    "chatgpt": "openai", "codex": "openai", "claude": "anthropic", "gemini": "google", "gemma": "google",
    "llama": "meta", "qwen": "alibaba", "qwq": "alibaba", "mistral": "mistral", "mixtral": "mistral",
    "voxtral": "mistral", "codestral": "mistral", "ministral": "mistral", "magistral": "mistral", "glm": "zhipu",
    "kimi": "moonshot", "grok": "xai", "nemotron": "nvidia", "parakeet": "nvidia", "canary": "nvidia",
    "mimo": "xiaomi", "deepseek": "deepseek", "minimax": "minimax", "command": "cohere",
}  # fmt: skip


def vendor_of(model: str, channel: str | None = None) -> str:
    """Vendor used to keep the fallback on a different company: 'gpt-6-luna' → 'openai', 'whisper-1' → 'openai'.
    Unknown families fall back to the family name; a model id with no family at all (Tavily's 'basic'/'advanced'
    search depths) belongs to its channel."""
    m = model.lower().split("/")[-1]
    family = re.split(r"[-_.:\d]", m, maxsplit=1)[0] or m
    if family == m and not re.search(r"\d", m) and channel and family not in _FAMILY_VENDOR:
        return channel
    return _FAMILY_VENDOR.get(family, family)


def _dominates(a: ProviderSummary, b: ProviderSummary) -> bool:
    ca, cb = a.effective_cost_per_case or 0.0, b.effective_cost_per_case or 0.0
    la, lb = a.latency_p50 or 0.0, b.latency_p50 or 0.0
    no_worse = a.score >= b.score and ca <= cb and la <= lb
    better = a.score > b.score or ca < cb or la < lb
    return no_worse and better


def pareto_front(items: list[ProviderSummary]) -> list[ProviderSummary]:
    cands = [s for s in items if s.n > 0]
    return [s for s in cands if not any(_dominates(o, s) for o in cands if o is not s)]


class Recommendation(BaseModel):
    primary: str | None
    fallback: str | None
    pareto: list[str]
    utilities: dict[str, float]
    rationale: list[str]


def _norm(values: dict[str, float], invert: bool) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return dict.fromkeys(values, 1.0)
    return {k: ((hi - v) if invert else (v - lo)) / (hi - lo) for k, v in values.items()}


def recommend(
    items: list[ProviderSummary], weights: dict[str, float] | None = None, min_pass_rate: float = 0.5
) -> Recommendation:
    w = {"quality": 0.6, "cost": 0.3, "latency": 0.1, **(weights or {})}
    cands = [s for s in items if s.n > 0 and s.errors < s.n]
    if not cands:
        return Recommendation(
            primary=None, fallback=None, pareto=[], utilities={}, rationale=["no successful runs"]
        )
    q = _norm({s.provider: s.score for s in cands}, invert=False)
    c = _norm({s.provider: s.effective_cost_per_case or 0.0 for s in cands}, invert=True)
    lat = _norm({s.provider: s.latency_p50 or 0.0 for s in cands}, invert=True)
    util = {p: round(w["quality"] * q[p] + w["cost"] * c[p] + w["latency"] * lat[p], 4) for p in q}
    front = pareto_front(cands)
    eligible = [s for s in front if s.pass_rate >= min_pass_rate] or front
    primary = max(eligible, key=lambda s: util[s.provider])
    others = [
        s
        for s in cands
        if vendor_of(s.model, s.channel) != vendor_of(primary.model, primary.channel)
        and s.pass_rate >= min_pass_rate
    ]
    fb = max(others, key=lambda s: util[s.provider]) if others else None
    why = [
        f"primary {primary.label}: score {primary.score:.2f}, pass {primary.pass_rate:.0%}, "
        f"${(primary.effective_cost_per_case or 0):.5f}/case, p50 {(primary.latency_p50 or 0) / 1000:.1f} s",
    ]
    if fb:
        why.append(
            f"fallback {fb.label}: different vendor ({vendor_of(fb.model, fb.channel)}), score {fb.score:.2f}"
        )
    else:
        why.append("no fallback from a different vendor met the pass-rate bar")
    cur = next((s for s in cands if s.current), None)
    cc, pc = (cur.effective_cost_per_case if cur else None), primary.effective_cost_per_case
    if cur and cur is not primary and cc and pc is not None:
        delta = (pc - cc) / cc
        why.append(
            f"vs current {cur.label}: score {primary.score - cur.score:+.2f}, cost {delta:+.0%} per case"
        )
    return Recommendation(
        primary=primary.provider,
        fallback=fb.provider if fb else None,
        pareto=[s.provider for s in front],
        utilities=util,
        rationale=why,
    )
