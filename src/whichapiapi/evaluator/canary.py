"""Canary comparison: run a small deterministic suite repeatedly, diff each provider's summary against a
stored baseline to catch silent model substitution or degradation on a channel.

Baseline policy (not user-configurable per run, on purpose): a clean run
(no alerts) refreshes the baseline, so a legitimate model upgrade becomes the new normal on its own. A run
that raises an alert does NOT refresh the baseline, so a real regression can't quietly become "the new
normal" just by running canary again — the caller must pass `reset_baseline=True` after reviewing it.
"""

from __future__ import annotations

from typing import Any

from whichapiapi.evaluator.runner import ProviderSummary

PASS_RATE_DROP_ALERT = 0.2  # absolute drop vs baseline
SCORE_DROP_ALERT = 0.15  # absolute drop vs baseline
LATENCY_RATIO_ALERT = 2.0  # p50 more than this many times the baseline p50


def new_baselines(summaries: list[ProviderSummary]) -> dict[str, dict[str, Any]]:
    return {
        s.provider: {"pass_rate": s.pass_rate, "score": s.score, "latency_p50": s.latency_p50, "n": s.n}
        for s in summaries
        if s.n > 0
    }


def compare_to_baseline(
    summaries: list[ProviderSummary], baselines: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One alert per provider that drifted from its stored baseline. Providers with no baseline yet (first
    run, or a new provider added to the suite) are skipped, not flagged."""
    alerts = []
    for s in summaries:
        base = baselines.get(s.provider)
        if base is None:
            continue
        reasons = []
        was_healthy = base["pass_rate"] > 0
        now_all_failed = s.n == 0 or s.errors == s.n
        if was_healthy and now_all_failed:
            # a provider that was ALREADY failing 100% stays failing without a fresh alert every run —
            # that's steady state, not new information; only a healthy -> broken transition is news.
            reasons.append("every call errored" if s.n else "no calls completed (n=0)")
        elif not now_all_failed:
            if base["pass_rate"] - s.pass_rate >= PASS_RATE_DROP_ALERT:
                reasons.append(f"pass rate dropped {base['pass_rate']:.0%} -> {s.pass_rate:.0%}")
            if base["score"] - s.score >= SCORE_DROP_ALERT:
                reasons.append(f"score dropped {base['score']:.2f} -> {s.score:.2f}")
            bp50, sp50 = base.get("latency_p50"), s.latency_p50
            if bp50 and sp50 and sp50 > bp50 * LATENCY_RATIO_ALERT:
                reasons.append(f"latency_p50 {bp50:.0f}ms -> {sp50:.0f}ms")
        if reasons:
            alerts.append(
                {
                    "provider": s.provider,
                    "label": s.label,
                    "reasons": reasons,
                    "possible_cause": "silent model substitution or channel degradation — verify manually "
                    "before trusting this provider's offers",
                }
            )
    return alerts
