"""Learn real-vs-list price multipliers from a channel's call history (ADR-0008).

A reseller that routes each call to a price group charges `list price x group_ratio`. Reading the key's own call
log gives the ratios this key actually gets, per model, before spending anything on an eval.
"""

from __future__ import annotations

import time
from collections import defaultdict

from whichapiapi.evaluator.balance import BalanceProbe, CallLogEntry
from whichapiapi.store.db import Store


def learn_from_log(store: Store, probe: BalanceProbe, channel: str, days: float = 14) -> dict[str, dict]:
    entries = probe.calls_since(time.time() - days * 86400, max_pages=50)
    learn_group_usage(store, entries, channel)
    by_model: dict[str, list[CallLogEntry]] = defaultdict(list)
    for e in entries:
        if e.group_ratio is not None:
            by_model[e.model].append(e)
    learned = {}
    for model, es in by_model.items():
        ratios = [float(e.group_ratio or 0.0) for e in es]
        key = f"ratio:{channel}/{model}"
        prev = store.cache_get(key) or {"max": 0.0, "n": 0}
        # re-reading an overlapping window must not count the same log lines again
        seen = prev.get("last_ts", 0)
        val = {
            "max": round(max(prev["max"], *ratios), 5),
            "mean": round(sum(ratios) / len(ratios), 5),
            "n": prev["n"] + sum(1 for e in es if e.ts > seen),
            "groups_seen": len(set(ratios)),
            "last_ts": max(seen, *(e.ts for e in es)),
        }
        store.cache_put(key, val)
        learned[model] = val
    return learned


def learn_group_usage(store: Store, entries: list[CallLogEntry], channel: str) -> dict[str, dict]:
    """Which price group actually answered this key's calls, per model (`groups:{channel}/{model}`).

    The consume log only lists calls that were billed, so a group that keeps refusing the key never shows up here
    (share 0) — that is exactly the signal: list price of such a group is not reachable. Overwritten on every
    run (a rolling window), unlike the ratio priors which accumulate."""
    by_model: dict[str, list[CallLogEntry]] = defaultdict(list)
    for e in entries:
        by_model[e.model].append(e)
    out = {}
    for model, es in by_model.items():
        groups: dict[str, dict] = {}
        for g in {e.group or "(none)" for e in es}:
            ge = [e for e in es if (e.group or "(none)") == g]
            groups[g] = {
                "n": len(ge),
                "share": round(len(ge) / len(es), 4),
                "mean_cost_usd": round(sum(e.cost_usd for e in ge) / len(ge), 8),
            }
        val = {
            "n": len(es),
            "groups": groups,
            "retried_share": round(sum(1 for e in es if e.retries) / len(es), 4),
            "window_from": min(e.ts for e in es),
        }
        store.cache_put(f"groups:{channel}/{model}", val)
        out[model] = val
    return out
