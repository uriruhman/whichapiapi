"""Field reliability: how models behave on the owner's real traffic, from other tools' call logs.

Reads Tributary's `usage.jsonl` (contract: tributary_mcp docs/CONTRACT.md — `type: "call"` rows with provider, model,
status ok|timeout|http_*|error, ms, tokens, cost; `type: "job"` rows name the model that answered a job;
`type: "rating"` rows carry good/bad verdicts per job). Aggregates per "provider:model": calls, ok rate, timeouts,
p50 latency, good/bad ratings. Used by `suggest_candidates` to push unreliable channel:model pairs down, and exposed
read-only (MCP `field_stats`). Paths: WHICHAPIAPI_FIELD_LOGS (os.pathsep-separated), default
`~/.local/share/tributary-*/usage.jsonl`.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MIN_CALLS = 5  # fewer calls than this: shown, but never used to demote
UNRELIABLE_OK_RATE = 0.7


def log_paths() -> list[Path]:
    env = os.environ.get("WHICHAPIAPI_FIELD_LOGS")
    if env is not None:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return sorted(Path.home().joinpath(".local/share").glob("tributary-*/usage.jsonl"))


def _rows(paths: list[Path], since: datetime) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(str(row.get("ts", "")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts >= since:
                out.append(row)
    return out


def stats(days: float = 14, paths: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    """{"provider:model": {calls, ok, ok_rate, timeouts, errors, p50_ms, good, bad, reliable}} over the last `days`."""
    rows = _rows(log_paths() if paths is None else paths, datetime.now(UTC) - timedelta(days=days))
    agg: dict[str, dict[str, Any]] = {}
    job_model: dict[str, str] = {}
    for r in rows:
        if r.get("type") == "call" and r.get("model"):
            key = f"{r.get('provider') or '?'}:{r['model']}"
            a = agg.setdefault(
                key, {"calls": 0, "ok": 0, "timeouts": 0, "errors": 0, "ms": [], "good": 0, "bad": 0}
            )
            a["calls"] += 1
            status = str(r.get("status") or "")
            if status == "ok":
                a["ok"] += 1
                if isinstance(r.get("ms"), (int, float)):
                    a["ms"].append(r["ms"])
            elif status == "timeout":
                a["timeouts"] += 1
            else:
                a["errors"] += 1
            if status == "ok" and r.get("job_id"):
                job_model[r["job_id"]] = key  # the call that answered the job
    for r in rows:
        if r.get("type") == "rating" and (key := job_model.get(r.get("job_id", ""))) and key in agg:
            verdict = r.get("verdict")
            if verdict in ("good", "bad"):
                agg[key][verdict] += 1
    out = {}
    for key, a in sorted(agg.items(), key=lambda kv: -kv[1]["calls"]):
        ok_rate = a["ok"] / a["calls"]
        out[key] = {
            "calls": a["calls"],
            "ok": a["ok"],
            "ok_rate": round(ok_rate, 3),
            "timeouts": a["timeouts"],
            "errors": a["errors"],
            "p50_ms": round(statistics.median(a["ms"])) if a["ms"] else None,
            "good": a["good"],
            "bad": a["bad"],
            "reliable": None if a["calls"] < MIN_CALLS else ok_rate >= UNRELIABLE_OK_RATE,
        }
    return out
