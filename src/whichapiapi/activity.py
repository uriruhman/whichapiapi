"""Activity log: every MCP tool call, REST request and CLI command, for reviewing how the project is used and where
it fails. Local only: JSON lines under `$WHICHAPIAPI_HOME/logs/activity-YYYY-MM.jsonl` (mode 600). Arguments are
kept (long values truncated, key-like strings redacted, workflow bodies reduced to their size); results are summarized.
`WHICHAPIAPI_ACTIVITY_LOG=0` turns it off.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from whichapiapi.transports.base import redact

_SECRET_WORDS = ("key", "token", "secret", "password", "authorization")
_MAX_STR = 300


def log_dir() -> Path:
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    return home / "logs"


def enabled() -> bool:
    return os.environ.get("WHICHAPIAPI_ACTIVITY_LOG", "1") not in ("0", "false", "no")


def _clean(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict):
        if depth > 3:
            return f"<dict {len(value)} keys>"
        if "nodes" in value and isinstance(value.get("nodes"), list):  # an n8n workflow: size, not content
            return f"<workflow {value.get('name', '?')!r}: {len(value['nodes'])} nodes>"
        return {
            k: (
                "<redacted>"
                if any(w in str(k).lower() for w in _SECRET_WORDS) and not str(k).endswith("_env")
                else _clean(v, depth + 1)
            )
            for k, v in list(value.items())[:40]
        }
    if isinstance(value, list | tuple):
        return [_clean(v, depth + 1) for v in list(value)[:20]] + (
            [f"<+{len(value) - 20}>"] if len(value) > 20 else []
        )
    if isinstance(value, str):
        text = redact(value)
        return text if len(text) <= _MAX_STR else f"{text[:_MAX_STR]}…<{len(text)} chars>"
    return value


def summarize(result: Any) -> dict[str, Any]:
    """Small, useful digest of a result: sizes and the headline fields, never the whole payload."""
    if isinstance(result, dict):
        out: dict[str, Any] = {"keys": sorted(result)[:15]}
        for k in ("count", "api_calls", "findings_by_kind", "potential_saving_monthly", "spent_real", "alerts",
                  "run_id", "error", "preset", "task", "status", "route", "cost_usd"):  # fmt: skip
            if k in result:
                out[k] = _clean(result[k])
        if isinstance(result.get("models"), list) and result["models"]:
            out["top"] = [m.get("model") or m.get("id") for m in result["models"][:3] if isinstance(m, dict)]
        if isinstance(result.get("offers"), list):
            out["offers"] = len(result["offers"])
        return out
    if isinstance(result, list):
        return {"items": len(result)}
    text = str(result)
    return {"chars": len(text), "head": _clean(text[:200])}


def log_event(
    surface: str,
    action: str,
    args: Any = None,
    *,
    ok: bool = True,
    duration_ms: float | None = None,
    result: Any = None,
    error: str | None = None,
    client: str | None = None,
) -> None:
    if not enabled():
        return
    now = datetime.now(UTC)
    event = {
        "ts": now.isoformat(timespec="milliseconds"),
        "surface": surface,  # mcp | rest | cli
        "action": action,
        "project": os.getcwd(),
        "client": client,
        "pid": os.getpid(),
        "ok": ok,
        "ms": round(duration_ms, 1) if duration_ms is not None else None,
        "args": _clean(args or {}),
        "result": summarize(result) if result is not None else None,
        "error": redact(error)[:500] if error else None,
    }
    try:
        d = log_dir()
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = d / f"activity-{now:%Y-%m}.jsonl"
        new = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if new:
            path.chmod(0o600)
    except OSError:
        pass  # logging must never break the tool


@contextmanager
def track(surface: str, action: str, args: Any = None, client: str | None = None) -> Iterator[dict[str, Any]]:
    """`with track("rest", "rank", args) as t: t["result"] = ...` — logs duration, result digest or the error."""
    box: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        yield box
    except BaseException as e:
        log_event(surface, action, args, ok=False, duration_ms=(time.perf_counter() - t0) * 1000,
                  error=f"{type(e).__name__}: {e}", client=client)  # fmt: skip
        raise
    log_event(surface, action, args, ok=box.get("ok", True), duration_ms=(time.perf_counter() - t0) * 1000,
              result=box.get("result"), error=box.get("error"), client=client)  # fmt: skip


def read_events(days: float = 7.0) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    out = []
    d = log_dir()
    if not d.exists():
        return out
    for path in sorted(d.glob("activity-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if datetime.fromisoformat(e["ts"]) >= since:
                out.append(e)
    return out


def report(days: float = 7.0) -> dict[str, Any]:
    """Per action: calls, errors, median/max duration; per project and client; the latest errors."""
    events = read_events(days)
    by_action: dict[str, dict[str, Any]] = {}
    for e in events:
        a = by_action.setdefault(f"{e['surface']}:{e['action']}", {"calls": 0, "errors": 0, "ms": []})
        a["calls"] += 1
        a["errors"] += 0 if e.get("ok") else 1
        if e.get("ms") is not None:
            a["ms"].append(e["ms"])
    for a in by_action.values():
        ms = sorted(a.pop("ms"))
        a["p50_ms"] = ms[len(ms) // 2] if ms else None
        a["max_ms"] = ms[-1] if ms else None
    count: dict[str, dict[str, int]] = {"project": {}, "client": {}}
    for e in events:
        for k in count:
            v = str(e.get(k) or "—")
            count[k][v] = count[k].get(v, 0) + 1
    return {
        "days": days,
        "events": len(events),
        "errors": sum(1 for e in events if not e.get("ok")),
        "by_action": dict(sorted(by_action.items(), key=lambda kv: -kv[1]["calls"])),
        "by_project": count["project"],
        "by_client": count["client"],
        "recent_errors": [
            {k: e.get(k) for k in ("ts", "surface", "action", "project", "error", "args")}
            for e in events
            if not e.get("ok")
        ][-10:],
    }


def router_usage(days: float = 7.0) -> dict[str, Any]:
    """Router traffic per route and per key: calls, failures, median latency, charged USD (from the activity log)."""
    events = [e for e in read_events(days) if e.get("surface") == "router"]
    by_route: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for e in events:
        res = e.get("result") or {}
        route = res.get("route") or ("(failed)" if not e.get("ok") else "(unknown, old log)")
        key = (e.get("args") or {}).get("caller") or (e.get("args") or {}).get("key") or "owner"
        for bucket, name in ((by_route, route), (by_key, key)):
            b = bucket.setdefault(name, {"calls": 0, "failed": 0, "cost_usd": 0.0, "ms": []})
            b["calls"] += 1
            b["failed"] += 0 if e.get("ok") else 1
            b["cost_usd"] += res.get("cost_usd") or 0.0
            if e.get("ms") is not None:
                b["ms"].append(e["ms"])
    for bucket in (by_route, by_key):
        for b in bucket.values():
            ms = sorted(b.pop("ms"))
            b["p50_ms"] = round(ms[len(ms) // 2]) if ms else None
            b["cost_usd"] = round(b["cost_usd"], 6)
    return {"days": days, "calls": len(events), "by_route": by_route, "by_key": by_key}
