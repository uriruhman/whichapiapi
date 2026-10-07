"""What the router remembers about each route (`channel:model`) between requests: cooldowns, penalties, reliability
and latency. Shared by every server process through `$WHICHAPIAPI_HOME/route-health.json` (mode 600, file-locked).

Error classes (mechanism borrowed from freellmapi's fallback loop, MIT; thresholds are theirs unless noted):

| upstream says                         | class        | effect                                                    |
|---------------------------------------|--------------|-----------------------------------------------------------|
| 401                                   | auth         | route cools 5 min (the key is probably wrong)             |
| 402 / "insufficient balance|quota"    | payment      | the whole channel cools 24 h                              |
| 403 + suspended/banned/disabled       | suspended    | the whole channel cools 24 h                              |
| 403 otherwise                         | forbidden    | route cools 24 h (model not allowed for this key)         |
| 404 / 410 / "model not found"         | gone         | route cools 1 h, 24 h after the second time within 1 h    |
| 413 / "context length"                | too_large    | no penalty, just try the next route                       |
| 429 + "daily|per day|quota exceeded"  | daily        | route cools until the next UTC midnight                   |
| 429                                   | rate         | 90 s → 5 min → 15 min → 1 h → 24 h on repeated 429s       |
| 5xx / 408 / timeout / connection      | transient    | penalty +1; 60 s cooldown after 3 failures in a row       |

`Retry-After` (seconds or HTTP date) and `x-ratelimit-remaining-* = 0` with a reset time win when they ask for longer.
Penalty: +3 per 429/402, +1 per transient failure, at most 10, −1 every 2 min and −1 per success. Reliability is a
success rate with a 2-day half-life (Beta(1,1) prior), used deterministically — no random exploration.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

HALF_LIFE_S = 2 * 86400
PENALTY = {"rate": 3, "daily": 3, "payment": 3, "transient": 1}
MAX_PENALTY = 10
DECAY_S = 120
RATE_LADDER = [90, 300, 900, 3600, 86400]
TRANSIENT_STREAK = 3
LATENCY_KEEP = 50

_DAILY = re.compile(
    r"daily|per[ -]day|requests per day|quota (?:has been )?exceeded|exceeded your current quota", re.I
)
_PAYMENT = re.compile(r"insufficient (?:balance|quota|credit|funds)|余额不足|payment required|top ?up", re.I)
_SUSPENDED = re.compile(r"suspend|banned|disabled|deactivated|account.*(?:locked|blocked)", re.I)
_GONE = re.compile(
    r"model[^.]{0,40}(?:not found|does not exist|not available|decommission|deprecated)|no such model", re.I
)
_TOO_LARGE = re.compile(
    r"context[_ ]length|too many tokens|maximum context|prompt is too long|too large", re.I
)


def _path() -> Path:
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    return home / "route-health.json"


@contextlib.contextmanager
def _locked():
    p = _path()
    with open(p.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            data = {}
        yield data
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.chmod(tmp, 0o600)
        tmp.replace(p)


def _read() -> dict[str, Any]:
    p = _path()
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def classify(status: int, text: str = "") -> str:
    """Error class of an upstream answer (see the module table). `status` 0 = timeout / connection error."""
    t = text or ""
    if status == 0 or status in (408, 425) or status >= 500:
        return "transient"
    if status == 401:
        return "auth"
    if status == 402 or (status in (400, 403, 429) and _PAYMENT.search(t)):
        return "payment"
    if status == 403:
        return "suspended" if _SUSPENDED.search(t) else "forbidden"
    if status in (404, 410) or (status == 400 and _GONE.search(t)):
        return "gone"
    if status == 413 or (status == 400 and _TOO_LARGE.search(t)):
        return "too_large"
    if status == 429:
        return "daily" if _DAILY.search(t) else "rate"
    return "client"  # other 4xx: the request itself is wrong — no fallback, no penalty


FALLBACK = {"auth", "payment", "suspended", "forbidden", "gone", "too_large", "daily", "rate", "transient"}


def retry_after_s(headers: Any, now: float | None = None) -> float | None:
    """Seconds the upstream asked us to wait: `Retry-After`, or a zero `x-ratelimit-remaining-*` with its reset."""
    if not headers:
        return None
    now = now or time.time()
    get = headers.get
    ra = get("retry-after")
    if ra:
        with contextlib.suppress(ValueError):
            return max(0.0, float(ra))
        with contextlib.suppress(TypeError, ValueError):
            return max(0.0, parsedate_to_datetime(ra).timestamp() - now)
    for kind in ("requests", "tokens"):
        if get(f"x-ratelimit-remaining-{kind}") in ("0", 0):
            reset = parse_duration(get(f"x-ratelimit-reset-{kind}"))
            if reset is not None:
                return reset
    return None


def parse_duration(v: Any) -> float | None:
    """'2m59.56s' / '1h2m' / '7.5s' / '30' (seconds) / a unix timestamp → seconds from now."""
    if v is None:
        return None
    s = str(v).strip()
    with contextlib.suppress(ValueError):
        x = float(s)
        return max(0.0, x - time.time()) if x > 1e9 else x
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", s)
    if not parts:
        return None
    mult = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(n) * mult[u] for n, u in parts)


def _secs_to_utc_midnight(now: float) -> float:
    d = datetime.fromtimestamp(now, UTC)
    nxt = (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - d).total_seconds()


def _decayed_penalty(r: dict[str, Any], now: float) -> float:
    p = float(r.get("penalty", 0))
    if p and r.get("penalty_at"):
        p = max(0.0, p - (now - r["penalty_at"]) // DECAY_S)
    return p


def _decay(r: dict[str, Any], now: float) -> None:
    """Age success/failure counts by the half-life since the last update."""
    last = r.get("seen_at")
    if last:
        f = 0.5 ** ((now - last) / HALF_LIFE_S)
        r["ok"] = r.get("ok", 0.0) * f
        r["fail"] = r.get("fail", 0.0) * f
    r["seen_at"] = now


def record(
    route: str,
    status: int,
    text: str = "",
    headers: Any = None,
    ttfb_ms: float | None = None,
    latency_ms: float | None = None,
    now: float | None = None,
    overhead: int | None = None,
) -> str:
    """Remember one attempt's outcome. Returns its error class ("ok" for a success)."""
    now = now or time.time()
    kind = "ok" if 200 <= status < 400 else classify(status, text)
    channel = route.partition(":")[0]
    wait = retry_after_s(headers, now)
    with _locked() as data:
        r = data.setdefault(route, {})
        _decay(r, now)
        if kind == "ok":
            r["ok"] = r.get("ok", 0.0) + 1
            r["streak"] = 0
            r["rate_n"] = 0
            r["penalty"] = max(0.0, _decayed_penalty(r, now) - 1)
            r["penalty_at"] = now
            for key, val in (("ttfb", ttfb_ms), ("lat", latency_ms), ("ovh", overhead)):
                if val is not None:
                    r[key] = [*r.get(key, []), round(val)][-LATENCY_KEEP:]
            if wait is not None and wait > 0:  # a success that says "you have 0 left until reset"
                r["until"] = max(r.get("until", 0), now + wait)
                r["why"] = "rate limit headroom 0"
            return kind
        if kind in ("client", "too_large"):
            return kind
        r["fail"] = r.get("fail", 0.0) + 1
        r["streak"] = r.get("streak", 0) + 1
        r["last_error"] = {"at": now, "status": status, "class": kind, "text": (text or "")[:160]}
        r["penalty"] = min(MAX_PENALTY, _decayed_penalty(r, now) + PENALTY.get(kind, 0))
        r["penalty_at"] = now
        cool = 0.0
        if kind == "auth":
            cool = 300
        elif kind in ("payment", "suspended"):
            ch = data.setdefault(f"{channel}:*", {})
            ch["until"] = max(ch.get("until", 0), now + 86400)
            ch["why"] = kind
        elif kind == "forbidden":
            cool = 86400
        elif kind == "gone":
            recent = [t for t in r.get("gone_at", []) if now - t < 3600] + [now]
            r["gone_at"] = recent[-3:]
            cool = 86400 if len(recent) >= 2 else 3600
        elif kind == "daily":
            cool = _secs_to_utc_midnight(now)
        elif kind == "rate":
            n = r.get("rate_n", 0)
            cool = RATE_LADDER[min(n, len(RATE_LADDER) - 1)]
            r["rate_n"] = n + 1
        elif kind == "transient" and r["streak"] >= TRANSIENT_STREAK:
            cool = 60
        cool = max(cool, wait or 0)
        if cool:
            r["until"] = max(r.get("until", 0), now + cool)
            r["why"] = kind
    return kind


def cooling(route: str, data: dict[str, Any] | None = None, now: float | None = None) -> float:
    """Seconds left on this route's (or its channel's) cooldown; 0 when it may be used."""
    now = now or time.time()
    data = _read() if data is None else data
    until = max(
        (data.get(route) or {}).get("until", 0),
        (data.get(route.partition(":")[0] + ":*") or {}).get("until", 0),
    )
    return max(0.0, until - now)


def reliability(r: dict[str, Any], now: float | None = None) -> tuple[float, float]:
    """(expected success rate, effective sample size) after the half-life decay."""
    now = now or time.time()
    f = 0.5 ** ((now - r.get("seen_at", now)) / HALF_LIFE_S)
    ok, fail = r.get("ok", 0.0) * f, r.get("fail", 0.0) * f
    return (ok + 1) / (ok + fail + 2), ok + fail


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return float(s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)])


def order(routes: list[str], now: float | None = None) -> list[str]:
    """Candidates in the order to try: routes not cooling first, ranked by their original position plus half the
    penalty, minus a bonus for proven reliability; cooling routes last, the soonest-available first."""
    now = now or time.time()
    data = _read()

    def key(item: tuple[int, str]) -> tuple[int, float, float]:
        i, route = item
        left = cooling(route, data, now)
        if left:
            return 1, left, 0.0
        r = data.get(route) or {}
        rel, n = reliability(r, now)
        demote = 2.0 if n >= 5 and rel < 0.5 else 0.0
        pen = _decayed_penalty(r, now)
        return 0, i + pen / 2 + demote, pen  # on a tie the less penalised route goes first

    return [r for _, r in sorted(enumerate(routes), key=key)]


def attempt_timeout_s(route: str, more_left: bool, stream: bool, default: float = 300.0) -> float:
    """How long to wait for this route before moving on. Only shortened while other candidates remain: first token
    (stream) within max(20 s, 2 × its p95 TTFB); whole answer (non-stream) within max(90 s, 3 × its p95 latency)."""
    if not more_left:
        return default
    r = _read().get(route) or {}
    if stream:
        p = p95(r.get("ttfb", []))
        return min(default, max(20.0, 2 * p / 1000)) if p else min(default, 45.0)
    p = p95(r.get("lat", []))
    return min(default, max(90.0, 3 * p / 1000)) if p else default


def snapshot(routes: list[str] | None = None, now: float | None = None) -> list[dict[str, Any]]:
    """Health of known routes (or the given ones) for `routing_info` / `provider_health`."""
    now = now or time.time()
    data = _read()
    out = []
    for route in routes or sorted(k for k in data if not k.endswith(":*")):
        r = data.get(route) or {}
        rel, n = reliability(r, now)
        out.append(
            {
                "route": route,
                "cooling_s": round(cooling(route, data, now)),
                "why": r.get("why") if cooling(route, data, now) else None,
                "penalty": round(_decayed_penalty(r, now), 1),
                "reliability": round(rel, 3),
                "samples": round(n, 1),
                "ttfb_p95_ms": p95(r.get("ttfb", [])),
                "latency_p95_ms": p95(r.get("lat", [])),
                # billed prompt tokens minus a local count of the request: a channel adding a hidden prompt
                "hidden_prompt_tokens_median": sorted(r["ovh"])[len(r["ovh"]) // 2] if r.get("ovh") else None,
                "last_error": r.get("last_error"),
            }
        )
    for key, r in data.items():
        if key.endswith(":*") and r.get("until", 0) > now:
            out.append({"route": key, "cooling_s": round(r["until"] - now), "why": r.get("why")})
    return out
