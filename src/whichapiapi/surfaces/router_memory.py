"""Short-term router memory (per server process; borrowed from freellmapi, MIT):

- **response cache** — opt-in (`x-whichapiapi-cache: 1` or WHICHAPIAPI_RESPONSE_CACHE=1), non-streamed calls with
  temperature ≤ 1; key = SHA-256 of the canonical request (model policy, messages, sampling params, tools,
  response_format, seed, reasoning_effort); TTL 1 h, LRU 5000. A hit costs nothing and is marked
  `x-whichapiapi-cache: hit`.
- **idempotency** — `Idempotency-Key` header on non-streamed calls: the same key + the same request replays the
  stored answer for 24 h; the same key with a different request → 409.
- **stickiness** — an `auto` conversation (same guest, same system prompt and first user message) stays on the
  route that answered it for 30 min while that route is healthy: consistent answers, and the provider's prompt
  cache keeps working (cached input is 10–20× cheaper).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any

CACHE_TTL_S = 3600
CACHE_MAX = 5000
IDEM_TTL_S = 86400
STICKY_TTL_S = 1800
_KEYS = ("messages", "temperature", "top_p", "n", "presence_penalty", "frequency_penalty", "logit_bias", "stop",
         "seed", "response_format", "reasoning_effort", "tools", "tool_choice", "max_tokens", "max_completion_tokens")  # fmt: skip

_cache: OrderedDict[str, tuple[float, int, dict[str, str], Any]] = OrderedDict()
_idem: dict[str, tuple[float, str, int, dict[str, str], Any]] = {}
_sticky: dict[str, tuple[float, str]] = {}


def fingerprint(body: dict[str, Any], guest: str | None) -> str:
    canon = {"model": body.get("model"), "guest": guest, **{k: body[k] for k in _KEYS if k in body}}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cache_wanted(body: dict[str, Any], header: str | None) -> bool:
    on = (header or "").strip() in ("1", "true", "yes") or os.environ.get("WHICHAPIAPI_RESPONSE_CACHE") == "1"
    temp = body.get("temperature")
    return on and not body.get("stream") and (temp is None or float(temp) <= 1.0)


def cache_get(key: str) -> tuple[int, dict[str, str], Any] | None:
    hit = _cache.get(key)
    if not hit or time.time() - hit[0] > CACHE_TTL_S:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return hit[1], {**hit[2], "x-whichapiapi-cache": "hit"}, hit[3]


def cache_put(key: str, status: int, headers: dict[str, str], data: Any) -> None:
    if status != 200:
        return
    _cache[key] = (
        time.time(),
        status,
        {k: v for k, v in headers.items() if k != "x-whichapiapi-cost-usd"},
        data,
    )
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


def idem_check(idem_key: str, guest: str | None, fp: str) -> tuple[int, dict[str, str], Any] | None:
    """A stored answer for this Idempotency-Key, a 409 when the key was used for another request, else None."""
    k = f"{guest}:{idem_key}"
    hit = _idem.get(k)
    if not hit or time.time() - hit[0] > IDEM_TTL_S:
        _idem.pop(k, None)
        return None
    if hit[1] != fp:
        return 409, {}, {"error": {"message": "Idempotency-Key reused with a different request",
                                   "type": "idempotency_key_conflict"}}  # fmt: skip
    return hit[2], {**hit[3], "x-whichapiapi-idempotent-replay": "1"}, hit[4]


def idem_put(
    idem_key: str, guest: str | None, fp: str, status: int, headers: dict[str, str], data: Any
) -> None:
    if status < 500:
        _idem[f"{guest}:{idem_key}"] = (time.time(), fp, status, headers, data)
    if len(_idem) > 20000:  # drop expired entries now and then
        now = time.time()
        for k in [k for k, v in _idem.items() if now - v[0] > IDEM_TTL_S]:
            _idem.pop(k, None)


def conversation_key(body: dict[str, Any], guest: str | None) -> str | None:
    """Same guest + same system prompt + same first user message = the same conversation."""
    msgs = body.get("messages") or []
    system = next((m.get("content") for m in msgs if m.get("role") in ("system", "developer")), "")
    first = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
    if first is None or len([m for m in msgs if m.get("role") == "user"]) < 2:
        return None  # a single-turn request isn't a conversation yet; nothing to stick to
    raw = json.dumps([guest, body.get("model"), system, first], ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def sticky_route(conv: str | None) -> str | None:
    if not conv:
        return None
    hit = _sticky.get(conv)
    if not hit or time.time() - hit[0] > STICKY_TTL_S:
        _sticky.pop(conv, None)
        return None
    return hit[1]


def remember_route(body: dict[str, Any], guest: str | None, route: str) -> None:
    """Called after a successful answer; keyed so the conversation's next turn finds it."""
    msgs = body.get("messages") or []
    system = next((m.get("content") for m in msgs if m.get("role") in ("system", "developer")), "")
    first = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
    if first is None:
        return
    raw = json.dumps([guest, body.get("model"), system, first], ensure_ascii=False, default=str)
    _sticky[hashlib.sha256(raw.encode()).hexdigest()] = (time.time(), route)
    if len(_sticky) > 20000:
        now = time.time()
        for k in [k for k, v in _sticky.items() if now - v[0] > STICKY_TTL_S]:
            _sticky.pop(k, None)
