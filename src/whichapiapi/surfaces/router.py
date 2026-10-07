"""Per-request routing: an OpenAI-compatible `/v1/chat/completions` that picks the model for each request.

`model` selects the policy: `auto` (task detected from the request), `auto:<task>` (general, coding, math, agents,
russian, writing, long_context, vision) and optionally `@<preset>` (optimal, value, best, fast, …) — e.g.
`auto:coding@value`. Candidates come from `core.suggest_candidates`: benchmark-weighted quality for that task, cost per
task including thinking tokens, one per creator, and only models the user's own keys can call. The request goes to the
first candidate; on a rate limit, a 5xx, a timeout or a connection error it moves on to the next (another vendor).
Any other `model` value is passed through to the first reachable channel unchanged.

Unlike a router that follows what other people spend on (OpenRouter's community signal) or a model trained on someone
else's traffic, the choice here is explicit, explainable and reproducible: the response carries
`x-whichapiapi-route` (what answered) and `x-whichapiapi-task` (what the request was classified as).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from whichapiapi import activity, channels, core, keypool, keys, route_health, tracing
from whichapiapi.cost.engine import call_cost
from whichapiapi.evaluator import tokenizers
from whichapiapi.surfaces import rescue
from whichapiapi.surfaces import router_memory as memory

_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}
CACHE_S = 600.0
MIN_OUTPUT_TOKENS = 1024
GUEST_MAX_TOKENS = 4096  # per-call answer cap for invited keys: bounds the cost of one request
UNPRICED_CHARGE = 0.01  # charged to a guest key when the answering route has no known price

_CODE = re.compile(
    r"```|\bdef |\bfunction\b|\bclass \w+|Traceback|\bSELECT\b.*\bFROM\b|=>|\bimport \w+|#include", re.S
)
_MATH = re.compile(
    r"\\frac|\\int|\bsolve\b|\bequation\b|\bintegral\b|\bderivative\b|\bprove\b|\d+\s*[\^*/]\s*\d+", re.I
)


def _text(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    parts, image = [], False
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_image", "image"):
                    image = True
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
    return "\n".join(parts), image


def classify(body: dict[str, Any]) -> str:
    """Cheap, deterministic task detection (no model call): images → vision, tools → agents, very long → long
    documents, code-looking → coding, formulas → math, mostly Cyrillic → russian, else general."""
    text, image = _text(body.get("messages") or [])
    if image:
        return "vision"
    if body.get("tools") or body.get("functions"):
        return "agents"
    if len(text) > 60_000:
        return "long_context"
    if _CODE.search(text):
        return "coding"
    if _MATH.search(text):
        return "math"
    letters = [c for c in text if c.isalpha()]
    if letters and sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters) > 0.3:
        return "russian"
    return "general"


def parse_model(model: str, body: dict[str, Any]) -> tuple[str | None, str]:
    """("coding", "value") from "auto:coding@value"; (None, "") when `model` is a concrete model id."""
    if not model.startswith("auto"):
        return None, ""
    policy, _, preset = model.partition("@")
    task = policy.partition(":")[2] or classify(body)
    return task, preset or "optimal"


def candidates(task: str, preset: str) -> list[str]:
    """`preset` "free": free-tier models on your own provider keys first (keypool), then the optimal paid ones."""
    key = (task, preset)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    if preset == "free":
        got = core.free_candidates(task, n=3) + core.suggest_candidates(task, n=3, preset="optimal")
    else:
        got = core.suggest_candidates(task, n=3, preset=preset)
    _CACHE[key] = (time.time(), got)
    return got


def _target(candidate: str) -> tuple[str, str, str, str | None] | None:
    """(base URL, key, upstream model id, keypool key id or None for env keys)."""
    channel, _, model = candidate.partition(":")
    key = os.environ.get(channels.key_env(channel) or "", "")
    url = channels.base_url(channel)
    if url and key:
        return url, key, model, None
    picked = keypool.pick(channel)
    url = keypool.base_url(channel)
    return (url, picked[1], model, picked[0]) if picked and url else None


def _passthrough_targets(model: str) -> list[str]:
    env = channels.with_key()
    return [f"{c}:{model}" for c in env + sorted(keypool.providers() - set(env))][:1]


def route_cost(candidate: str, usage: Any) -> float | None:
    """USD for one answered call: list price of the route × the learned real/list ratio (1 when unknown)."""
    channel, _, model = candidate.partition(":")
    if not isinstance(usage, dict):
        return None
    st = core._store(None)
    offer = st.get_offer(f"{channel}/{model}")
    if offer is None or not offer.price.is_known:
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cost = call_cost(
        offer.price,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(cached),
    )
    ratio = (st.cache_get(f"ratio:{channel}/{model}") or {}).get("max") or 1.0
    return None if cost is None else cost * float(ratio)


def _charge(guest: str | None, candidate: str, usage: Any) -> float | None:
    cost = route_cost(candidate, usage)
    if guest:
        keys.charge(guest, UNPRICED_CHARGE if cost is None else cost)
    return cost


async def complete(
    body: dict[str, Any],
    client: httpx.AsyncClient | None = None,
    guest: str | None = None,
    trace: bool = False,
    request_headers: Any = None,
) -> tuple[int, dict[str, str], Any]:
    """Route one chat completion, with the short-term memory of `router_memory`: Idempotency-Key replays, the opt-in
    response cache, conversation stickiness, and the request-size guard (WHICHAPIAPI_MAX_REQUEST_TOKENS)."""
    hdr = request_headers or {}
    limit = int(os.environ.get("WHICHAPIAPI_MAX_REQUEST_TOKENS") or 0)
    if limit and len(json.dumps(body.get("messages") or [], ensure_ascii=False)) // 4 > limit:
        return (
            413,
            {},
            {"error": {"message": f"request is over ~{limit} tokens", "type": "request_too_large"}},
        )
    fp = memory.fingerprint(body, guest)
    idem = None if body.get("stream") else hdr.get("idempotency-key")
    if idem and (replay := memory.idem_check(idem[:255], guest, fp)):
        return replay
    cached = memory.cache_wanted(body, hdr.get("x-whichapiapi-cache"))
    if cached and (hit := memory.cache_get(fp)):
        _log(
            str(body.get("model")),
            None,
            hit[1].get("x-whichapiapi-route"),
            [],
            hit[0],
            None,
            None,
            guest,
            0.0,
        )
        return hit
    task, preset = parse_model(str(body.get("model") or "auto"), body)
    if preset == "fusion":
        if guest or body.get("stream") or body.get("tools"):
            return 400, {}, {"error": {"message": "@fusion: owner only, without stream or tools"}}
        return await fuse(body, task or "general", client)
    sticky = memory.sticky_route(memory.conversation_key(body, guest))
    status, headers, data = await _complete(body, client, guest, trace, prefer=sticky)
    route = headers.get("x-whichapiapi-route")
    if status < 400 and route:
        memory.remember_route(body, guest, route)
        if sticky == route:
            headers = {**headers, "x-whichapiapi-sticky": "1"}
        if cached:
            memory.cache_put(fp, status, headers, data)
    if idem:
        memory.idem_put(idem[:255], guest, fp, status, headers, data)
    return status, headers, data


async def _complete(
    body: dict[str, Any],
    client: httpx.AsyncClient | None = None,
    guest: str | None = None,
    trace: bool = False,
    prefer: str | None = None,
    only: str | None = None,
) -> tuple[int, dict[str, str], Any]:
    """Route one chat completion. Returns (status, extra headers, JSON body or an async byte iterator for streams).
    `guest` = the name of an invited key: only `auto` policies, answers capped at GUEST_MAX_TOKENS, cost charged.

    Candidates are ordered by route health (cooldowns, penalties, reliability — see route_health). Each failed
    attempt is classified: rate limits, 5xx, timeouts, auth/payment/forbidden/gone and too-large errors move on to the
    next candidate; other 4xx are the client's fault and are returned at once. While other candidates remain, an
    attempt has a deadline (first token for streams, whole answer otherwise); a stream is only committed to the client
    once its first real chunk arrived, so an upstream that fails before that is replaced invisibly."""
    model = str(body.get("model") or "auto")
    task, preset = parse_model(model, body)
    if guest and not task:
        return (
            400,
            {},
            {"error": {"message": "invited keys can use model 'auto' or 'auto:<task>[@<preset>]' only"}},
        )
    route = [only] if only else candidates(task, preset) if task else _passthrough_targets(model)
    route = [c for c in route_health.order(route) if _target(c) is not None]
    if not route:
        return (
            503,
            {},
            {"error": {"message": "no reachable model: add a provider key (see ~/.config/secrets)"}},
        )
    ready = [c for c in route if not route_health.cooling(c)]
    route = ready or route[:1]  # everything cooling: try the one that frees up first
    if prefer in route:  # the conversation's previous route, while healthy
        route = [prefer] + [c for c in route if c != prefer]
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
    tried: list[str] = []
    trail: list[str] = []
    last: tuple[int, Any] = (502, {"error": {"message": "all candidates failed"}})
    started = time.perf_counter()
    for i, cand in enumerate(route):
        url, key, upstream_model, key_id = _target(cand)  # type: ignore[misc]
        tried.append(cand)
        more_left = i < len(route) - 1
        stream = bool(body.get("stream"))
        deadline = route_health.attempt_timeout_s(cand, more_left, stream)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {**body, "model": upstream_model}
        if stream:
            payload["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        note = ""
        for field in (
            "max_tokens",
            "max_completion_tokens",
        ):  # a tiny cap is eaten by reasoning → empty answer
            if task and isinstance(body.get(field), int) and body[field] < MIN_OUTPUT_TOKENS:
                payload[field], note = (
                    MIN_OUTPUT_TOKENS,
                    f"{field} raised to {MIN_OUTPUT_TOKENS} (reasoning models)",
                )
        if guest:
            fields = [f for f in ("max_tokens", "max_completion_tokens") if isinstance(payload.get(f), int)]
            for f in fields or ["max_tokens"]:
                if not isinstance(payload.get(f), int) or payload[f] > GUEST_MAX_TOKENS:
                    payload[f] = GUEST_MAX_TOKENS

        def meta(cand: str = cand, note: str = note) -> dict[str, str]:
            m = {
                "x-whichapiapi-route": cand,
                "x-whichapiapi-task": task or "passthrough",
                "x-whichapiapi-tried": ",".join(tried),
                "x-whichapiapi-trail": ";".join(trail),
            }
            if note:
                m["x-whichapiapi-note"] = note
            return m

        t0 = time.perf_counter()
        if stream:
            outcome = await _open_stream(http, url, payload, headers, deadline)
            ms = (time.perf_counter() - t0) * 1000
            if outcome[0] == "ok":
                resp, primed = outcome[1], outcome[2]
                route_health.record(cand, resp.status_code, headers=resp.headers, ttfb_ms=ms)
                trail.append(f"{cand}={resp.status_code}/{round(ms)}ms")
                traced = (guest, body) if trace and guest else None
                return (
                    resp.status_code,
                    meta(),
                    _stream(resp, primed, model, task, cand, list(tried), t0, guest, traced),
                )
            status, text, rheaders = outcome[1], outcome[2], outcome[3]
            kind = route_health.record(cand, status, text, rheaders)
            if key_id:
                keypool.report(key_id, status, text)
            trail.append(f"{cand}={status or 'timeout'}:{kind}/{round(ms)}ms")
            if kind == "client":
                _log(model, task, None, tried, status, t0, None, guest)
                return status, meta(), _error_body(text)
            last = (status or 504, _error_body(text))
            continue
        try:
            resp = await http.post(
                f"{url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(deadline, connect=15.0),
            )
        except httpx.HTTPError as e:
            ms = (time.perf_counter() - t0) * 1000
            kind = route_health.record(cand, 0, type(e).__name__)
            trail.append(f"{cand}=timeout:{kind}/{round(ms)}ms")
            last = (
                504 if isinstance(e, httpx.TimeoutException) else 502,
                {"error": {"message": f"{cand}: {type(e).__name__}"}},
            )
            continue
        ms = (time.perf_counter() - t0) * 1000
        data = (
            _json_or_none(resp)
            if resp.headers.get("content-type", "").startswith("application/json")
            else {"raw": resp.text}
        )
        status = resp.status_code
        if status < 400 and isinstance(data, dict) and data.get("error") and not data.get("choices"):
            status = 502  # an error inside a 200: treat it as an upstream failure
        if status >= 400:
            text = resp.text[:500]
            kind = route_health.record(cand, status, text, resp.headers)
            if key_id:
                keypool.report(key_id, status, text)
            trail.append(f"{cand}={status}:{kind}/{round(ms)}ms")
            if kind == "client":
                _log(model, task, cand, tried, status, t0, None, guest)
                return status, meta(), data
            last = (status, data if isinstance(data, dict) else {"error": {"message": text[:300]}})
            continue
        usage = data.get("usage") if isinstance(data, dict) else None
        route_health.record(
            cand, status, headers=resp.headers, latency_ms=ms, overhead=_overhead(body, cand, usage)
        )
        trail.append(f"{cand}={status}/{round(ms)}ms")
        out = meta()
        repaired = rescue.repair(body, data)
        if repaired:
            out["x-whichapiapi-repaired"] = ",".join(repaired)
        cost = _charge(guest, cand, usage)
        if cost is not None:
            out["x-whichapiapi-cost-usd"] = f"{cost:.6f}"
        _log(model, task, cand, tried, status, t0, usage, guest, cost, trail)
        if trace and guest:
            msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
            _trace(guest, body, task, cand, tried, status, t0, usage, cost, msg.get("content"), data, trail)
        return status, out, data
    _log(model, task, None, tried, last[0], started, None, guest, None, trail)
    return last[0], {"x-whichapiapi-tried": ",".join(tried), "x-whichapiapi-trail": ";".join(trail)}, last[1]


def _overhead(body: dict[str, Any], cand: str, usage: Any) -> int | None:
    """Billed prompt tokens minus a local count of the request (free passive check for hidden channel prompts)."""
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int) or body.get("tools"):
        return None
    family = tokenizers.family_of(cand)
    expected = tokenizers.chat_prompt_tokens(body.get("messages") or [], family) if family else None
    return None if expected is None else usage["prompt_tokens"] - expected


def _json_or_none(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


def _error_body(text: str) -> dict[str, Any]:
    with contextlib.suppress(ValueError):
        data = json.loads(text)
        if isinstance(data, dict) and "error" in data:
            return data
    return {"error": {"message": (text or "upstream failed")[:300]}}


def _first_real(line: bytes) -> str | None:
    """'ok' for an SSE line that carries the answer (text, reasoning, a tool call, a finish, [DONE]); 'error' for an
    in-band error event; None for keep-alives and empty role chunks."""
    if line.startswith(b"data: [DONE]"):
        return "ok"
    if not line.startswith(b"data: {"):
        return None
    try:
        event = json.loads(line[6:])
    except ValueError:
        return None
    if isinstance(event, dict) and event.get("error"):
        return "error"
    for ch in event.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
            return "ok"
        if ch.get("finish_reason"):
            return "ok"
    return None


async def _open_stream(
    http: httpx.AsyncClient, url: str, payload: dict[str, Any], headers: dict[str, str], deadline: float
) -> tuple[Any, ...]:
    """Open an upstream stream and wait for its first real chunk within `deadline` seconds.
    ("ok", response, primed bytes) or ("fail", status (0 = timeout/connection), error text, response headers)."""
    req = http.build_request("POST", f"{url}/chat/completions", json=payload, headers=headers)
    try:
        resp = await asyncio.wait_for(http.send(req, stream=True), deadline)
    except (TimeoutError, httpx.HTTPError) as e:
        return "fail", 0, type(e).__name__, None
    if resp.status_code >= 400:
        text = (await resp.aread()).decode(errors="replace")[:500]
        await resp.aclose()
        return "fail", resp.status_code, text, resp.headers
    primed = b""
    it = resp.aiter_bytes()
    loop_end = time.perf_counter() + deadline
    try:
        while True:
            left = loop_end - time.perf_counter()
            if left <= 0:
                raise TimeoutError
            try:
                chunk = await asyncio.wait_for(it.__anext__(), left)
            except StopAsyncIteration:
                await resp.aclose()
                return "fail", 502, "stream ended before any answer", resp.headers
            primed += chunk
            for line in primed.split(b"\n"):
                state = _first_real(line.strip())
                if state == "error":
                    await resp.aclose()
                    return "fail", 502, line.decode(errors="replace")[6:500], resp.headers
                if state == "ok":
                    return "ok", _Primed(resp, it), primed
    except (TimeoutError, httpx.HTTPError) as e:
        await resp.aclose()
        return "fail", 0, f"no first token within {round(deadline)} s ({type(e).__name__})", None


class _Primed:
    """An upstream response whose first chunks were already read: `aiter_raw` continues the same iterator."""

    def __init__(self, resp: httpx.Response, it: AsyncIterator[bytes]):
        self._resp, self._it = resp, it
        self.status_code, self.headers = resp.status_code, resp.headers

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return self._it

    async def aclose(self) -> None:
        await self._resp.aclose()


async def _stream(
    resp: Any,
    primed: bytes,
    model: str,
    task: str | None,
    cand: str,
    tried: list[str],
    t0: float,
    guest: str | None,
    traced: tuple[str, dict[str, Any]] | None = None,
) -> AsyncIterator[bytes]:
    """Pass the SSE bytes through untouched; read the final `usage` chunk (stream_options.include_usage) to charge,
    and collect the answer's text deltas when the call is traced."""
    usage, tail, parts = None, b"", []
    try:
        async for chunk in _chain(primed, resp.aiter_bytes()):
            yield chunk
            tail = (tail + chunk)[-65536:]
            *lines, tail = tail.split(b"\n")
            for line in lines:
                if not line.startswith(b"data: {"):
                    continue
                with contextlib.suppress(ValueError, AttributeError, IndexError, TypeError):
                    event = json.loads(line[6:])
                    usage = event.get("usage") or usage
                    if traced and event.get("choices"):
                        parts.append((event["choices"][0].get("delta") or {}).get("content") or "")
    finally:
        await resp.aclose()
        cost = _charge(guest, cand, usage) if resp.status_code < 400 else None
        _log(model, task, cand, tried, resp.status_code, t0, usage, guest, cost)
        if traced:
            _trace(
                traced[0],
                traced[1],
                task,
                cand,
                tried,
                resp.status_code,
                t0,
                usage,
                cost,
                "".join(parts),
                None,
            )


async def _chain(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    if first:
        yield first
    async for chunk in rest:
        yield chunk


def _trace(
    name: str,
    body: dict[str, Any],
    task: str | None,
    route: str,
    tried: list[str],
    status: int,
    t0: float,
    usage: Any,
    cost: float | None,
    output: Any,
    raw: Any,
    trail: list[str] | None = None,
) -> None:
    tracing.write(
        name,
        {
            "type": "chat",
            "requested": body.get("model"),
            "task": task,
            "model": route.partition(":")[2],
            "route": route,
            "tried": tried,
            "trail": trail or [],
            "status": status,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "messages": body.get("messages"),
            "params": {k: v for k, v in body.items() if k not in ("messages", "model")},
            "output": output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
            "usage": usage,
            "cost_usd": cost,
            **({"error": raw.get("error")} if isinstance(raw, dict) and raw.get("error") else {}),
        },
    )


def _log(
    model: str,
    task: str | None,
    route: str | None,
    tried: list[str],
    status: int,
    t0: float | None,
    usage: Any,
    guest: str | None = None,
    cost: float | None = None,
    trail: list[str] | None = None,
) -> None:
    activity.log_event(
        "router",
        model,
        {"task": task, "tried": tried, "trail": trail or [], "caller": guest or "owner"},
        ok=route is not None and status < 400,
        duration_ms=(time.perf_counter() - t0) * 1000 if t0 else None,
        result={"route": route, "status": status, "usage": usage, "cost_usd": cost},
    )


def models_list() -> dict[str, Any]:
    """`/v1/models`: the routing policies a client can pick, for tools that list models before calling."""
    ids = ["auto"] + [
        f"auto:{t}"
        for t in ("general", "coding", "math", "agents", "russian", "writing", "long_context", "vision")
    ]
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "whichapiapi"} for i in ids]}


def dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode()


# ---------------------------------------------------------------- embeddings


def embedding_targets(model: str) -> list[str]:
    """Reachable channels that sell exactly this embedding model ("auto" = the cheapest embedding model on any
    reachable channel). Never another model: vectors of different models are not comparable."""
    st = core._store(None)
    usable = core.reachable_channels()
    offers = [o for c in usable for o in st.find_offers(channel=c, capability="llm.embeddings", limit=2000)
              if o.group is None and o.price.is_known]  # fmt: skip
    if model == "auto" and offers:
        model = min(offers, key=lambda o: o.price.input_per_1m or 0).model
    return [f"{o.channel}:{o.model}" for o in offers if o.model == model or o.model.endswith("/" + model)]


async def embed(
    body: dict[str, Any], client: httpx.AsyncClient | None = None, guest: str | None = None
) -> tuple[int, dict[str, str], Any]:
    """`/v1/embeddings` with fallback across channels selling the same model, route health and guest charging."""
    model = str(body.get("model") or "auto")
    route = [c for c in route_health.order(embedding_targets(model)) if _target(c) is not None]
    if not route:
        return 404, {}, {"error": {"message": f"no reachable channel sells embedding model {model!r}"}}
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
    trail: list[str] = []
    last: tuple[int, Any] = (502, {"error": {"message": "all channels failed"}})
    for cand in route:
        url, key, upstream, key_id = _target(cand)  # type: ignore[misc]
        t0 = time.perf_counter()
        try:
            resp = await http.post(f"{url}/embeddings", json={**body, "model": upstream},
                                   headers={"Authorization": f"Bearer {key}"})  # fmt: skip
        except httpx.HTTPError as e:
            route_health.record(cand, 0, type(e).__name__)
            trail.append(f"{cand}=timeout:transient/{round((time.perf_counter() - t0) * 1000)}ms")
            continue
        ms = (time.perf_counter() - t0) * 1000
        kind = route_health.record(cand, resp.status_code, resp.text[:300], resp.headers, latency_ms=ms)
        if key_id:
            keypool.report(key_id, resp.status_code, resp.text[:100])
        trail.append(f"{cand}={resp.status_code}{'' if kind == 'ok' else ':' + kind}/{round(ms)}ms")
        data = _json_or_none(resp)
        if resp.status_code < 400:
            cost = _charge(guest, cand, data.get("usage") if isinstance(data, dict) else None)
            meta = {"x-whichapiapi-route": cand, "x-whichapiapi-trail": ";".join(trail)}
            if cost is not None:
                meta["x-whichapiapi-cost-usd"] = f"{cost:.6f}"
            _log(
                "embeddings",
                None,
                cand,
                [c for c in route[: len(trail)]],
                resp.status_code,
                t0,
                None,
                guest,
                cost,
                trail,
            )
            return resp.status_code, meta, data
        if kind == "client":
            return resp.status_code, {"x-whichapiapi-trail": ";".join(trail)}, data
        last = (resp.status_code, data)
    return last[0], {"x-whichapiapi-trail": ";".join(trail)}, last[1]


# ---------------------------------------------------------------- fusion


FUSION_PROMPT = (
    "Several assistants answered the same request. Write the single best answer to the request: keep what is "
    "correct and useful in any of them, fix mistakes, drop repetition. Answer directly, don't mention the "
    "assistants.\n\n# Request\n{request}\n\n{answers}"
)


async def fuse(
    body: dict[str, Any], task: str, client: httpx.AsyncClient | None = None
) -> tuple[int, dict[str, str], Any]:
    """`auto:<task>@fusion` (owner only, no streaming/tools): the top 3 candidates answer in parallel, then the
    best-ranked one that answered writes the final answer from all of them. ~4× the cost of one call."""
    import asyncio

    cands = candidates(task, "optimal")[:3]
    single = [_complete({**body, "model": c.partition(":")[2]}, client, prefer=None, only=c) for c in cands]
    results = await asyncio.gather(*single)
    answers = []
    for c, (status, _, data) in zip(cands, results, strict=True):
        if status < 400 and isinstance(data, dict) and data.get("choices"):
            text = (data["choices"][0].get("message") or {}).get("content") or ""
            if text.strip():
                answers.append((c, text))
    if not answers:
        return results[0]
    if len(answers) == 1:
        status, headers, data = next(r for r in results if r[0] < 400)
        return status, {**headers, "x-whichapiapi-fusion": "1 answer"}, data
    request = (body.get("messages") or [{}])[-1].get("content")
    joined = "\n\n".join(f"# Answer {i + 1}\n{t}" for i, (_, t) in enumerate(answers))
    synth = {
        "model": "x",
        "messages": [{"role": "user", "content": FUSION_PROMPT.format(request=request, answers=joined)}],
    }
    status, headers, data = await _complete(synth, client, only=answers[0][0])
    return status, {**headers, "x-whichapiapi-fusion": ",".join(c for c, _ in answers)}, data
