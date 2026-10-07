"""Public, OpenAI-compatible endpoint for invited people (friends testing the router).

Exposed on its own port behind nginx/HTTPS — the router, plus the read-only MCP tools for tester keys at `/mcp`
(surfaces/tester_mcp.py); never the owner's full MCP or REST catalog:
`GET /` docs page, `GET /models`, `POST /chat/completions`, each also under `/v1/…` so both
`base_url="https://host"` and `base_url="https://host/v1"` work. Auth: `Authorization: Bearer <key>` with a key from
`whichapiapi keys add` (USD limit, spend charged per call) or the owner's WHICHAPIAPI_MCP_TOKEN (no limit).

Guards for invited keys: `auto` policies only, answers capped at 4096 tokens, a per-key rate limit, HTTP 402 at the
key's limit, and HTTP 503 when the shared custom-channel balance falls under WHICHAPIAPI_GUEST_MIN_BALANCE (default $0.40),
so friends can never drain the balance the owner's live services run on.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from whichapiapi import channels, keys
from whichapiapi.evaluator.balance import BalanceProbe
from whichapiapi.surfaces import router, wire

RATE_PER_MIN = int(os.environ.get("WHICHAPIAPI_GUEST_RATE_PER_MIN", "20"))
_hits: dict[str, deque[float]] = defaultdict(deque)
_balance: dict[str, Any] = {"at": 0.0, "usd": None}


def _err(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": code, "code": code}}, status_code=status)


def _who(request: Request) -> tuple[str | None, dict[str, Any] | None, JSONResponse | None]:
    """("owner", None, None) | (guest name, record, None) | (None, None, error response)."""
    got = (
        request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        or request.headers.get("x-api-key", "").strip()
    )  # Anthropic SDKs send x-api-key
    owner = os.environ.get("WHICHAPIAPI_MCP_TOKEN")
    if owner and got and hmac.compare_digest(got, owner):
        return "owner", None, None
    found = keys.find(got) if got else None
    if not found:
        return (
            None,
            None,
            _err(401, "invalid or missing API key (Authorization: Bearer wa-…)", "invalid_api_key"),
        )
    return found[0], found[1], None


def shared_balance_usd() -> float | None:
    """Remaining USD on the primary custom (new-api) channel, cached for a minute (None when unknown)."""
    if time.time() - _balance["at"] > 60:
        ch = channels.primary()
        key = os.environ.get(channels.key_env(ch) or "") if ch else None
        _balance["usd"] = (
            BalanceProbe("newapi", channels.root_url(ch) or "", key).remaining_usd() if ch and key else None
        )
        _balance["at"] = time.time()
    return _balance["usd"]


def _guest_denied(name: str, rec: dict[str, Any]) -> JSONResponse | None:
    if keys.remaining(rec) <= 0:
        return _err(
            402, f"key '{name}' reached its limit of ${rec.get('limit_usd', 0):.2f}", "insufficient_quota"
        )
    q, now = _hits[name], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        return _err(429, f"rate limit: {RATE_PER_MIN} requests per minute", "rate_limit_exceeded")
    q.append(now)
    floor = float(os.environ.get("WHICHAPIAPI_GUEST_MIN_BALANCE", "0.40"))
    bal = shared_balance_usd()
    if bal is None or bal < floor:
        return _err(503, "the shared balance is paused for guests right now, try later", "balance_protected")
    return None


async def chat(request: Request) -> Response:
    who, rec, denied = _who(request)
    if denied:
        return denied
    guest = None if who == "owner" else who
    if guest and (denied := _guest_denied(guest, rec or {})):
        return denied
    raw = await request.body()
    if len(raw) > 2 * 1024 * 1024:
        return _err(413, "request too large (2 MiB max)", "request_too_large")
    try:
        body = json.loads(raw)
    except ValueError:
        return _err(400, "invalid JSON", "invalid_request_error")
    if not isinstance(body, dict):
        return _err(400, "the body must be a JSON object", "invalid_request_error")
    tester = bool(guest) and (rec or {}).get("role", "guest") == "tester"
    status, headers, data = await router.complete(
        body, guest=guest, trace=tester, request_headers=request.headers
    )
    if hasattr(data, "__aiter__"):  # X-Accel-Buffering: nginx must pass SSE chunks through as they come
        headers = {**headers, "X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        return StreamingResponse(data, status_code=status, headers=headers, media_type="text/event-stream")
    return JSONResponse(data, status_code=status, headers=headers)


async def _read_json(request: Request) -> dict[str, Any] | JSONResponse:
    raw = await request.body()
    if len(raw) > 8 * 1024 * 1024:
        return _err(413, "request too large (8 MiB max)", "request_too_large")
    try:
        body = json.loads(raw)
    except ValueError:
        return _err(400, "invalid JSON", "invalid_request_error")
    return (
        body
        if isinstance(body, dict)
        else _err(400, "the body must be a JSON object", "invalid_request_error")
    )


def _wire_endpoint(kind: str):
    async def endpoint(request: Request) -> Response:
        who, rec, denied = _who(request)
        if denied:
            return denied
        guest = None if who == "owner" else who
        if guest and (denied := _guest_denied(guest, rec or {})):
            return denied
        body = await _read_json(request)
        if isinstance(body, JSONResponse):
            return body
        tester = bool(guest) and (rec or {}).get("role", "guest") == "tester"
        status, headers, data = await wire.handle(
            kind, body, guest=guest, trace=tester, request_headers=request.headers
        )
        if hasattr(data, "__aiter__"):
            headers = {**headers, "X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
            return StreamingResponse(
                data, status_code=status, headers=headers, media_type="text/event-stream"
            )
        return JSONResponse(data, status_code=status, headers=headers)

    return endpoint


async def embeddings(request: Request) -> Response:
    who, rec, denied = _who(request)
    if denied:
        return denied
    guest = None if who == "owner" else who
    if guest and (denied := _guest_denied(guest, rec or {})):
        return denied
    body = await _read_json(request)
    if isinstance(body, JSONResponse):
        return body
    status, headers, data = await router.embed(body, guest=guest)
    return JSONResponse(data, status_code=status, headers=headers)


async def count_tokens(request: Request) -> Response:
    denied = _who(request)[2]
    if denied:
        return denied
    body = await _read_json(request)
    if isinstance(body, JSONResponse):
        return body
    return JSONResponse({"input_tokens": wire.count_tokens_estimate(body)})


async def models(request: Request) -> Response:
    denied = _who(request)[2]
    return denied or JSONResponse(router.models_list())


async def usage(request: Request) -> Response:
    """GET /usage — the caller's own limit and spend."""
    who, rec, denied = _who(request)
    if denied:
        return denied
    if who == "owner":
        return JSONResponse({"key": "owner", "limit_usd": None})
    rec = rec or {}
    return JSONResponse(
        {
            "key": who,
            "limit_usd": rec.get("limit_usd"),
            "spent_usd": round(rec.get("spent_usd", 0.0), 6),
            "remaining_usd": round(keys.remaining(rec), 6),
            "calls": rec.get("calls", 0),
        }
    )


_LOGIN = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Вход</title><body style="font:16px system-ui;max-width:360px;margin:15vh auto;padding:0 16px">
<form method=post><p>Токен владельца</p><input name=token type=password autofocus style="width:100%;font-size:16px">
<p><button>Войти</button></form>"""


async def dashboard(request: Request) -> Response:
    """GET /dashboard — the owner's dashboard. Login once with the owner token (form → HttpOnly cookie), never in the
    URL, so it doesn't land in proxy logs or browser history."""
    owner = os.environ.get("WHICHAPIAPI_MCP_TOKEN")
    if not owner:
        return _err(404, "not found", "not_found")
    if request.method == "POST":
        form = await request.form()
        if hmac.compare_digest(str(form.get("token") or ""), owner):
            resp = Response(status_code=303, headers={"Location": "/dashboard"})
            resp.set_cookie("waa_owner", _owner_cookie(owner), max_age=30 * 86400, httponly=True, secure=True,
                            samesite="strict")  # fmt: skip
            return resp
        return HTMLResponse(_LOGIN, status_code=401)
    if not hmac.compare_digest(request.cookies.get("waa_owner", ""), _owner_cookie(owner)):
        return HTMLResponse(_LOGIN)
    import asyncio

    from whichapiapi.surfaces.cli import render_dashboard

    days = float(request.query_params.get("days") or 7)
    html = await asyncio.to_thread(render_dashboard, min(max(days, 1), 30))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


def _owner_cookie(owner: str) -> str:
    return hmac.new(owner.encode(), b"whichapiapi-dashboard", "sha256").hexdigest()


async def docs(request: Request) -> Response:
    host = request.headers.get("host", "whichapiapi.sluice.stream")
    page = (Path(__file__).parent / "api_docs.html").read_text(encoding="utf-8")
    return HTMLResponse(page.replace("__HOST__", host.replace("<", "").replace('"', "")))


PUBLIC_HOST = os.environ.get("WHICHAPIAPI_PUBLIC_HOST", "whichapiapi.sluice.stream")


def app() -> Starlette:
    from starlette.routing import Mount

    from whichapiapi.surfaces.tester_mcp import build

    mcp_app = build(f"https://{PUBLIC_HOST}").http_app(
        path="/mcp", allowed_hosts=[PUBLIC_HOST, "127.0.0.1:*", "localhost:*", "testserver"]
    )
    routes = [
        Route("/", docs, methods=["GET"]),
        Route("/v1", docs, methods=["GET"]),
        Route("/dashboard", dashboard, methods=["GET", "POST"]),
    ]
    for prefix in ("", "/v1"):
        routes += [
            Route(f"{prefix}/chat/completions", chat, methods=["POST"]),
            Route(f"{prefix}/models", models, methods=["GET"]),
            Route(f"{prefix}/usage", usage, methods=["GET"]),
            Route(f"{prefix}/messages", _wire_endpoint("anthropic"), methods=["POST"]),
            Route(f"{prefix}/messages/count_tokens", count_tokens, methods=["POST"]),
            Route(f"{prefix}/responses", _wire_endpoint("responses"), methods=["POST"]),
            Route(f"{prefix}/embeddings", embeddings, methods=["POST"]),
        ]
    routes.append(Mount("/", app=mcp_app))
    return Starlette(routes=routes, lifespan=mcp_app.lifespan)


def serve(host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    uvicorn.run(app(), host=host, port=port, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
