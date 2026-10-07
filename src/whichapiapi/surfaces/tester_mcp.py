"""Hosted MCP for tester keys: the project's read-only tools over HTTPS, so friends can try it from Codex / Claude /
Cursor without the code being published. Mounted by `public_api` at `/mcp`; bearer auth with a tester key (or the
owner token). Nothing here spends money except `route_chat`, which goes through the router and is charged to the
key; nothing reads or writes files on the server. Every call is traced for tester keys (tracing.py).
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware, MiddlewareContext

from whichapiapi import activity, keys, tracing
from whichapiapi.surfaces import mcp as full

INSTRUCTIONS = """Which API API (tester access): choose which model/API to use for a task and what it really costs.
- suggest_candidates(task, with_scores=True): today's best models for a task (general, coding, math, agents,
  russian, writing, long_context, vision) with quality, cost per task incl. thinking tokens, speed.
- rank_models(preset, task): the full ranking with presets (best, optimal, value, fast, …).
- find_offers / compare_prices / estimate_cost: every way to buy a model, real vs list price, monthly cost.
- model_benchmarks / benchmark_boards: scores from Artificial Analysis, LMArena, SWE-bench, BFCL, OpenRouter usage.
- audit_n8n(workflow): paste an n8n workflow JSON, get overpay / fallback / risk findings.
- route_chat(messages, model="auto"): ask the router; the model is picked per request (charged to your key).
- my_usage(): your key's limit and spend.
Please call report_feedback whenever something is wrong, confusing or missing — that is what testers are for.
This key is a tester key: tool calls and chats are logged in full for the project owner."""


class KeyVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        owner = os.environ.get("WHICHAPIAPI_MCP_TOKEN")
        if owner and token == owner:
            return AccessToken(token=token, client_id="owner", scopes=["owner"])
        found = keys.find(token)
        if found and found[1].get("role", "guest") == "tester":
            return AccessToken(token=token, client_id=found[0], scopes=["tester"])
        return None


def _caller() -> str:
    tok = get_access_token()
    return tok.client_id if tok else "anonymous"


class _Trace(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        name, t0 = _caller(), time.perf_counter()
        args = context.message.arguments
        with activity.track("tester-mcp", context.message.name, args, client=name) as t:
            try:
                result = await call_next(context)
            except Exception as e:
                if name != "owner":
                    tracing.write(name, {"type": "tool", "tool": context.message.name, "arguments": args,
                                         "error": f"{type(e).__name__}: {e}"})  # fmt: skip
                raise
            digest = getattr(result, "structured_content", None) or str(getattr(result, "content", ""))
            t["result"] = str(digest)[:500]
            if name != "owner":
                tracing.write(
                    name,
                    {
                        "type": "tool",
                        "tool": context.message.name,
                        "arguments": args,
                        "result": digest,
                        "latency_ms": round((time.perf_counter() - t0) * 1000),
                    },
                )
            return result


def build(base_url: str = "https://whichapiapi.sluice.stream") -> FastMCP:
    server = FastMCP("whichapiapi", instructions=INSTRUCTIONS, auth=KeyVerifier(base_url=base_url))
    server.add_middleware(_Trace())
    for tool in (
        full.suggest_candidates,
        full.rank_models,
        full.find_offers,
        full.compare_prices,
        full.estimate_cost,
        full.model_benchmarks,
        full.benchmark_boards,
    ):
        server.add_tool(tool)

    @server.tool
    def audit_n8n(workflow: dict[str, Any] | list[Any], calls_per_month: int | None = None) -> dict[str, Any]:
        """Audit an n8n workflow (paste the exported JSON): for every AI call — overpay (same model cheaper
        elsewhere), cheaper models with an equal-or-better benchmark score, missing fallback, training-data risk,
        deprecated/unknown models, keys hardcoded in nodes. Read-only."""
        return full.audit_n8n(workflow=workflow, calls_per_month=calls_per_month)

    @server.tool
    async def route_chat(
        messages: list[dict[str, Any]], model: str = "auto", max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Send a chat to the router: `model` = "auto" | "auto:<task>" | "auto:<task>@<preset>" (optimal, best,
        value, fast). Returns the answer, which model answered, what was tried and the cost (charged to your key)."""
        from whichapiapi.surfaces import router

        name = _caller()
        guest = None if name == "owner" else name
        if guest:
            rec = (keys.list_keys().get(guest)) or {}
            if keys.remaining(rec) <= 0:
                return {"error": f"key limit of ${rec.get('limit_usd', 0):.2f} reached"}
        body: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens:
            body["max_tokens"] = max_tokens
        status, headers, data = await router.complete(body, guest=guest, trace=bool(guest))
        msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
        return {
            "status": status,
            "answer": msg.get("content"),
            "route": headers.get("x-whichapiapi-route"),
            "task": headers.get("x-whichapiapi-task"),
            "tried": headers.get("x-whichapiapi-tried"),
            "cost_usd": float(headers["x-whichapiapi-cost-usd"])
            if "x-whichapiapi-cost-usd" in headers
            else None,
            **({"error": data.get("error")} if isinstance(data, dict) and data.get("error") else {}),
        }

    @server.tool
    def my_usage() -> dict[str, Any]:
        """Your key's limit, spend and remaining budget."""
        name = _caller()
        rec = keys.list_keys().get(name) or {}
        return {k: rec.get(k) for k in ("limit_usd", "spent_usd", "calls", "last_used")} | {
            "key": name,
            "remaining_usd": round(keys.remaining(rec), 6) if rec else None,
        }

    @server.tool
    def report_feedback(
        summary: str,
        what_happened: str,
        category: str = "bug",
        severity: str = "medium",
        tool: str = "",
        evidence: str = "",
        suggested_fix: str = "",
    ) -> dict[str, str]:
        """Tell the maintainer about a bug, wrong data, friction, a missing feature, or what you liked.
        category: bug | wrong-data | friction | missing-feature | praise; severity: high | medium | low."""
        from whichapiapi.feedback import add_entry

        return add_entry(
            summary,
            what_happened,
            category,
            severity,
            tool,
            evidence,
            suggested_fix,
            project=f"tester:{_caller()}",
        )

    return server
