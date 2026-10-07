"""MCP server (ADR-0006). Tools map 1:1 to `whichapiapi.core`; no logic lives here.

Run:  whichapiapi mcp            (stdio, for Claude Code / Cursor / any MCP client)
      whichapiapi mcp --http     (streamable HTTP on 127.0.0.1:8765/mcp; also serves the
                                  changedetection.io webhook at /webhooks/changedetection)
"""

from __future__ import annotations

import contextlib
import functools
import hmac
import ipaddress
import json
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.middleware import Middleware, MiddlewareContext
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from whichapiapi import activity, core
from whichapiapi.schema.load_profile import LoadProfile
from whichapiapi.selector.constraints import Constraints

INSTRUCTIONS = """Which API API helps choose which API/model to use for a task and what it really costs.
- find_offers / compare_prices: every way to buy a model (official, aggregators, the user's own reseller channels),
  with list prices, real-price multipliers learned from the user's own call logs, staleness and risk notes.
  find_offers(sort="value", benchmark=...) ranks by benchmark score per unit of price (LMArena arenas and
  categories, CC BY 4.0; see benchmark_boards / model_benchmarks) once pull_benchmarks has been run; offers
  without a matched score are ranked last, not treated as bad.
  Offers may also carry `batch_discount`/`off_peak_discount` derived from LiteLLM's pricing data (MIT) once
  pull_conditions has been run.
- estimate_cost: monthly cost for a load profile (requests/month, tokens per request, cache/batch shares).
  Applies real batch/off-peak discounts when `pull_conditions` has populated them for that model.
- cross_check_price: one live lookup against InferIndex (the closest competitor) for price sanity and promo
  detection — informational only, never a source of truth, use sparingly (their API is rate-limited).
- create_suite: turn a task description + example inputs into a runnable benchmark (you draft prompt, cases,
  criteria and candidates; it writes the files). Then eval_plan / run_eval.
- suite_from_traces: the same from production traces (Langfuse, Arize Phoenix / OpenTelemetry spans, JSON lines):
  real inputs as cases, production answers as the judge's reference. refresh_suite: which new models a saved suite
  should add now (today's best for its task); cached results keep re-runs cheap.
- eval_plan / run_eval: test candidates on the USER'S OWN examples (a promptfoo-style suite). run_eval spends
  real money within a hard budget — always show eval_plan to the user and get consent before run_eval.
- run_canary: a small (5-case) fixed suite that repeats over time to catch silent model substitution or
  degradation on a channel, diffed against a stored baseline. Also spends real money (small) — ask first.
- audit_n8n: read-only audit of an n8n workflow's AI calls (overpay, missing fallback, training-data risk, keys
  in nodes). Never spends money; suggest run_eval before any model switch it proposes.
- price_change_alerts: pricing-page changes recorded by the optional changedetection.io webhook
  (/webhooks/changedetection in --http mode) — self-hosted, never spends money, needs no consent to read.
Never present a price as certain: cite `source` and `verified_at`, and flag `stale` or `risk` fields.

Feedback (please do this without being asked): when a tool errors, returns something wrong, contradictory or stale,
is confusing to use, or lacks something you needed, call report_feedback once per root cause - tool, arguments,
what came back, what you expected. Never include API keys or personal data. It goes to the maintainers' inbox and
is how this server gets fixed; it costs nothing and needs no user consent."""


class _ActivityLog(Middleware):
    """Every tool call goes to the activity log (whichapiapi.activity): tool, arguments, calling project, client,
    duration, a result digest or the error."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        client = None
        with contextlib.suppress(Exception):
            info = context.fastmcp_context.session.client_params.clientInfo  # type: ignore[union-attr]
            client = f"{info.name} {info.version}"
        with activity.track("mcp", context.message.name, context.message.arguments, client=client) as t:
            result = await call_next(context)
            t["result"] = (
                getattr(result, "structured_content", None) or str(getattr(result, "content", ""))[:500]
            )
            return result


mcp = FastMCP("whichapiapi", instructions=INSTRUCTIONS)
mcp.add_middleware(_ActivityLog())


WEBHOOK_MAX_BYTES = 64 * 1024


def _suite(path: str) -> str:
    """Suites run code (python assertions) and send their prompts/files to APIs with the user's keys, so an MCP
    client may only point at suites under trusted roots: WHICHAPIAPI_SUITE_ROOTS (os.pathsep-separated), default the
    server's working directory."""
    roots = [
        Path(r).resolve()
        for r in os.environ.get("WHICHAPIAPI_SUITE_ROOTS", os.getcwd()).split(os.pathsep)
        if r
    ]
    p = Path(path).resolve()
    if not any(p.is_relative_to(r) for r in roots):
        raise ValueError(
            f"suite {path} is outside the allowed roots {[str(r) for r in roots]} (WHICHAPIAPI_SUITE_ROOTS)"
        )
    return str(p)


def _profile(p: dict[str, Any] | None) -> LoadProfile | None:
    return LoadProfile.model_validate(p) if p else None


def _constraints(c: dict[str, Any] | None) -> Constraints | None:
    return Constraints.model_validate(c) if c else None


@mcp.tool
def find_offers(
    query: str | None = None,
    capability: str | None = None,
    channel: str | None = None,
    constraints: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    sort: str = "input",
    limit: int = 20,
    include_free: bool = False,
    benchmark: str = "lmarena/text/overall",
) -> dict[str, Any]:
    """Search offers (ways to buy a capability/model). `query` is a model name, e.g. "deepseek-v4.1-flash".
    `constraints` keys: channel_types, exclude_user_channels, require_features (tools|json_schema|vision|reasoning),
    min_context, max_input_per_1m, max_output_per_1m, payment_method, no_kyc, zero_retention,
    no_training_on_data, region, min_uptime, strict. `profile` keys: requests_per_month, input_tokens,
    output_tokens, cached_share, batch_share. `sort`: input | output | monthly (needs profile) | value
    (benchmark score per unit of price; `benchmark` picks the board, e.g. "lmarena/text/coding",
    "lmarena/text/russian", "lmarena/webdev/overall" — list them with benchmark_boards; run pull_benchmarks once)."""
    return core.find_offers(
        query,
        capability,
        channel,
        _constraints(constraints),
        _profile(profile),
        sort,
        limit,
        include_free=include_free,
        benchmark=benchmark,
    )


@mcp.tool
def compare_prices(
    model: str,
    profile: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    include_groups: bool = False,
) -> dict[str, Any]:
    """All channels selling one model, cheapest real monthly cost first. include_groups adds reseller price
    groups / per-upstream variants (informational: keys are usually routed automatically)."""
    return core.compare_prices(
        model, _profile(profile), _constraints(constraints), include_groups=include_groups
    )


@mcp.tool
def estimate_cost(offer_id: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Monthly cost of one offer (id like "openrouter/deepseek/deepseek-v4.1-flash") for a load profile.
    Applies batch/off-peak discounts if the offer has them (its own data, or LiteLLM-derived priors from
    pull_conditions) and `profile.batch_share`/an off-peak share are set."""
    return core.estimate_cost(offer_id, _profile(profile))


@mcp.tool
def group_usage(channel: str | None = None) -> list[dict[str, Any]]:
    """Which reseller price group actually answered the user's calls, per model (share of calls, mean cost,
    retried share). Use it to sanity-check a "cheapest group" price: a group with share 0 is not reachable."""
    return core.group_usage(channel)


@mcp.tool
def learned_prices(channel: str | None = None) -> list[dict[str, Any]]:
    """Real/list price multipliers learned from the user's own channel call logs (e.g. a reseller that routes
    calls to cheaper price groups)."""
    return core.learned_prices(channel)


@mcp.tool
def pull_offers(
    sources: list[str] | None = None, newapi: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """Refresh the offer catalog. sources: models-dev, openrouter. newapi: [{"url", "channel", "key_env"}]."""
    return core.pull_offers(sources, newapi)


@mcp.tool
def pull_benchmarks() -> dict[str, Any]:
    """Refresh third-party benchmark scores: every LMArena arena (text, vision, webdev, search, document, agent,
    image/video generation) and category (coding, math, hard_prompts, russian, industries, ...), CC BY 4.0.
    Free, no key; takes about a minute. Powers find_offers(sort="value", benchmark=...) and model_benchmarks."""
    return core.pull_benchmarks()


@mcp.tool
def benchmark_boards() -> list[dict[str, Any]]:
    """Benchmark boards we hold scores for ("lmarena/text/coding", ...) and how many models each covers."""
    return core.benchmark_boards()


@mcp.tool
def model_benchmarks(model: str) -> dict[str, Any]:
    """Every benchmark score we hold for one model (all boards, with rank, confidence interval, votes, date)."""
    return core.model_benchmarks(model)


@mcp.tool
def pull_conditions() -> dict[str, Any]:
    """Refresh per-model batch/off-peak discount conditions from LiteLLM's public pricing JSON (MIT; read
    over HTTP, the `litellm` package is never installed — ADR-0003). Makes find_offers/estimate_cost's
    batch/off-peak discounts real instead of only defined-but-unused fields."""
    return core.pull_conditions()


@mcp.tool
def conditions_priors() -> list[dict[str, Any]]:
    """Cached LiteLLM-derived conditions (batch/off-peak discount fractions) by model."""
    return core.conditions_priors()


@mcp.tool
def cross_check_price(model: str) -> dict[str, Any]:
    """Compare our cheapest known offer for a model against a live InferIndex lookup (price sanity + promo
    detection). One live, rate-limited network call to the closest competitor's public API — informational
    only, not a source of truth."""
    return core.cross_check_price(model)


@mcp.tool
def eval_plan(suite_path: str, only: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    """Estimate calls and cost of running an eval suite, without calling anything."""
    return core.eval_plan(_suite(suite_path), only, limit)


@mcp.tool
async def run_eval(
    suite_path: str,
    budget: float = 0.25,
    only: list[str] | None = None,
    limit: int | None = None,
    judge: str | None = None,
) -> dict[str, Any]:
    """Run an eval suite on real APIs (SPENDS MONEY, capped by `budget` USD). Ask the user first.
    Returns per-model scores, list vs real cost, primary + fallback recommendation and a Markdown report."""
    return await core.arun_eval(_suite(suite_path), budget, only, limit, judge)


@mcp.tool
async def run_canary(
    suite_path: str = "examples/canary/suite.yaml", budget: float = 0.05, reset_baseline: bool = False
) -> dict[str, Any]:
    """Run the small canary suite (SPENDS MONEY, capped by `budget` USD — tiny, but ask the user first) and
    diff it against a stored per-provider baseline: catches a channel silently substituting or degrading the
    model behind a model name. `alerts` is empty on a clean run. A run WITH alerts does not refresh the
    baseline (so a real regression can't silently become normal) — pass `reset_baseline=True` after review."""
    return await core.arun_canary(_suite(suite_path), budget, reset_baseline)


@mcp.tool
def price_change_alerts(limit: int = 40) -> list[dict[str, Any]]:
    """Recent changedetection.io "this pricing page changed" notifications, newest first, with any offers
    whose source matches the changed page's domain. Only populated once the webhook has received something —
    see the self-host recipe in docs/ARCHITECTURE.md."""
    return core.price_change_alerts(limit=limit)


@mcp.custom_route("/webhooks/changedetection", methods=["POST"])
async def changedetection_webhook(request: Request) -> Response:
    """HTTP endpoint (only served in `--http` mode) for changedetection.io's notification webhook. Point a
    watch's notification URL at this to record "the pricing page changed" — never spends money and never
    re-fetches automatically, it just flags which offers/channels to re-pull.

    Custom routes are not covered by the MCP bearer auth, so this checks its own shared secret: set
    WHICHAPIAPI_WEBHOOK_TOKEN and use `.../webhooks/changedetection?token=<it>` as the notification URL."""
    expected = os.environ.get("WHICHAPIAPI_WEBHOOK_TOKEN") or os.environ.get("WHICHAPIAPI_MCP_TOKEN")
    if expected and not hmac.compare_digest(request.query_params.get("token", ""), expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.body()
    if len(body) > WEBHOOK_MAX_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    try:
        payload = json.loads(body)
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "expected a JSON object"}, status_code=400)
    record = core.ingest_price_change_notification(payload)
    return JSONResponse(record)


@mcp.tool
def rank_models(
    preset: str = "optimal",
    task: str | None = None,
    weights: list[float] | None = None,
    profile: str | None = None,
    limit: int = 10,
    min_quality: float | None = None,
    min_tps: float | None = None,
    max_cost: float | None = None,
    vision: bool = False,
    open_only: bool = False,
) -> dict[str, Any]:
    """Best model variants (reasoning effort included) for a task. Presets: best, optimal, value, fast, ultrafast,
    realtime, coding, math, agents, russian, writing, vision, long_context, open. `weights` = [quality, cost, speed];
    `profile` = chat|classify|summarize|rag|agent_step|code|long_doc (tokens per task). Cost per task includes
    estimated thinking tokens, so a model that thinks 10× longer shows 10× the output cost."""
    filters = {
        k: v for k, v in {"min_quality": min_quality, "min_tps": min_tps, "max_cost": max_cost}.items() if v
    }
    if vision:
        filters["vision"] = True
    if open_only:
        filters["open_only"] = True
    w = tuple(weights) if weights and len(weights) == 3 else None
    return core.rank_models(preset, task=task, weights=w, profile=profile, limit=limit, **filters)


@mcp.tool
def suggest_candidates(
    task: str = "general",
    n: int = 4,
    preset: str = "optimal",
    with_scores: bool = False,
    compare: list[str] | None = None,
) -> Any:
    """Best model variants for a task that the user's own keys can call ("channel:model"), one per creator; pairs
    that often fail on the owner's real traffic (field logs) go last. with_scores=True or compare=[ids you use now]
    → {"candidates": [...], "compare": [...]} rows with quality (0-100 for the task), cost_per_task (USD, thinking
    tokens included), seconds_per_task, score (same scale for both lists), why, field (real-traffic reliability)."""
    return core.suggest_candidates(task, n, preset, with_scores=with_scores, compare=compare)


@mcp.tool
def field_stats(days: float = 14) -> dict[str, Any]:
    """Read-only reliability of channel:model pairs on the owner's real traffic (Tributary usage logs): calls, ok
    rate, timeouts, errors, p50 latency, good/bad ratings. No API calls."""
    from whichapiapi.evaluator import field

    return field.stats(days)


@mcp.tool
def canary_status() -> list[dict[str, Any]]:
    """Read-only: the last canary run per suite (time, per-provider pass rate and score vs baseline, alerts about a
    channel silently substituting or degrading a model). Never calls an API or spends; run_canary does."""
    return core.canary_status()


@mcp.tool
def route_health(routes: list[str] | None = None) -> list[dict[str, Any]]:
    """Read-only: what the router remembers per route ("channel:model"): cooldown left and why (rate limit, daily
    quota, payment, auth, gone…), decaying penalty, success rate (2-day half-life), p95 first-token and total latency,
    the median hidden-prompt tokens a channel bills on top of the request, and the last error."""
    from whichapiapi import route_health as rh

    return rh.snapshot(routes)


@mcp.tool
def routing_info(task: str = "general", preset: str = "optimal") -> dict[str, Any]:
    """Read-only: which routes `auto:<task>@<preset>` would try right now and in what order (benchmark/cost
    candidates re-ordered by route health), with each route's health."""
    from whichapiapi import route_health as rh
    from whichapiapi.surfaces import router

    cands = router.candidates(task, preset)
    order = rh.order(cands)
    return {"task": task, "preset": preset, "candidates": cands, "order": order, "health": rh.snapshot(order)}


@mcp.tool
def usage_summary(days: float = 7.0) -> dict[str, Any]:
    """Read-only: router traffic over the last `days` per route and per key (calls, failures, p50 latency, charged
    USD), plus the balance burn forecast of the primary custom channel."""
    from whichapiapi import activity

    return {**activity.router_usage(days), "balance": core.balance_forecast()}


@mcp.tool
def integrity_status() -> list[dict[str, Any]]:
    """Read-only: the last model-integrity verdict per watched channel:model — is the model behind the name the one
    you pay for? Checks: tokenizer (billed token counts vs the claimed family's tokenizer; catches a Chinese model
    behind a Western name), hidden system prompt injected by the channel, self-identification, China-censorship
    behaviour, fingerprint vs its own baseline and its cheaper sibling (Sol vs Luna), response metadata. Status
    ok / warn / alert per check. Prefer models without alerts. Never calls an API; run_integrity does."""
    return core.integrity_status()


@mcp.tool
async def run_integrity(
    watch_path: str | None = None, models: list[str] | None = None, reset_baseline: bool = False
) -> dict[str, Any]:
    """Run the model-integrity probes (SPENDS MONEY: ~20 short calls per model, about $0.002 per model on a cheap reseller —
    ask the user first). `models` limits the run to some model ids of the watch file. Alerts mean the channel likely
    serves a different model (or fakes usage); a run with alerts does not refresh the baseline."""
    default = Path(core.__file__).resolve().parents[2] / "examples/integrity/watch.yaml"
    return await core.arun_integrity(
        _suite(watch_path or str(default)), reset_baseline=reset_baseline, only=models
    )


@mcp.tool
def create_suite(
    directory: str,
    title: str,
    prompt: str,
    cases: list[dict[str, Any]],
    candidates: list[str] | None = None,
    criteria: str = "",
    system: str = "",
    json_output: bool = False,
    contains_pii: bool = False,
    budget_usd: float = 0.10,
    task: str = "general",
) -> dict[str, Any]:
    """Build a benchmark from a task description (the "describe it → benchmark" flow): YOU turn the user's task into
    a prompt template with {{var}} placeholders, 5–20 realistic cases (each {"vars": {...}, "expected"?: "substring"}),
    acceptance criteria for the LLM judge, and candidates ("channel:model", current choice first; pick them with
    rank_models / find_offers) — or leave `candidates` empty to get the 4 best variants for `task` (general, coding,
    math, agents, russian, writing, long_context, vision) that the user's own keys can call. Writes suite.yaml, prompt.json, tests.yaml into `directory` (under the allowed suite
    roots). Then call eval_plan, show the cost, and run_eval only after the user agrees."""
    from whichapiapi.implementer.suite_draft import draft_suite

    target = Path(_suite(str(Path(directory) / "suite.yaml"))).parent
    candidates = candidates or core.suggest_candidates(task)
    return draft_suite(
        target, title, prompt, cases, candidates, criteria, system, json_output,
        budget_usd=budget_usd, contains_pii=contains_pii, task=task,
    )  # fmt: skip


@mcp.tool
def suite_from_traces(
    traces: str,
    directory: str,
    title: str = "Production traffic",
    task: str = "general",
    sample: int = 20,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Build a benchmark from production traces: `traces` is a file (Langfuse observations JSON, OTLP/JSON spans from
    Arize Phoenix / OpenTelemetry GenAI, or JSON lines of {messages|input, output, model}) under the allowed roots.
    Real inputs become cases, production's answers the judge's reference; candidates = the production model (if the
    user's keys can call it) + today's best for `task`, or your own `candidates`
    (channel:model, current first). Traffic is treated as personal data. Then eval_plan →
    run_eval after the user agrees."""
    from whichapiapi.implementer import traces as tr

    calls = tr.parse(Path(_suite(traces)).read_text(encoding="utf-8"))
    target = Path(_suite(str(Path(directory) / "suite.yaml"))).parent
    return tr.build_suite(calls, target, title, task=task, sample=sample, candidates=candidates)


@mcp.tool
def refresh_suite(suite: str, n: int = 4, write: bool = False) -> dict[str, Any]:
    """New models worth adding to a saved benchmark: today's best `n` variants for the suite's task
    (x-whichapiapi.task) that the user's keys can call and the suite doesn't have yet. With write=True they're appended
    to suite.yaml; then eval_plan → run_eval (cached results keep the old candidates free)."""
    from whichapiapi.implementer import refresh

    path = Path(_suite(suite))
    found = refresh.new_candidates(path, n=n)
    if write and found["new"]:
        found["unknown_channels"] = refresh.add_providers(path, found["new"])
        found["written"] = True
    return found


@mcp.tool
def report_feedback(
    summary: str,
    what_happened: str,
    category: str = "bug",
    severity: str = "medium",
    tool: str = "",
    evidence: str = "",
    suggested_fix: str = "",
) -> dict[str, str]:
    """Send the maintainers a bug report, a wrong-data report, friction or a missing feature about THIS server.
    category: bug | wrong-data | friction | missing-feature | praise; severity: high | medium | low. Be concrete:
    which tool, which arguments, what came back, what you expected. No secrets or personal data."""
    from whichapiapi.feedback import add_entry

    return add_entry(summary, what_happened, category, severity, tool, evidence, suggested_fix)


@mcp.tool
def audit_n8n(
    workflow: dict[str, Any] | list[Any] | None = None,
    path: str | None = None,
    calls_per_month: int | None = None,
) -> dict[str, Any]:
    """Audit n8n workflow JSON (pass `workflow` inline, or `path` to an export file under the allowed roots):
    for every AI call — overpay (same model cheaper elsewhere), cheaper models with an equal-or-better benchmark
    score (suggest run_eval before switching), no fallback on failure, training-data risk, deprecated/unknown
    models, runtime-chosen models, and API keys written into nodes. Read-only, never spends money."""
    from whichapiapi.implementer.audit import audit_n8n as _audit

    if workflow is None and path is None:
        raise ValueError("pass `workflow` or `path`")
    data = workflow if workflow is not None else Path(_suite(path)).read_text(encoding="utf-8")
    return _audit(data, calls_per_month=calls_per_month)


# ---------------------------------------------------------------- REST (read-only, same HTTP server)
# Plain JSON endpoints for scripts and n8n HTTP nodes. They never spend money. Custom routes are not covered by the
# MCP bearer auth, so they check WHICHAPIAPI_MCP_TOKEN themselves when it is set.


def _logged(fn: Any) -> Any:
    """Record a REST request in the activity log (path, query, status, duration)."""

    @functools.wraps(fn)
    async def wrapper(request: Request) -> Response:
        with activity.track("rest", request.url.path, dict(request.query_params)) as t:
            resp = await fn(request)
            t["ok"] = resp.status_code < 400
            body = getattr(resp, "body", None)  # streams have none
            t["result"] = {"status": resp.status_code, "bytes": len(body) if body is not None else None}
            return resp

    return wrapper


def _rest_denied(request: Request) -> JSONResponse | None:
    token = os.environ.get("WHICHAPIAPI_MCP_TOKEN")
    got = (
        request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        or request.headers.get("x-api-key", "").strip()
    )
    if token and not hmac.compare_digest(got, token):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _q(request: Request, name: str, default: Any = None, cast: Any = str) -> Any:
    v = request.query_params.get(name)
    try:
        return cast(v) if v not in (None, "") else default
    except ValueError:
        return default


@mcp.custom_route("/api/offers", methods=["GET"])
@_logged
async def rest_offers(request: Request) -> Response:
    """GET /api/offers?query=&capability=&channel=&sort=input|output|value&benchmark=&limit="""
    if denied := _rest_denied(request):
        return denied
    res = core.find_offers(
        _q(request, "query"),
        _q(request, "capability"),
        _q(request, "channel"),
        sort=_q(request, "sort", "input"),
        limit=min(_q(request, "limit", 25, int), 200),
        benchmark=_q(request, "benchmark", "lmarena/text/overall"),
    )
    return JSONResponse(res)


@mcp.custom_route("/api/rank", methods=["GET"])
@_logged
async def rest_rank(request: Request) -> Response:
    """GET /api/rank?preset=optimal&task=&profile=&limit= — best model variants for a task."""
    if denied := _rest_denied(request):
        return denied
    return JSONResponse(
        core.rank_models(
            _q(request, "preset", "optimal"),
            task=_q(request, "task"),
            profile=_q(request, "profile"),
            limit=min(_q(request, "limit", 10, int), 100),
        )
    )


@mcp.custom_route("/api/models/{model:path}/benchmarks", methods=["GET"])
@_logged
async def rest_benchmarks(request: Request) -> Response:
    if denied := _rest_denied(request):
        return denied
    return JSONResponse(core.model_benchmarks(request.path_params["model"]))


@mcp.custom_route("/api/audit/n8n", methods=["POST"])
@_logged
async def rest_audit_n8n(request: Request) -> Response:
    """POST a workflow JSON (or an export) → audit report. Read-only; body limit 5 MB."""
    if denied := _rest_denied(request):
        return denied
    body = await request.body()
    if len(body) > 5 * 1024 * 1024:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    from whichapiapi.implementer.audit import audit_n8n as _audit

    try:
        return JSONResponse(_audit(body.decode("utf-8"), calls_per_month=_q(request, "calls", None, int)))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------- per-request router (OpenAI-compatible)


@mcp.custom_route("/v1/chat/completions", methods=["POST"])
@_logged
async def router_chat(request: Request) -> Response:
    """model: "auto" | "auto:<task>" | "auto:<task>@<preset>" | a concrete model. See surfaces/router.py."""
    if denied := _rest_denied(request):
        return denied
    from starlette.responses import StreamingResponse

    from whichapiapi.surfaces import router

    try:
        body = json.loads(await request.body())
    except ValueError:
        return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)
    status, headers, data = await router.complete(body, request_headers=request.headers)
    if hasattr(data, "__aiter__"):
        return StreamingResponse(data, status_code=status, headers=headers, media_type="text/event-stream")
    return JSONResponse(data, status_code=status, headers=headers)


async def _wire_route(request: Request, kind: str) -> Response:
    if denied := _rest_denied(request):
        return denied
    from starlette.responses import StreamingResponse

    from whichapiapi.surfaces import wire

    try:
        body = json.loads(await request.body())
    except ValueError:
        return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)
    status, headers, data = await wire.handle(kind, body, request_headers=request.headers)
    if hasattr(data, "__aiter__"):
        return StreamingResponse(data, status_code=status, headers=headers, media_type="text/event-stream")
    return JSONResponse(data, status_code=status, headers=headers)


@mcp.custom_route("/v1/messages", methods=["POST"])
@_logged
async def router_messages(request: Request) -> Response:
    """Anthropic Messages wire protocol on the router (Claude Code: ANTHROPIC_BASE_URL)."""
    return await _wire_route(request, "anthropic")


@mcp.custom_route("/v1/messages/count_tokens", methods=["POST"])
async def router_count_tokens(request: Request) -> Response:
    if denied := _rest_denied(request):
        return denied
    from whichapiapi.surfaces import wire

    return JSONResponse({"input_tokens": wire.count_tokens_estimate(json.loads(await request.body()))})


@mcp.custom_route("/v1/responses", methods=["POST"])
@_logged
async def router_responses(request: Request) -> Response:
    """OpenAI Responses wire protocol on the router (Codex: wire_api = "responses")."""
    return await _wire_route(request, "responses")


@mcp.custom_route("/v1/embeddings", methods=["POST"])
@_logged
async def router_embeddings(request: Request) -> Response:
    """Embeddings with fallback across channels selling the same model (`model`: an id, or "auto" = cheapest)."""
    if denied := _rest_denied(request):
        return denied
    from whichapiapi.surfaces import router

    status, headers, data = await router.embed(json.loads(await request.body()))
    return JSONResponse(data, status_code=status, headers=headers)


@mcp.custom_route("/v1/models", methods=["GET"])
async def router_models(request: Request) -> Response:
    if denied := _rest_denied(request):
        return denied
    from whichapiapi.surfaces import router

    return JSONResponse(router.models_list())


@mcp.resource("whichapiapi://taxonomy")
def taxonomy() -> list[dict[str, Any]]:
    """Capability taxonomy (llm.chat, search.web, speech.stt, …) with pricing units and default scorers."""
    return core.taxonomy()


def _loopback(host: str) -> bool:
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(http: bool = False, host: str = "127.0.0.1", port: int = 8765) -> None:
    if http:
        # HTTP exposes spend-capable tools: require a bearer token (WHICHAPIAPI_MCP_TOKEN) unless loopback-only.
        if token := os.environ.get("WHICHAPIAPI_MCP_TOKEN"):
            mcp.auth = StaticTokenVerifier(tokens={token: {"client_id": "owner", "scopes": []}})
        elif not _loopback(host):
            raise SystemExit(
                f"refusing to serve on {host} without WHICHAPIAPI_MCP_TOKEN (anyone could spend your keys)"
            )
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run(show_banner=False)
