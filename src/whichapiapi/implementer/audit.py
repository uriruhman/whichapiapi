"""Audit an automation (Phase 4): each paid AI call in an n8n workflow against the offer catalog and benchmarks.

Findings: the same model is sold cheaper elsewhere (overpay), a model with an equal-or-better benchmark score costs
less (worth an eval), no fallback when the call fails, the offer may train on your data, the model is deprecated or
unknown to every catalog, the model is only chosen at runtime, an API key is written into the workflow.
Read-only: nothing is changed or called. Money figures use a load profile we state in the report (runs per month
from the schedule trigger; tokens from the visible prompt, else defaults) — loops inside a run multiply them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from whichapiapi import core
from whichapiapi.cost.engine import monthly_cost
from whichapiapi.implementer.n8n import ApiUse, Extraction, extract
from whichapiapi.schema.load_profile import LoadProfile
from whichapiapi.schema.naming import model_key
from whichapiapi.store.db import Store

DEFAULT_CALLS = 1000  # per month, when a workflow has no schedule (webhook, chat, manual)
DEFAULT_IN, DEFAULT_OUT = 2000, 500
OVERPAY_SHARE = 0.2  # cheaper by at least 20 % ...
OVERPAY_USD = 0.05  # ... and 5 cents a month
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


class Finding(BaseModel):
    severity: str  # high | medium | low | info
    kind: str
    workflow: str
    node: str
    message: str
    saving_monthly: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _profile(use: ApiUse, calls: int | None) -> tuple[LoadProfile, str]:
    runs = calls or (round(use.runs_per_month) if use.runs_per_month else DEFAULT_CALLS)
    basis = (
        f"{calls} calls/month (given)"
        if calls
        else f"{runs} runs/month from the schedule"
        if use.runs_per_month
        else f"{DEFAULT_CALLS} calls/month (assumed: no schedule)"
    )
    tokens_in = max(300, int(use.prompt_chars / 3.5)) if use.prompt_chars else DEFAULT_IN
    return LoadProfile(
        requests_per_month=max(runs, 1), input_tokens=tokens_in, output_tokens=DEFAULT_OUT
    ), basis


def _monthly(view: dict[str, Any]) -> float | None:
    return view.get("monthly_real_estimate", view.get("monthly_list"))


def _better_value(
    use: ApiUse, current_monthly: float, profile: LoadProfile, st: Store
) -> list[dict[str, Any]]:
    """Up to 3 cheaper models on the same channel with an equal-or-better LMArena score."""
    here = core.benchmark_score(st, use.model or "")
    if use.capability != "llm.chat" or not here:
        return []
    index = core._bench_index(st)
    seen, out = set(), []
    for o in st.find_offers(channel=use.provider, capability="llm.chat", limit=5000):
        key = model_key(o.model)
        if o.group or key in seen or key == model_key(use.model or "") or not o.price.is_known:
            continue
        seen.add(key)
        score = core._resolve_bench(index, o.model)[1].get(core.DEFAULT_BENCHMARK)
        cost = monthly_cost(o, profile)
        if score and score["score"] >= here["score"] and cost and cost.per_month < current_monthly * 0.8:
            out.append(
                {"offer_id": o.id, "monthly": round(cost.per_month, 4), "lmarena_overall": score["score"]}
            )
    return sorted(out, key=lambda x: x["monthly"])[:3]


def _audit_use(use: ApiUse, calls: int | None, st: Store) -> tuple[list[Finding], dict[str, Any]]:
    f: list[Finding] = []
    row: dict[str, Any] = {
        "workflow": use.workflow,
        "node": use.node,
        "capability": use.capability,
        "provider": use.provider,
        "model": use.model,
        "runs_per_month": use.runs_per_month,
    }

    def add(severity: str, kind: str, message: str, **kw: Any) -> None:
        f.append(
            Finding(severity=severity, kind=kind, workflow=use.workflow, node=use.node, message=message, **kw)
        )

    if use.local:
        add(
            "info",
            "self_hosted",
            f"{use.provider} runs locally: no per-call price (hardware and upkeep instead)",
        )
        return f, row
    if not use.has_fallback:
        fix = (
            "retry with backoff (SDK max_retries) and a fallback model from another vendor"
            if use.node_type == "code"
            else "'Retry On Fail' on the node, or 'Enable Fallback Model' on the chain with a second model from "
            "another vendor"
            if use.node_type.startswith(("@n8n", "n8n-"))
            else "the module's error handler with a retry, and a fallback route to another vendor"
        )
        add(
            "medium",
            "no_fallback",
            f"no retry and no fallback: one provider outage or rate limit stops it ({fix})",
        )
    if use.model is None:
        if use.model_is_expression:
            add(
                "low",
                "runtime_model",
                "model is chosen at runtime (expression): audit the values it can take",
            )
        else:
            add("low", "no_model", "no model found in the node (unfinished node, or set elsewhere)")
        return f, row
    if use.model_is_expression:
        add("info", "runtime_model", f"model is chosen at runtime; audited with its default {use.model}")

    profile, basis = _profile(use, calls)
    row["load"] = basis
    cmp = core.compare_prices(use.model, profile, include_free=True, store=st)
    views = [v for v in cmp["offers"] if _monthly(v) is not None]
    if not views:
        add(
            "medium",
            "unknown_model",
            f"{use.model} is not in any catalog: renamed, retired, or a private name?",
        )
        return f, row
    current = next((v for v in views if v["channel"] == use.provider), None)
    cheapest = views[0]
    row["cheapest"] = {"offer_id": cheapest["offer_id"], "monthly": _monthly(cheapest)}
    if current is None:
        add("info", "channel_not_in_catalog", f"{use.provider} is not a catalog channel for {use.model}")
        return f, row
    now = _monthly(current) or 0.0
    row["monthly_now"] = now
    if note := current.get("source_note"):
        add("medium", "deprecated", f"{use.model} on {use.provider}: {note}")
    if use.model.endswith(":free") or current.get("risk"):
        add(
            "medium",
            "training_data",
            "free/BYO route: prompts may be logged or used for training by the upstream",
        )
    saving = now - (_monthly(cheapest) or now)
    if saving >= OVERPAY_USD and saving >= OVERPAY_SHARE * now:
        add(
            "medium",
            "overpay",
            f"same model {round(100 * saving / now)} % cheaper via {cheapest['offer_id']} "
            f"[{cheapest.get('channel_type')}{', stale data' if cheapest.get('stale') else ''}] "
            f"(${_monthly(cheapest):.2f} vs ${now:.2f} a month at {basis})",
            saving_monthly=round(saving, 4),
            details={"cheapest": cheapest["offer_id"], "source": cheapest.get("source")},
        )
    if alternatives := _better_value(use, now, profile, st):
        add(
            "low",
            "better_value",
            f"{len(alternatives)} cheaper model(s) on {use.provider} with an equal-or-better LMArena score — "
            "worth a run_eval on this node's real inputs before switching",
            saving_monthly=round(now - alternatives[0]["monthly"], 4),
            details={"alternatives": alternatives},
        )
    return f, row


def audit_n8n(data: Any, calls_per_month: int | None = None, store: Store | None = None) -> dict[str, Any]:
    """Audit n8n workflow JSON (one workflow, a list, or an export). Read-only."""
    return audit_extraction(extract(data), calls_per_month, store)


def audit_extraction(
    ex: Extraction, calls_per_month: int | None = None, store: Store | None = None
) -> dict[str, Any]:
    """Audit already-extracted AI calls (n8n, Make blueprint or source code). Read-only."""
    st = core._store(store)
    findings: list[Finding] = []
    rows = []
    for use in ex.uses:
        fs, row = _audit_use(use, calls_per_month, st)
        findings += fs
        rows.append(row)
    findings += [
        Finding(
            severity="high",
            kind="hardcoded_secret",
            workflow=s.workflow,
            node=s.node,
            message=f"API key written into the node ({s.field}: {s.preview}) — move it to n8n credentials and rotate it",
        )
        for s in ex.secrets
    ]
    findings.sort(key=lambda x: (SEVERITY_ORDER.get(x.severity, 9), -(x.saving_monthly or 0.0)))
    by_kind: dict[str, int] = {}
    for x in findings:
        by_kind[x.kind] = by_kind.get(x.kind, 0) + 1
    return {
        "workflows": ex.workflows,
        "api_calls": len(ex.uses),
        "findings_by_kind": by_kind,
        "potential_saving_monthly": round(
            sum(x.saving_monthly or 0 for x in findings if x.kind == "overpay"), 4
        ),
        "findings": [x.model_dump() for x in findings],
        "calls": rows,
        "note": "Read-only audit. Monthly figures assume the load in each call's `load`; loops inside a run multiply "
        "them. Switching models changes quality: validate with run_eval on the node's real inputs first.",
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Automation audit",
        "",
        f"{report['workflows']} workflows · {report['api_calls']} AI calls · findings: "
        + ", ".join(f"{k} {v}" for k, v in sorted(report["findings_by_kind"].items()))
        + f" · potential saving ${report['potential_saving_monthly']:.2f}/month (same models, cheaper channels)",
        "",
        "| severity | workflow | node | finding |",
        "|---|---|---|---|",
    ]
    for x in report["findings"]:
        if x["severity"] == "info":
            continue
        msg = x["message"].replace("|", "\\|")
        lines.append(f"| {x['severity']} | {x['workflow']} | {x['node']} | {msg} |")
    lines += ["", report["note"], ""]
    return "\n".join(lines)
