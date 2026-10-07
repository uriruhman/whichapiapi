"""Run report → terminal table, Markdown, JSON."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from whichapiapi.evaluator.runner import Plan, ProviderSummary, RunReport
from whichapiapi.selector.pareto import Recommendation


def _usd(v: float | None, digits: int = 5) -> str:
    return "—" if v is None else f"${v:.{digits}f}"


def _s(v: float | None) -> str:
    return "—" if v is None else f"{v / 1000:.1f}s"


def _ratio(s: ProviderSummary) -> str:
    if s.measured_per_case is None or not s.cost_per_case:
        return "—"
    return f"{s.measured_per_case / s.cost_per_case:.0%}"


def _routes(routes: dict[str, int], limit: int = 3) -> str:
    top = sorted(routes.items(), key=lambda kv: -kv[1])[:limit]
    return ", ".join(f"{r} ×{n}" for r, n in top) or "—"


def _mark(s: ProviderSummary, rec: Recommendation) -> str:
    if s.provider == rec.primary:
        return "★"
    if s.provider == rec.fallback:
        return "↺"
    return "·" if s.provider in rec.pareto else ""


def _ordered(rep: RunReport, rec: Recommendation) -> list[ProviderSummary]:
    return sorted(rep.summaries, key=lambda s: -rec.utilities.get(s.provider, -1))


def plan_table(plan: Plan) -> Table:
    t = Table(title=f"Plan (cap ${plan.cap:.2f})")
    t.add_column("provider", no_wrap=True)
    for col in ("calls", "in tok", "out tok", "$/1M in", "$/1M out", "learned ×", "est. cost"):
        t.add_column(col, justify="right")
    for r in [*plan.rows, *([plan.judge] if plan.judge else [])]:
        p = r.price
        per_call = p is not None and p.input_per_1m is None and p.output_per_1m is None
        if per_call:  # priced per request / minute / char: token estimates would be noise
            unit = (
                f"{_usd(p.per_unit, 4)}/{p.unit}"
                if p.per_unit is not None
                else f"{_usd(p.per_request, 4)}/req"
            )
            tokens, prices = ("—", "—"), (unit, "")
        else:
            tokens = (f"{r.est_input_tokens:,}", f"{r.est_output_tokens:,}")
            prices = (_usd(p.input_per_1m if p else None, 3), _usd(p.output_per_1m if p else None, 3))
        t.add_row(
            r.provider + (" (judge)" if r is plan.judge else ""),
            f"{r.calls - r.cached} (+{r.cached} cached)" if r.cached else str(r.calls),
            *tokens,
            *prices,
            "—" if r.ratio is None else f"{r.ratio:.2f}",
            _usd(r.est_cost, 4),
        )
    t.caption = f"estimated total {_usd(plan.total, 4)}" + (
        f" · unpriced: {plan.unpriced}" if plan.unpriced else ""
    )
    return t


def results_table(rep: RunReport, rec: Recommendation) -> Table:
    t = Table(title=f"Results · run {rep.run_id}")
    t.add_column("")
    t.add_column("provider", no_wrap=True)
    for col in (
        "score",
        "pass",
        "err",
        "p50",
        "list $/case",
        "real $/case",
        "real/list",
        "wasted",
        "real $/pass",
    ):
        t.add_column(col, justify="right")
    for s in _ordered(rep, rec):
        t.add_row(
            _mark(s, rec),
            s.label + (" (current)" if s.current else ""),
            f"{s.score:.2f}",
            f"{s.pass_rate:.0%}",
            str(s.errors),
            _s(s.latency_p50),
            _usd(s.cost_per_case),
            _usd(s.measured_per_case),
            _ratio(s),
            _usd(s.wasted_cost, 4) if s.wasted_cost else "—",
            _usd(s.real_cost_per_success),
        )
    t.caption = "★ primary · ↺ fallback · · Pareto-optimal · real = charged by the channel (its call log)"
    return t


def print_report(rep: RunReport, rec: Recommendation, console: Console | None = None) -> None:
    con = console or Console()
    con.print(results_table(rep, rec))
    for line in rec.rationale:
        con.print(f"• {line}")
    con.print(
        f"this run: list-price {_usd(rep.spent_computed, 4)} · actually charged {_usd(rep.spent_measured, 4)}"
        f" (judge {_usd(rep.judge_measured, 4)})" + (" · STOPPED BY BUDGET" if rep.stopped_by_budget else "")
    )


def to_markdown(rep: RunReport, rec: Recommendation) -> str:
    lines = [
        f"# Eval report: {rep.description or rep.suite}",
        "",
        f"- Run `{rep.run_id}` · suite `{rep.suite}` · judge: {rep.judge_label or '—'}",
        f"- This run cost: list price {_usd(rep.spent_computed, 4)}, **actually charged {_usd(rep.spent_measured, 4)}**"
        + (" — **stopped by budget**" if rep.stopped_by_budget else ""),
        "",
        "## Recommendation",
        "",
        *[f"- {r}" for r in rec.rationale],
        "",
        "## Results",
        "",
        "| | Provider | Score | Pass | Err | p50 | List $/case | Real $/case | Real/list | Wasted $ | Real $/pass | Routed to |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in _ordered(rep, rec):
        lines.append(
            f"| {_mark(s, rec)} | {s.label}{' (current)' if s.current else ''} | {s.score:.2f} | {s.pass_rate:.0%} | "
            f"{s.errors} | {_s(s.latency_p50)} | {_usd(s.cost_per_case)} | {_usd(s.measured_per_case)} | "
            f"{_ratio(s)} | {_usd(s.wasted_cost, 4) if s.wasted_cost else '—'} | {_usd(s.real_cost_per_success)} | "
            f"{_routes(s.routes)} |"
        )
    lines += [
        "",
        "★ primary · ↺ fallback · · Pareto-optimal. *List* = tokens × public price list; *real* = what the channel "
        "charged per its own call log (the reseller routes each call to a price group). *Wasted* = charged for "
        "attempts that produced nothing usable (timeouts, truncated reasoning, upstream retries). "
        "*Real $/pass* = (real cost of all cases + wasted) / passed cases.",
    ]
    if rep.judge_routes:
        lines += ["", f"Judge calls routed to: {_routes(rep.judge_routes, 5)}"]
    lines += ["", "## Pass rate by check", ""]
    metrics = sorted({k for s in rep.summaries for k in s.assertion_pass_rates})
    if metrics:
        lines.append("| Provider | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---:|" * len(metrics))
        for s in _ordered(rep, rec):
            vals = [
                f"{s.assertion_pass_rates[m]:.0%}" if m in s.assertion_pass_rates else "—" for m in metrics
            ]
            lines.append(f"| {s.label} | " + " | ".join(vals) + " |")
    lines += ["", "## Failures (first failing check per case)", ""]
    for c in rep.cases:
        if c.skipped or c.passed:
            continue
        reason = c.error or next((f"{g.metric or g.type}: {g.reason}" for g in c.grades if not g.passed), "")
        lines.append(f"- `{c.provider}` #{c.test_idx} {c.description or ''}: {reason[:240]}")
    return "\n".join(lines) + "\n"


def save(rep: RunReport, rec: Recommendation, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(to_markdown(rep, rec), encoding="utf-8")
    payload = {"report": rep.model_dump(mode="json"), "recommendation": rec.model_dump(mode="json")}
    (out_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir
