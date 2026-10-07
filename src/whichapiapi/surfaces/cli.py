"""`whichapiapi` CLI (Typer). Thin: every command delegates to core modules."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import click
import typer
import yaml
from rich.console import Console
from rich.table import Table

from whichapiapi import core
from whichapiapi.adapters.models_dev import ModelsDevAdapter
from whichapiapi.adapters.newapi import NewApiAdapter
from whichapiapi.adapters.openrouter import OpenRouterAdapter
from whichapiapi.evaluator.balance import BalanceProbe
from whichapiapi.evaluator.export import to_promptfoo
from whichapiapi.evaluator.learn import learn_from_log
from whichapiapi.evaluator.runner import BudgetError, EvalRunner, RunReport, summarize
from whichapiapi.report.render import plan_table, print_report, save
from whichapiapi.schema.suite import ProviderSpec, load_suite
from whichapiapi.selector.pareto import recommend
from whichapiapi.store.db import Store

app = typer.Typer(
    help="Which API API: pick APIs for automations by testing them on your tasks.", no_args_is_help=True
)
eval_app = typer.Typer(help="Run eval suites (promptfoo-compatible YAML).", no_args_is_help=True)
offers_app = typer.Typer(help="Pull and query offers (ways to buy a capability).", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(offers_app, name="offers")
models_app = typer.Typer(
    help="Pick the best model variant for a task (quality, cost per task, speed).", no_args_is_help=True
)
app.add_typer(models_app, name="models")
apply_app = typer.Typer(
    help="Apply audit fixes to n8n (reviewed diff, backup, rollback).", no_args_is_help=True
)
app.add_typer(apply_app, name="apply")
audit_app = typer.Typer(help="Audit existing automations (read-only).", no_args_is_help=True)
app.add_typer(audit_app, name="audit")
keys_app = typer.Typer(help="API keys for invited people (public router).", no_args_is_help=True)
app.add_typer(keys_app, name="keys")
providers_app = typer.Typer(
    help="Your own provider keys (free tiers etc.), encrypted, rotated by the router.", no_args_is_help=True
)
app.add_typer(providers_app, name="providers")
integrity_app = typer.Typer(
    help="Is the model behind a model name the one you pay for? (substitution checks)", no_args_is_help=True
)
app.add_typer(integrity_app, name="integrity")
con = Console()


def _filter(suite, only: list[str] | None, limit: int | None):
    if only:
        suite.providers = [p for p in suite.providers if any(o in p.id or o in p.label for o in only)]
    if limit:
        suite.tests = suite.tests[:limit]
    return suite


@eval_app.command("plan")
def eval_plan(
    suite_path: Path,
    only: list[str] = typer.Option(None, "--only", help="Substring filter on provider id/label (repeatable)"),
    limit: int = typer.Option(None, help="Use only the first N tests"),
    budget: float = typer.Option(None, help="Run cap in USD (default: suite x-whichapiapi.budget.run_usd)"),
):
    """Show calls and estimated cost without calling anything."""
    suite = _filter(load_suite(suite_path), only, limit)
    runner = EvalRunner(suite, Store(), suite_path=str(suite_path), budget=budget)
    con.print(plan_table(runner.plan()))


@eval_app.command("run")
def eval_run(
    suite_path: Path,
    only: list[str] = typer.Option(None, "--only", help="Substring filter on provider id/label (repeatable)"),
    limit: int = typer.Option(None, help="Use only the first N tests"),
    budget: float = typer.Option(None, help="Run cap in USD"),
    concurrency: int = typer.Option(4, "-j", "--concurrency"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached results (spends money)"),
    partial: bool = typer.Option(
        False, "--partial", help="Start even if the estimate exceeds the cap; stop at cap"
    ),
    allow_unpriced: bool = typer.Option(False, help="Run providers without a known price"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip llm-rubric assertions (no judge spend)"),
    retry_errors: bool = typer.Option(
        False, "--retry-errors", help="Re-call cases whose cached result is an error"
    ),
    judge: str = typer.Option(
        None, "--judge", help="Override the judge provider id, e.g. myreseller:claude-sonnet-5"
    ),
    out: Path = typer.Option(None, help="Report dir (default reports/<run_id>)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Don't ask for confirmation"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Run the suite against every provider, grade, and recommend primary + fallback."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(asctime)s %(message)s")
    suite = _filter(load_suite(suite_path), only, limit)
    if judge:
        suite.ext.judge = {"provider": {"id": judge, "label": f"judge {judge.split(':', 1)[-1]}"}}
    runner = EvalRunner(
        suite,
        Store(),
        suite_path=str(suite_path),
        budget=budget,
        concurrency=concurrency,
        use_cache=not no_cache,
        allow_unpriced=allow_unpriced,
        use_judge=not no_judge,
        retry_errors=retry_errors,
    )
    plan = runner.plan()
    con.print(plan_table(plan))
    if not yes and not typer.confirm(f"Spend up to ${runner.cap:.2f}?"):
        raise typer.Exit(1)
    try:
        rep = asyncio.run(runner.run(confirm_over_budget=partial))
    except BudgetError as e:
        con.print(f"[red]refused:[/red] {e}")
        raise typer.Exit(2) from e
    rec = recommend(rep.summaries, suite.ext.weights, suite.ext.min_pass_rate)
    print_report(rep, rec, con)
    path = save(rep, rec, out or Path("reports") / rep.run_id)
    con.print(f"report: {path}/report.md")


@eval_app.command("refresh")
def eval_refresh(
    suite_path: Path,
    n: int = typer.Option(4, "-n", help="How many of today's best models for the task to check"),
    preset: str = typer.Option("optimal", help="Selection preset (see `models presets`)"),
    write: bool = typer.Option(False, "--write", help="Add the new models to the suite file"),
    run: bool = typer.Option(False, "--run", help="Re-run the suite (cached results make old models free)"),
    budget: float = typer.Option(None, help="Run cap in USD (default: the suite's)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Don't ask for confirmation"),
):
    """Leaderboard auto-refresh: add the best new models for the suite's task (x-whichapiapi.task), re-run, and exit 3
    when the recommended primary changes. Dry run unless --write/--run; meant for a timer."""
    from whichapiapi import activity
    from whichapiapi.implementer import refresh

    found = refresh.new_candidates(suite_path, n=n, preset=preset)
    if found.get("skipped"):
        con.print(f"skipped: {found['skipped']}")
        return
    con.print(f"task [b]{found['task']}[/b] · today's top {n}: {', '.join(found['top']) or '—'}")
    con.print(f"new for this suite: {', '.join(found['new']) or 'none'}")
    if found["new"] and not write:
        con.print("dry run: add them with --write (and --run to evaluate)")
    if found["new"] and write:
        unknown = refresh.add_providers(suite_path, found["new"])
        con.print(f"added {len(found['new'])} to {suite_path}")
        if unknown:
            con.print(f"[yellow]set base_url/key_env for channels {unknown} in the suite[/yellow]")
    if not run or (found["new"] and not write):  # a run without the new models proves nothing
        return
    suite = load_suite(suite_path)
    runner = EvalRunner(suite, Store(), suite_path=str(suite_path), budget=budget)
    con.print(plan_table(runner.plan()))
    if not yes and not typer.confirm(f"Spend up to ${runner.cap:.2f}?"):
        raise typer.Exit(1)
    try:
        rep = asyncio.run(runner.run())
    except BudgetError as e:
        con.print(f"[red]refused:[/red] {e}")
        raise typer.Exit(2) from e
    rec = recommend(rep.summaries, suite.ext.weights, suite.ext.min_pass_rate)
    path = save(rep, rec, Path("reports") / rep.run_id)
    con.print(f"primary {rec.primary} · fallback {rec.fallback} · report {path}/report.md")
    before = refresh.record_winner(suite_path, rec.primary, rec.fallback)
    if before:
        activity.log_event(
            "refresh",
            str(suite_path),
            {"task": found["task"], "new": found["new"]},
            result={"primary": rec.primary, "was": before.get("primary")},
        )
        con.print(f"[b red]new winner:[/b red] {before.get('primary')} → {rec.primary}")
        raise typer.Exit(3)


@eval_app.command("canary")
def eval_canary(
    suite_path: Path = typer.Argument(Path("examples/canary/suite.yaml")),
    budget: float = typer.Option(0.05, help="Run cap in USD"),
    reset_baseline: bool = typer.Option(
        False, "--reset-baseline", help="Discard the stored baseline and record this run as the new one"
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Don't ask for confirmation"),
):
    """Run the canary suite and diff it against a stored baseline: catches a channel silently substituting
    or degrading the model behind a model name. Spends real money (small, capped by --budget)."""
    if not yes and not typer.confirm(f"Spend up to ${budget:.2f} on the canary suite?"):
        raise typer.Exit(1)
    res = core.run_canary(str(suite_path), budget=budget, reset_baseline=reset_baseline)
    if res["alerts"]:
        con.print("[red]ALERTS[/red]")
        for a in res["alerts"]:
            con.print(f"  {a['label']}: {'; '.join(a['reasons'])}")
    else:
        con.print("[green]no alerts[/green]" + (" (baseline recorded)" if res["baseline_reset"] else ""))
    con.print(f"run {res['run_id']} · spent real ${res['spent_real'] or 0:.4f}")
    if res["alerts"]:
        raise typer.Exit(2)  # lets cron/systemd notice a regression (docs/OPERATIONS.md)


@eval_app.command("report")
def eval_report(run_dir: Path, suite_path: Path = typer.Option(None, "--suite", help="Suite for weights")):
    """Re-aggregate and re-render a saved run (report.json) without calling anything."""
    data = json.loads((run_dir / "report.json").read_text())
    rep = RunReport.model_validate(data["report"])
    suite = load_suite(suite_path) if suite_path else None
    specs = {p.id: p for p in suite.providers} if suite else {}
    rep.summaries = [
        summarize(
            specs.get(s.provider) or ProviderSpec(id=s.provider, label=s.label, model=s.model, channel=s.channel,
                                                  current=s.current),
            [c for c in rep.cases if c.provider == s.provider],
            s.measured_cost,
            s.price,
            s.wasted_cost,
        )
        for s in rep.summaries
    ]  # fmt: skip
    ext = suite.ext if suite else None
    rec = recommend(rep.summaries, ext.weights if ext else None, ext.min_pass_rate if ext else 0.5)
    print_report(rep, rec, con)
    save(rep, rec, run_dir)
    con.print(f"report: {run_dir}/report.md")


@eval_app.command("reconcile")
def eval_reconcile(
    suite_path: Path,
    run_dir: Path,
    lookback_min: float = typer.Option(
        60, help="Read the call log from this long before the run's first spend"
    ),
):
    """Attach real charges to a saved run from the channel's call log (no model calls), then re-render."""
    data = json.loads((run_dir / "report.json").read_text())
    rep = RunReport.model_validate(data["report"])
    suite = load_suite(suite_path)
    store = Store()
    started = next((r for r in store.ledger_rows(rep.run_id)), None)
    since = (
        datetime.fromisoformat(started["ts"]).timestamp() if started else time.time()
    ) - lookback_min * 60
    runner = EvalRunner(suite, store, suite_path=str(suite_path))
    rep = runner.reconcile(rep, since)
    (run_dir / "report.json").write_text(
        json.dumps({"report": rep.model_dump(mode="json")}, ensure_ascii=False)
    )
    eval_report(run_dir, suite_path)


@eval_app.command("compare")
def eval_compare(run_a: Path, run_b: Path):
    """Agreement between two runs of one suite (e.g. two judges): rank correlation + per-case verdicts."""
    from whichapiapi.report.compare import compare_runs, to_markdown

    con.print(to_markdown(compare_runs(run_a, run_b)))


@eval_app.command("pairwise")
def eval_pairwise(
    suite_path: Path,
    run_dir: Path,
    a: str = typer.Option(..., "--a", help="Provider id or label"),
    b: str = typer.Option(..., "--b", help="Provider id or label"),
    max_cases: int = typer.Option(None, help="Compare only the first N cases"),
    budget: float = typer.Option(None, help="Run cap in USD"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Don't ask for confirmation"),
):
    """Head-to-head of two providers from a saved run, judged per case in both orders (2 judge calls per case)."""
    data = json.loads((run_dir / "report.json").read_text())
    rep = RunReport.model_validate(data["report"])
    suite = load_suite(suite_path)
    runner = EvalRunner(suite, Store(), suite_path=str(suite_path), budget=budget)
    if not yes and not typer.confirm(f"Spend up to ${runner.cap:.2f} on judge calls?"):
        raise typer.Exit(1)
    res = asyncio.run(runner.pairwise(rep, a, b, max_cases))
    con.print(
        f"[bold]{res['a']}[/bold] vs [bold]{res['b']}[/bold] · {res['cases']} cases · A wins {res['a_wins']} · "
        f"B wins {res['b_wins']} · ties {res['ties']} · A win rate {res['a_win_rate']} · judge {res['judge']}"
    )
    con.print(
        f"position-biased verdicts (same slot won both orders, counted as ties): {res['position_biased']} · "
        f"judge cost list ${res['judge_cost_computed']:.4f} · real ${res['judge_cost_measured'] or 0:.4f}"
    )
    for r in res["per_case"]:
        con.print(f"  {r['case']}: {r['winner']} {r['votes']}")


@eval_app.command("learn")
def eval_learn(suite_path: Path, days: float = typer.Option(14, help="How far back to read the call log")):
    """Learn real/list price multipliers per model from each channel's call log (no spend)."""
    suite = load_suite(suite_path)
    store = Store()
    for name, ch in suite.ext.channels.items():
        key = os.environ.get(ch.key_env)
        if not ch.balance or not key:
            con.print(f"{name}: skipped (no balance probe or {ch.key_env} unset)")
            continue
        learned = learn_from_log(store, BalanceProbe(ch.balance, ch.base_url, key), name, days)
        t = Table(title=f"{name}: real/list price multipliers from the call log")
        for col in ("model", "calls", "mean ×", "max ×", "groups"):
            t.add_column(col, justify="left" if col == "model" else "right")
        for m, v in sorted(learned.items()):
            t.add_row(m, str(v["n"]), f"{v['mean']:.3f}", f"{v['max']:.3f}", str(v["groups_seen"]))
        con.print(t)
        g = Table(title=f"{name}: which price group actually answered (share of calls)")
        for col in ("model", "calls", "served by", "retried"):
            g.add_column(col, justify="left" if col in ("model", "served by") else "right")
        for row in core.group_usage(name, store):
            served = ", ".join(
                f"{k} {x['share']:.0%}"
                for k, x in sorted(row["groups"].items(), key=lambda kv: -kv[1]["share"])[:4]
            )
            g.add_row(
                row["channel_model"].split("/", 1)[-1], str(row["n"]), served, f"{row['retried_share']:.0%}"
            )
        con.print(g)


@eval_app.command("export-promptfoo")
def eval_export(suite_path: Path, out: Path = typer.Option(..., "-o")):
    """Write a plain promptfoo config for the same suite."""
    out.write_text(yaml.safe_dump(to_promptfoo(load_suite(suite_path)), allow_unicode=True, sort_keys=False))
    con.print(f"wrote {out}")


@offers_app.command("pull")
def offers_pull(
    source: str = typer.Argument(
        ..., help="all | models-dev | openrouter | deepinfra | huggingface | edenai | newapi"
    ),
    url: str = typer.Option(None, help="new-api base URL, e.g. https://api.example.com"),
    channel: str = typer.Option(None, help="Channel name to store new-api offers under, e.g. myreseller"),
    key_env: str = typer.Option(None, help="Env var holding the key for this channel"),
):
    """Fetch offers from a source into the local store (history kept). `all` = every public catalog
    (+ the new-api channel given by --url/--channel)."""
    if source == "all" or source in core.CATALOGS:
        newapi = [{"url": url, "channel": channel, "key_env": key_env}] if url and channel else None
        sources = core.CATALOGS if source == "all" else [source]
        for name, res in core.pull_offers(sources, newapi if source == "all" else None).items():
            con.print(f"{name}: {res}")
        return
    if source == "models-dev":
        adapter = ModelsDevAdapter()
    elif source == "openrouter":
        adapter = OpenRouterAdapter()
    elif source == "newapi":
        if not url or not channel:
            raise typer.BadParameter("--url and --channel are required for newapi")
        adapter = NewApiAdapter(url, channel, key_env=key_env)
    else:
        raise typer.BadParameter(f"unknown source {source}")
    items = list(adapter.fetch())
    new, changed = Store().upsert_offers(items)
    con.print(f"{source}: {len(items)} offers · {new} new · {changed} changed")


@offers_app.command("pull-benchmarks")
def offers_pull_benchmarks():
    """Refresh benchmark scores: every LMArena arena and category (CC BY 4.0). Free, ~1 minute."""
    con.print(core.pull_benchmarks())


@offers_app.command("pull-quality", hidden=True)
def offers_pull_quality():
    """Old name of pull-benchmarks."""
    con.print(core.pull_benchmarks())


@offers_app.command("benchmarks")
def offers_benchmarks(model: str = typer.Argument(None, help="Model to show; omit to list the boards")):
    """Benchmark boards we hold, or every score for one model."""
    t = Table()
    if model is None:
        for col in ("benchmark", "models", "license"):
            t.add_column(col, justify="right" if col == "models" else "left")
        for row in core.benchmark_boards():
            t.add_row(row["benchmark"], str(row["models"]), row.get("license") or "")
    else:
        res = core.model_benchmarks(model)
        for col in ("benchmark", "score", "rank", "95% CI", "votes", "date"):
            t.add_column(col, justify="left" if col == "benchmark" else "right")
        for board, b in res["boards"].items():
            ci = f"{b['ci_low']}–{b['ci_high']}" if b.get("ci_low") is not None else ""
            t.add_row(
                board,
                str(b["score"]),
                str(b.get("rank") or ""),
                ci,
                str(b.get("votes") or ""),
                b.get("date") or "",
            )
        t.title = res["model_key"]
    con.print(t)


@offers_app.command("quality")
def offers_quality(
    benchmark: str = typer.Option("lmarena/text/overall", "-b", help="Board, see `offers benchmarks`"),
    limit: int = typer.Option(40, "-n"),
):
    """Leaderboard for one benchmark board, best first."""
    t = Table(title=benchmark)
    for col in ("model_key", "score", "rank", "organization", "date"):
        t.add_column(col, justify="right" if col in ("score", "rank") else "left")
    for row in core.leaderboard(benchmark, limit):
        t.add_row(
            row["model_key"],
            str(row["score"]),
            str(row.get("rank") or ""),
            row.get("organization") or "",
            row.get("date") or "",
        )
    con.print(t)


@offers_app.command("pull-conditions")
def offers_pull_conditions():
    """Refresh per-model batch/off-peak discount conditions from LiteLLM's pricing JSON (MIT, HTTP only —
    the `litellm` package is never installed, ADR-0003). Powers the batch/off-peak toggles in `estimate_cost`."""
    con.print(core.pull_conditions())


@offers_app.command("conditions")
def offers_conditions(limit: int = typer.Option(40, "-n")):
    """List cached LiteLLM-derived conditions (batch/off-peak discounts)."""
    t = Table()
    for col in ("model_key", "batch_discount", "off_peak_discount", "off_peak_window"):
        t.add_column(col)
    for row in core.conditions_priors()[:limit]:
        off_peak = row.get("off_peak") or {}
        t.add_row(
            row["model_key"],
            str(row.get("batch_discount", "")),
            str(off_peak.get("discount", "")),
            off_peak.get("window", ""),
        )
    con.print(t)


@offers_app.command("cross-check")
def offers_cross_check(model: str = typer.Argument(..., help="Model name, e.g. deepseek/deepseek-v3.2")):
    """Compare our cheapest known offer against a live InferIndex lookup (price sanity + promo detection).
    Informational only — InferIndex is the closest competitor's data, not a source of truth."""
    con.print(core.cross_check_price(model))


@offers_app.command("price-changes")
def offers_price_changes(limit: int = typer.Option(40, "-n")):
    """Recent changedetection.io "pricing page changed" notifications (self-hosted, optional — see
    docs/ARCHITECTURE.md). Only populated once the /webhooks/changedetection endpoint has received something."""
    t = Table()
    for col in ("received_at", "watch_title", "watch_url", "matched_channels"):
        t.add_column(col)
    for row in core.price_change_alerts(limit=limit):
        t.add_row(
            row.get("received_at", ""),
            row.get("watch_title", ""),
            row.get("watch_url", ""),
            ", ".join(row.get("matched_channels", [])),
        )
    con.print(t)


@offers_app.command("list")
def offers_list(
    model: str = typer.Option(None, "-m"),
    channel: str = typer.Option(None, "-c"),
    limit: int = typer.Option(40, "-n"),
    sort: str = typer.Option("input", help="input | output"),
):
    """Search offers in the local store, cheapest first."""
    items = [o for o in Store().find_offers(model=model, channel=channel, limit=5000) if o.price.is_known]
    key = (
        (lambda o: o.price.output_per_1m or 0) if sort == "output" else (lambda o: o.price.input_per_1m or 0)
    )
    t = Table()
    t.add_column("offer", no_wrap=True)
    for col in ("type", "$/1M in", "$/1M out", "cache read", "ctx", "conf."):
        t.add_column(col, justify="right")
    for o in sorted(items, key=key)[:limit]:
        p = o.price
        t.add_row(
            o.id,
            o.channel_type,
            f"{p.input_per_1m}",
            f"{p.output_per_1m}",
            f"{p.cache_read_per_1m or '—'}",
            f"{o.context_window or '—'}",
            o.provenance.confidence,
        )
    con.print(t)


@app.command("mcp")
def mcp_cmd(
    http: bool = typer.Option(False, "--http", help="Serve streamable HTTP instead of stdio"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
):
    """Run the MCP server (stdio by default)."""
    from whichapiapi.surfaces.mcp import serve

    serve(http=http, host=host, port=port)


@audit_app.command("make")
def audit_make_cmd(
    path: Path = typer.Argument(..., help="Make scenario blueprint JSON (Scenario → ⋯ → Export blueprint)"),
    calls: int = typer.Option(None, help="Calls per month for every AI module (default 1000)"),
):
    """Audit a Make scenario's AI modules (same findings as the n8n audit). Read-only."""
    from whichapiapi.implementer.audit import audit_extraction, to_markdown
    from whichapiapi.implementer.sources import extract_make

    con.print(to_markdown(audit_extraction(extract_make(path.read_text(encoding="utf-8")), calls)))


@audit_app.command("code")
def audit_code_cmd(
    path: Path = typer.Argument(..., help="Source file or directory"),
    calls: int = typer.Option(None, help="Calls per month per call site (default 1000)"),
):
    """Audit AI API calls found in source code (endpoint URLs / SDK imports + model literals). Read-only."""
    from whichapiapi.implementer.audit import audit_extraction, to_markdown
    from whichapiapi.implementer.sources import extract_code

    con.print(to_markdown(audit_extraction(extract_code(path), calls)))


@audit_app.command("n8n-suite")
def audit_n8n_suite_cmd(
    node: str = typer.Argument(..., help="Name of the AI node (as shown in the audit)"),
    path: Path = typer.Option(None, "--file", help="Workflow JSON / export"),
    docker: str = typer.Option(None, help="Export from this n8n container instead"),
    alt: list[str] = typer.Option(
        None, "--alt", help="Alternative provider ids, e.g. openrouter:deepseek/deepseek-v4.1-flash"
    ),
    out: Path = typer.Option(
        Path("n8n-suite"), "-o", help="Directory to write suite.yaml, prompt.json, tests.yaml"
    ),
    from_executions: int = typer.Option(
        0, help="Fill tests from the last N saved executions (needs N8N_API_KEY)"
    ),
):
    """Turn one AI node into an eval suite (its prompt, current model vs alternatives, TODO inputs to fill from real
    executions), so a model switch the audit suggests is tested on this node's own traffic."""
    from whichapiapi.implementer.suite_gen import build_suite

    if docker:
        data = _n8n_export(docker)
    elif path:
        data = path.read_text(encoding="utf-8")
    else:
        raise typer.BadParameter("give --file or --docker")
    executions = None
    if from_executions:
        from whichapiapi.implementer.apply import N8nClient
        from whichapiapi.implementer.n8n import load

        client = N8nClient()
        wf_id = next(
            (w.get("id") for w in load(data) if any(n.get("name") == node for n in w.get("nodes") or [])),
            None,
        )
        resp = client.http.get(
            f"{client.base}/executions",
            params={"workflowId": wf_id, "includeData": "true", "limit": from_executions},
            headers=client.headers,
        )
        resp.raise_for_status()
        executions = resp.json().get("data") or []
    files = build_suite(data, node, alternatives=alt or [], executions=executions)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    con.print(
        f"wrote {', '.join(files)} to {out}/ — fill tests.yaml with real inputs, then `eval plan {out}/suite.yaml`"
    )


@audit_app.command("traces-suite")
def audit_traces_suite_cmd(
    source: str = typer.Argument(
        ..., help="Trace export (Langfuse JSON, OTLP/JSON spans, JSON lines) or 'langfuse' to fetch via API"
    ),
    out: Path = typer.Option(
        Path("traces-suite"), "-o", help="Directory for suite.yaml, prompt.json, tests.yaml"
    ),
    title: str = typer.Option("Production traffic", help="Suite description"),
    task: str = typer.Option("general", help="Task for picking candidates (general, coding, russian, …)"),
    sample: int = typer.Option(20, help="How many distinct inputs to keep"),
    name: str = typer.Option(None, help="Langfuse: only generations with this name"),
    since: str = typer.Option(None, help="Langfuse: ISO time to start from"),
    limit: int = typer.Option(500, help="Langfuse: max generations to read"),
    alt: list[str] = typer.Option(None, "--alt", help="Candidates instead of the automatic pick"),
):
    """Production traces → eval suite: real inputs as cases, production's answers as the judge's reference, the
    production model vs today's best for the task. Langfuse needs LANGFUSE_PUBLIC_KEY/SECRET_KEY (+ LANGFUSE_HOST)."""
    from whichapiapi.implementer import traces as tr

    if source == "langfuse":
        pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if not pk or not sk:
            raise typer.BadParameter("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        data: object = tr.fetch_langfuse(host, pk, sk, name=name, since=since, limit=limit)
    else:
        data = Path(source).read_text(encoding="utf-8")
    calls = tr.parse(data)
    result = tr.build_suite(calls, out, title, task=task, sample=sample, candidates=alt or None)
    con.print_json(data={k: v for k, v in result.items() if k != "next"})
    con.print(f"next: whichapiapi eval plan {out}/suite.yaml")


@apply_app.command("n8n-retries")
def apply_n8n_retries(
    workflow: str = typer.Argument(..., help="Workflow name, part of it, or id"),
    yes: bool = typer.Option(False, "--yes", help="Apply (default: show the diff only)"),
    repo: Path = typer.Option(
        None, help="Git checkout holding the workflow JSON, updated with the same change"
    ),
):
    """Add 'Retry On Fail' (3 tries, 5 s apart) to AI calls without retry, error output or fallback model.
    Shows a diff; with --yes backs up the live workflow, applies, verifies, and updates the git copy."""
    from whichapiapi.implementer.apply import N8nClient, retry_fix

    rep = retry_fix(N8nClient(), workflow, dry_run=not yes, repo_dir=repo)
    con.print(f"[bold]{rep['workflow']}[/bold] ({rep['id']}): {len(rep['changes'])} node(s) to change")
    for c in rep["changes"]:
        con.print(f"  • {c['node']} ← protects {', '.join(c['because'])}")
    if rep["diff"]:
        con.print(rep["diff"], markup=False, highlight=False)
    if yes and rep["changes"]:
        state = "applied and verified" if rep["applied"] else f"NOT applied: {rep.get('not_applied')}"
        con.print(f"{state} · backup {rep.get('backup')} · git copy {rep.get('git_file')}")
        con.print(f"undo: whichapiapi apply n8n-rollback {rep.get('backup')}")
    elif rep["changes"]:
        con.print("dry run — nothing sent. Re-run with --yes to apply.")


@apply_app.command("n8n-fallbacks")
def apply_n8n_fallbacks(
    workflow: str = typer.Argument(..., help="Workflow name, part of it, or id"),
    model: str = typer.Option(..., help="Fallback model id on that channel, e.g. deepseek-v4.1-flash"),
    credential_id: str = typer.Option(..., help="n8n credential id of ANOTHER channel (type openAiApi)"),
    credential_name: str = typer.Option("Fallback", help="Its name in n8n"),
    base_url: str = typer.Option(None, help="OpenAI-compatible base URL of that channel"),
    yes: bool = typer.Option(False, "--yes", help="Apply (default: show the plan only)"),
    repo: Path = typer.Option(
        None, help="Git checkout holding the workflow JSON, updated with the same change"
    ),
):
    """Attach a fallback model from another channel to every single-model chain/agent ("Enable Fallback Model").
    n8n uses it only when the primary fails. Backup, verify and git sync as with n8n-retries."""
    from whichapiapi.implementer.apply import N8nClient, fallback_fix

    cred = {"openAiApi": {"id": credential_id, "name": credential_name}}
    rep = fallback_fix(N8nClient(), workflow, model, cred, base_url, dry_run=not yes, repo_dir=repo)
    con.print(f"[bold]{rep['workflow']}[/bold]: {len(rep['plans'])} chain(s)")
    for p in rep["plans"]:
        con.print(f"  • {p['chain']}: + {p['new_node']} ({p['model']})")
    if yes and rep["plans"]:
        state = "applied and verified" if rep["applied"] else f"NOT applied: {rep.get('not_applied')}"
        con.print(f"{state} · backup {rep.get('backup')} · git copy {rep.get('git_file')}")
    elif rep["plans"]:
        con.print("plan only — nothing sent. Re-run with --yes to apply.")


@apply_app.command("n8n-rollback")
def apply_n8n_rollback(backup: Path = typer.Argument(..., help="Backup file printed by n8n-retries")):
    """Restore a workflow from its backup."""
    from whichapiapi.implementer.apply import N8nClient, rollback

    con.print(rollback(N8nClient(), backup))


@models_app.command("rank")
def models_rank(
    preset: str = typer.Option("optimal", "-p", help="See `models presets`"),
    task: str = typer.Option(
        None, "-t", help="general|coding|math|agents|russian|writing|long_context|vision"
    ),
    profile: str = typer.Option(None, help="chat|classify|summarize|rag|agent_step|code|long_doc"),
    limit: int = typer.Option(10, "-n"),
):
    """Rank model variants: quality blended from benchmarks for the task, cost per task incl. thinking tokens,
    seconds per task."""
    res = core.rank_models(preset, task=task, profile=profile, limit=limit)
    t = Table(
        title=f"{preset} · task {res['task']} · {res['profile']['in_tokens']}→{res['profile']['out_tokens']} tok"
    )
    for col in ("model", "score", "quality", "$/1k tasks", "s/task", "think tok", "buy at"):
        t.add_column(col, justify="left" if col in ("model", "buy at") else "right")
    for m in res["models"]:
        t.add_row(
            m["model"],
            f"{m['score']:.1f}",
            f"{m['quality']:.0f}" + ("" if m["coverage"] >= 0.99 else f" ({m['coverage']:.0%})"),
            f"{m['cost_per_1k_tasks']:.3f}",
            f"{m['seconds_per_task']:.1f}" if m["seconds_per_task"] else "—",
            str(m["thinking_tokens"] or 0),
            (m["buy"] or {}).get("offer_id") or "",
        )
    con.print(t)


@models_app.command("presets")
def models_presets():
    """Ready-made categories."""
    from whichapiapi.selector.cards import PRESETS

    for key, p in PRESETS.items():
        con.print(f"{key:13} {p['title']}")


@models_app.command("page")
def models_page(out: Path = typer.Argument(Path("model-picker.html"))):
    """Build the interactive model picker page (sliders for quality / cost / speed) with today's data embedded."""
    data = json.dumps(core.export_model_data(), ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    template = (Path(__file__).parent / "picker.html").read_text(encoding="utf-8")
    out.write_text(template.replace("/*__MODEL_DATA__*/", data), encoding="utf-8")
    con.print(f"{out}: {out.stat().st_size // 1024} KB")


@models_app.command("export")
def models_export(out: Path = typer.Argument(Path("model-data.json"))):
    """Write the data the interactive model picker uses (cards, scoring tables, media leaderboards)."""
    data = core.export_model_data()
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    con.print(f"{out}: {len(data['cards'])} model variants, {out.stat().st_size // 1024} KB")


def _n8n_export(container: str) -> str:
    """All workflows of a local n8n container, via its own CLI (read-only; no API key needed)."""
    import subprocess

    script = (
        "n8n export:workflow --all --output=/tmp/wf.json >/dev/null && cat /tmp/wf.json && rm /tmp/wf.json"
    )
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", script], check=True, capture_output=True, text=True
    ).stdout


@audit_app.command("n8n")
def audit_n8n_cmd(
    path: Path = typer.Argument(None, help="Workflow JSON (one workflow, a list, or an n8n export)"),
    docker: str = typer.Option(None, help="Export all workflows from this n8n container instead of a file"),
    calls: int = typer.Option(None, help="Calls per month for every AI node (default: from the schedule)"),
    out: Path = typer.Option(None, help="Also write the Markdown report here"),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report"),
):
    """Audit n8n workflows: overpay, cheaper equal-quality models, missing fallbacks, training-data risk,
    deprecated/unknown models, API keys written into nodes. Read-only: nothing is changed or called."""
    from whichapiapi.implementer.audit import audit_n8n, to_markdown

    if docker:
        data = _n8n_export(docker)
    elif path:
        data = path.read_text(encoding="utf-8")
    else:
        raise typer.BadParameter("give a workflow file or --docker <container>")
    rep = audit_n8n(data, calls_per_month=calls)
    md = to_markdown(rep)
    if out:
        out.write_text(md, encoding="utf-8")
    con.print_json(json.dumps(rep, ensure_ascii=False)) if as_json else con.print(md)


@app.command()
def spend():
    """Show total computed and measured eval spend recorded in the ledger."""
    s = Store()
    con.print(f"list-price ${s.spent_total():.4f} · real ${s.real_spent_total():.4f}")
    if cap := os.environ.get("WHICHAPIAPI_BUDGET_TOTAL"):
        con.print(f"global cap ${float(cap):.2f}")


@app.command("runs-page")
def runs_page(
    out: Path = typer.Argument(Path("runs.html")),
    roots: list[Path] = typer.Option(
        None, "--dir", help="Where run folders live (default: reports/, docs/reports/)"
    ),
    limit: int = typer.Option(30, help="Newest N runs"),
):
    """Dashboard of your own eval runs (Pareto chart, per-provider table, cases explorer) as one HTML page."""
    dirs = []
    for root in roots or [Path("reports"), Path("docs/reports")]:
        dirs += [p.parent for p in root.glob("*/report.json")]
    dirs = sorted(set(dirs), key=lambda d: (d / "report.json").stat().st_mtime, reverse=True)[:limit]
    runs = []
    for d in dirs:
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")).get("report") or {}
        for c in rep.get("cases") or []:
            c["output"] = (c.get("output") or "")[:600]  # keep the page small
        runs.append({"dir": d.name, "report": rep})
    data = json.dumps(runs, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (Path(__file__).parent / "runs.html").read_text(encoding="utf-8")
    out.write_text(template.replace("/*__RUNS__*/", data), encoding="utf-8")
    con.print(f"{out}: {len(runs)} runs, {out.stat().st_size // 1024} KB")


@keys_app.command("add")
def keys_add(
    name: str = typer.Argument(..., help="Who the key is for"),
    limit: float = typer.Option(0.20, "--limit", help="USD this key may spend"),
    role: str = typer.Option(
        "tester", help="tester: router + hosted MCP, fully traced · guest: router only, no content logged"
    ),
):
    """Create (or replace) a key; it is printed once and stored only as a hash."""
    from whichapiapi import keys

    key = keys.add(name, limit, role)
    con.print(f"{role} key for [b]{name}[/b] (limit ${limit:.2f}) — shown once:")
    print(key)


@keys_app.command("list")
def keys_list():
    """Keys with limit, spend and last use."""
    from whichapiapi import keys

    t = Table("name", "role", "hint", "limit $", "spent $", "calls", "last used", "revoked")
    for name, r in keys.list_keys().items():
        t.add_row(
            name, r.get("role", "guest"), r.get("hint", ""), f"{r.get('limit_usd', 0):.2f}", f"{r.get('spent_usd', 0):.4f}",
            str(r.get("calls", 0)), r.get("last_used", "—"), "yes" if r.get("revoked") else "",
        )  # fmt: skip
    con.print(t)


@keys_app.command("limit")
def keys_limit(name: str, usd: float):
    """Change a key's USD limit."""
    from whichapiapi import keys

    con.print("ok" if keys.set_limit(name, usd) else f"no key named {name!r}")


@keys_app.command("revoke")
def keys_revoke(name: str):
    """Disable a key at once."""
    from whichapiapi import keys

    con.print("revoked" if keys.revoke(name) else f"no key named {name!r}")


@app.command("traces")
def traces_cmd(
    name: str = typer.Argument(None, help="Tester key name (default: all)"),
    limit: int = typer.Option(20, "-n"),
    full: bool = typer.Option(False, "--full", help="Print whole records as JSON"),
):
    """What testers sent and got back (full traces of tester keys)."""
    from whichapiapi import tracing

    for r in tracing.read(name, limit):
        if full:
            print(json.dumps(r, ensure_ascii=False, indent=1))
            continue
        if r.get("type") == "chat":
            last = next(
                (m.get("content") for m in reversed(r.get("messages") or []) if m.get("role") == "user"), ""
            )
            con.print(
                f"{r['ts'][:19]} [b]{r['key']}[/b] chat {r.get('requested')}→{r.get('route')} "
                f"${r.get('cost_usd') or 0:.5f} {r.get('latency_ms')}ms\n  ▸ {str(last)[:160]}\n  ◂ {str(r.get('output'))[:160]}"
            )
        else:
            con.print(
                f"{r['ts'][:19]} [b]{r['key']}[/b] tool {r.get('tool')} {json.dumps(r.get('arguments'), ensure_ascii=False)[:160]}"
            )


@app.command("serve-api")
def serve_api(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8766)):
    """Public OpenAI-compatible router for invited keys (put nginx/HTTPS in front)."""
    from whichapiapi.surfaces.public_api import serve

    serve(host, port)


@app.command()
def activity(
    days: float = typer.Option(7, help="Look back this many days"),
    tail: int = typer.Option(0, help="Also print the last N events"),
):
    """Usage log of MCP tools, REST requests and CLI commands: calls, errors, durations, projects, latest errors."""
    from whichapiapi import activity as act

    rep = act.report(days)
    con.print(
        f"{rep['events']} events · {rep['errors']} errors · last {days:g} days · log dir {act.log_dir()}"
    )
    t = Table()
    for col in ("action", "calls", "errors", "p50 ms", "max ms"):
        t.add_column(col, justify="left" if col == "action" else "right")
    for name, a in rep["by_action"].items():
        t.add_row(name, str(a["calls"]), str(a["errors"]), str(a["p50_ms"] or ""), str(a["max_ms"] or ""))
    con.print(t)
    con.print("projects: " + ", ".join(f"{k} ({v})" for k, v in rep["by_project"].items()))
    con.print("clients: " + ", ".join(f"{k} ({v})" for k, v in rep["by_client"].items()))
    for e in rep["recent_errors"]:
        con.print(f"[red]{e['ts']} {e['surface']}:{e['action']}[/red] {e['error']}")
    for e in act.read_events(days)[-tail:] if tail else []:
        con.print_json(json.dumps(e, ensure_ascii=False))


def main() -> None:
    """Console entry point: runs the CLI and records the command in the activity log."""
    import sys

    from whichapiapi.activity import track

    argv = sys.argv[1:]
    action = " ".join(a for a in argv[:2] if not a.startswith("-")) or "help"
    if action in ("mcp", "activity"):  # the server logs its own tool calls; reading the log isn't activity
        app()
        return
    with track("cli", action, {"argv": argv}) as t:
        try:
            app(standalone_mode=False)
        except SystemExit as e:
            t["ok"] = e.code in (0, None)
            raise
        except click.exceptions.Exit as e:
            t["ok"] = e.exit_code == 0
            raise SystemExit(e.exit_code) from e


if __name__ == "__main__":
    main()


_MARK = {
    "ok": "[green]ok[/green]",
    "info": "[dim]info[/dim]",
    "warn": "[yellow]warn[/yellow]",
    "alert": "[red]ALERT[/red]",
}


def _notify_owner(text: str) -> None:
    """Best effort: send to the owner's Telegram via ~/.local/bin/tg-owner when it exists (see ~/.claude/CLAUDE.md)."""
    import shutil
    import subprocess

    tool = shutil.which("tg-owner") or str(Path.home() / ".local/bin/tg-owner")
    if Path(tool).exists():
        subprocess.run([tool, "send", text], env={**os.environ, "TG_OWNER_FROM": "whichapiapi integrity"},
                       timeout=60, check=False)  # fmt: skip


@integrity_app.command("run")
def integrity_run(
    watch: Path = typer.Argument(Path("examples/integrity/watch.yaml")),
    model: list[str] = typer.Option([], "--model", "-m", help="Only these models (repeatable)"),
    reset_baseline: bool = typer.Option(
        False, "--reset-baseline", help="Record this run as the new baseline"
    ),
    notify: bool = typer.Option(
        False, "--notify", help="Send alerts (and status changes) to the owner's Telegram"
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Don't ask for confirmation"),
):
    """Check that each watched model is what its name says: tokenizer, hidden prompt, identity, censorship,
    behavioural fingerprint vs its own baseline and its cheaper sibling. ~20 short calls per model (real money).
    Exit code 2 on alerts."""
    if not yes and not typer.confirm(f"Run paid integrity probes for {watch}?"):
        raise typer.Exit(1)
    before = {f"{r['channel']}:{r['model']}": r["status"] for r in core.integrity_status()}
    res = core.run_integrity(str(watch), reset_baseline=reset_baseline, only=model or None)
    lines = []
    for m in res["models"]:
        con.print(f"{_MARK[m['status']]} [bold]{m['channel']}:{m['model']}[/bold] ({m['family']})"
                  + (" · baseline recorded" if m["baseline_reset"] else ""))  # fmt: skip
        for c in m["checks"]:
            con.print(f"   {_MARK[c['status']]} {c['check']}: {c['detail']}")
        if m["groups"]:
            con.print(f"   [dim]groups: {m['groups']}[/dim]")
        name = f"{m['channel']}:{m['model']}"
        if m["status"] == "alert" or (before.get(name) and before[name] != m["status"]):
            bad = [f"{c['check']}: {c['detail']}" for c in m["checks"] if c["status"] in ("alert", "warn")]
            lines.append(f"{m['status'].upper()} {name}\n  " + "\n  ".join(bad or ["all checks ok"]))
    con.print(f"run {res['run_id']} · spent real ${res['spent_real']:.5f}"
              + (f" · {res.get('unmatched_calls')} calls not found in the log" if res.get("unmatched_calls") else ""))  # fmt: skip
    if notify and lines:
        _notify_owner("Model integrity check:\n" + "\n".join(lines))
    if res["alerts"]:
        raise typer.Exit(2)


@integrity_app.command("status")
def integrity_status_cmd():
    """The last verdict per watched model (no calls, no spend)."""
    for r in core.integrity_status():
        con.print(f"{_MARK[r['status']]} {r['channel']}:{r['model']} · {r['at']}")
        for c in r["checks"]:
            if c["status"] in ("warn", "alert"):
                con.print(f"   {_MARK[c['status']]} {c['check']}: {c['detail']}")


_DEFAULT_ROUTER = os.environ.get("WHICHAPIAPI_ROUTER_URL", "https://whichapiapi.sluice.stream")


def _router_key(key: str | None) -> str:
    k = key or os.environ.get("WHICHAPIAPI_KEY") or os.environ.get("WHICHAPIAPI_MCP_TOKEN")
    if not k:
        con.print("[red]no key: pass --key wa-… (or set WHICHAPIAPI_KEY)[/red]")
        raise typer.Exit(1)
    return k


@app.command("setup")
def setup_cmd(
    tool: str = typer.Argument(..., help="claude | codex | aider | opencode | continue"),
    url: str = typer.Option(_DEFAULT_ROUTER, help="Router base URL"),
    key: str | None = typer.Option(
        None, help="Router key (wa-… or the owner token); default $WHICHAPIAPI_KEY"
    ),
    model: str = typer.Option("auto:coding", help="Routing policy the tool should use"),
    global_: bool = typer.Option(
        False, "--global", help="claude: write ~/.claude/settings.json (all projects)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show the diff"),
):
    """Point a coding agent at the router: edits its config (backup kept next to it, key redacted in the diff)."""
    from whichapiapi import agents

    res = agents.setup(tool, url, _router_key(key), model, dry_run=dry_run, global_=global_)
    con.print(res["diff"] or "[dim]no change[/dim]", markup=False)
    if not dry_run and res["changed"]:
        con.print(
            f"[green]wrote[/green] {res['path']}"
            + (f" (backup {res['backup']})" if res.get("backup") else "")
        )
        if res.get("next"):
            con.print(res["next"])


@app.command("launch", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def launch_cmd(
    ctx: typer.Context,
    tool: str = typer.Argument(..., help="claude | codex | aider"),
    url: str = typer.Option(_DEFAULT_ROUTER, help="Router base URL"),
    key: str | None = typer.Option(None, help="Router key; default $WHICHAPIAPI_KEY"),
    model: str = typer.Option("auto:coding", help="Routing policy"),
):
    """Start a coding agent against the router without touching any config file (extra args go to the tool)."""
    from whichapiapi import agents

    env, args = agents.launch_env(tool, url, _router_key(key), model)
    exe = {"claude": "claude", "codex": "codex", "aider": "aider"}[tool]
    if not shutil.which(exe):
        con.print(f"[red]{exe} is not installed[/red]")
        raise typer.Exit(1)
    os.execvpe(exe, [exe, *args, *ctx.args], env)


@app.command("doctor")
def doctor_cmd(url: str = typer.Option(_DEFAULT_ROUTER, help="Router base URL")):
    """Where does each coding agent really send its requests? routed / shadowed / elsewhere / unreachable / unknown."""
    from whichapiapi import agents

    colors = {"routed": "green", "shadowed": "yellow", "degraded": "yellow", "elsewhere": "dim", "unknown": "dim",
              "unreachable": "red"}  # fmt: skip
    for r in agents.doctor(url):
        con.print(f"[{colors.get(r['verdict'], 'white')}]{r['verdict']:<11}[/] {r['tool']:<9} {r['detail']}")


@providers_app.command("add")
def providers_add(
    provider: str = typer.Argument(
        ..., help="groq, google, cerebras, mistral, nvidia, openrouter, … (see list)"
    ),
    label: str = typer.Option("", help="A name for this key"),
    base_url: str | None = typer.Option(
        None, help="OpenAI-compatible base URL for a provider not in the list"
    ),
):
    """Add a key (asked for with hidden input, never echoed or written in clear)."""
    from whichapiapi import keypool

    key = typer.prompt(f"{provider} API key", hide_input=True)
    con.print(keypool.add(provider, key, label, base_url))


@providers_app.command("import")
def providers_import(path: Path = typer.Argument(..., help=".env or JSON file with provider keys")):
    """Import keys from a .env (GROQ_API_KEY=…, GEMINI_API_KEY=…) or JSON file; providers detected by variable name."""
    from whichapiapi import keypool

    pairs = keypool.parse_file(path.read_text())
    for prov, key in pairs:
        rec = keypool.add(prov, key)
        con.print(f"  {rec['provider']:<12} {rec['hint']}  ({rec['label']})")
    con.print(f"{len(pairs)} key(s) imported")


@providers_app.command("list")
def providers_list():
    """Stored keys (hints only) and every provider the pool knows."""
    from whichapiapi import keypool

    for k in keypool.listing():
        state = "on" if k["enabled"] else f"[red]off[/red] ({k.get('last_error')})"
        con.print(f"{k['id']}  {k['provider']:<12} {k['hint']:<14} {k['label']:<16} {state}")
    con.print(f"[dim]known providers: {', '.join(sorted(keypool.PROVIDERS))}[/dim]")


@providers_app.command("remove")
def providers_remove(key_id: str):
    from whichapiapi import keypool

    con.print("removed" if keypool.remove(key_id) else "no such key id")


def render_dashboard(days: float = 7.0) -> str:
    data = json.dumps(core.dashboard_data(days), ensure_ascii=False, separators=(",", ":"), default=str)
    template = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    return template.replace("/*__DATA__*/", data.replace("</", "<\\/"))


@app.command("dashboard")
def dashboard_cmd(
    out: Path = typer.Argument(Path("dashboard.html")),
    days: float = typer.Option(7.0, help="Period"),
):
    """The owner's dashboard as one HTML file: real vs list spend, router traffic and attempt trails, route health,
    model integrity, balance forecast, keys. Also served live at /dashboard on the public host (owner login)."""
    out.write_text(render_dashboard(days), encoding="utf-8")
    con.print(f"{out}: {out.stat().st_size // 1024} KB")
