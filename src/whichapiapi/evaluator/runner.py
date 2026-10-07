"""Eval runner: plan → budget check → call candidates → grade → aggregate (ARCHITECTURE §4.1).

Providers run one after another (cases concurrently inside a provider). After each provider block the runner
reads the channel's own accounting (per-call log where available, ADR-0008), attaches the *measured* cost and
routing to every call, and re-bases the budget on real money. Judging runs as a second phase, measured the
same way.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from whichapiapi.cost.engine import call_cost
from whichapiapi.evaluator.assertions import AssertContext, GradingResult, combine, render, run_assertion
from whichapiapi.evaluator.balance import BalanceProbe
from whichapiapi.evaluator.judge import (
    JUDGE_SCHEMA,
    PAIRWISE_SCHEMA,
    judge_messages,
    pairwise_messages,
    parse_grade,
    parse_winner,
)
from whichapiapi.schema.offer import Price
from whichapiapi.schema.suite import Assertion, Prompt, ProviderSpec, Suite, TestCase, judge_spec
from whichapiapi.store.db import Store
from whichapiapi.transports.base import CallResult, Transport, Usage
from whichapiapi.transports.deepgram import DeepgramTransport
from whichapiapi.transports.mock import MockTransport
from whichapiapi.transports.openai_compat import OpenAICompatTransport
from whichapiapi.transports.openai_stt import OpenAISTTTransport
from whichapiapi.transports.tavily import TavilyTransport

log = logging.getLogger(__name__)

LOG_SETTLE_S = 3.0  # give the channel a moment to write its call log
TRANSIENT_ERRORS = ("RateLimitError", "InternalServerError", "APIConnectionError", "budget")


class BudgetError(RuntimeError):
    pass


# ------------------------------------------------------------------ result models


class CaseResult(BaseModel):
    provider: str
    test_idx: int
    prompt: str
    description: str | None = None
    output: str = ""
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    cost: float | None = None  # computed: usage x list price (cached or not)
    cost_measured: float | None = None  # what the channel actually charged (its call log), if known
    route: str | None = None  # where the channel routed the call, e.g. "group-a x0.037"
    cached: bool = False
    grades: list[GradingResult] = Field(default_factory=list)
    passed: bool = False
    score: float = 0.0
    skipped: str | None = None  # e.g. "budget"


class ProviderSummary(BaseModel):
    provider: str
    label: str
    model: str
    channel: str | None
    current: bool = False
    n: int
    errors: int
    skipped: int
    pass_rate: float
    score: float
    latency_p50: float | None
    latency_p95: float | None
    cost_per_case: float | None  # list price
    cost_total: float | None
    cost_per_success: float | None
    measured_cost: float | None = None  # actually paid in this run (fresh calls only)
    measured_per_case: float | None = None  # real cost per case, cached cases included when measured earlier
    wasted_cost: float = 0.0  # charged in this run for attempts that produced nothing usable
    real_cost_per_success: float | None = None  # (real cost of all cases + wasted) / passed cases
    routes: dict[str, int] = Field(default_factory=dict)
    price: Price | None = None
    assertion_pass_rates: dict[str, float] = Field(default_factory=dict)

    @property
    def effective_cost_per_case(self) -> float | None:
        return self.measured_per_case if self.measured_per_case is not None else self.cost_per_case


class PlanRow(BaseModel):
    provider: str
    calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost: float | None
    price: Price | None
    ratio: float | None = None  # learned measured/list multiplier applied to the estimate
    cached: int = 0  # calls already answered in the results cache: free, not in the estimate


class Plan(BaseModel):
    rows: list[PlanRow]
    judge: PlanRow | None
    total: float | None
    cap: float
    unpriced: list[str]


class RunReport(BaseModel):
    run_id: str
    suite: str
    description: str
    plan: Plan
    cases: list[CaseResult]
    summaries: list[ProviderSummary]
    judge_label: str | None
    judge_cost: float = 0.0
    judge_measured: float | None = None
    judge_routes: dict[str, int] = Field(default_factory=dict)
    spent_computed: float = 0.0
    spent_measured: float | None = None
    stopped_by_budget: bool = False


# ------------------------------------------------------------------ helpers


def _cache_key(base_url: str, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> str:
    blob = json.dumps([base_url, model, messages, config], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _render_messages(prompt: Prompt, vars_: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**m, "content": render(str(m.get("content", "")), vars_)} for m in prompt.messages]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1], 1)


TransportFactory = Callable[[ProviderSpec], tuple[Transport, str]]


class EvalRunner:
    def __init__(
        self,
        suite: Suite,
        store: Store,
        *,
        suite_path: str = "",
        budget: float | None = None,
        concurrency: int = 4,
        use_cache: bool = True,
        allow_unpriced: bool = False,
        transport_factory: TransportFactory | None = None,
        env: dict[str, str] | None = None,
        measure: bool = True,
        use_judge: bool = True,
        retry_errors: bool = False,
    ):
        self.suite = suite
        self.store = store
        self.suite_path = suite_path
        self.cap = min(x for x in (budget, suite.ext.budget.run_usd) if x is not None)
        self.concurrency = concurrency
        self.use_cache = use_cache
        self.allow_unpriced = allow_unpriced
        self.env = env if env is not None else dict(os.environ)
        self.transport_factory = transport_factory or self._default_transport
        self.measure = measure
        self.retry_errors = retry_errors
        self.judge = judge_spec(suite) if use_judge else None
        self.run_id = uuid.uuid4().hex[:12]
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = asyncio.Lock()
        self._stopped = False

    # -------------------------------------------------------------- wiring
    def _default_transport(self, spec: ProviderSpec) -> tuple[Transport, str]:
        if spec.channel == "mock":
            return MockTransport(), "mock://"
        ch = self.suite.ext.channels[spec.channel or ""]
        key = self.env.get(ch.key_env)
        if not key:
            raise BudgetError(f"env var {ch.key_env} is not set (needed for {spec.id})")
        if ch.protocol == "tavily":
            return TavilyTransport(ch.base_url, key, timeout=ch.timeout_s), ch.base_url
        if ch.protocol == "deepgram":
            dg = DeepgramTransport(ch.base_url, key, timeout=ch.timeout_s, base_dir=self.suite.base_dir)
            return dg, ch.base_url
        if ch.protocol == "openai_stt":
            stt = OpenAISTTTransport(ch.base_url, key, timeout=ch.timeout_s, base_dir=self.suite.base_dir)
            return stt, ch.base_url
        return OpenAICompatTransport(ch.base_url, key, timeout=ch.timeout_s), ch.base_url

    def price_for(self, spec: ProviderSpec) -> Price | None:
        if spec.price is not None:
            return spec.price
        if spec.channel == "mock":
            return Price(input_per_1m=0.0, output_per_1m=0.0)
        offer = self.store.get_offer(f"{spec.channel}/{spec.model}")
        return offer.price if offer and offer.price.is_known else None

    def _probe(self, spec: ProviderSpec | None) -> BalanceProbe | None:
        if spec is None or not self.measure or spec.channel in (None, "mock"):
            return None
        ch = self.suite.ext.channels.get(spec.channel)
        if not ch or not ch.balance or not self.env.get(ch.key_env):
            return None
        return BalanceProbe(ch.balance, ch.base_url, self.env[ch.key_env])

    # -------------------------------------------------------------- learned price ratio
    def learned_ratio(self, spec: ProviderSpec) -> float | None:
        """Max measured/list ratio seen for this channel+model: a conservative planning multiplier."""
        hit = self.store.cache_get(f"ratio:{spec.channel}/{spec.model}")
        return float(hit["max"]) if hit else None

    def _learn_ratio(self, spec: ProviderSpec, pairs: list[tuple[float, float]]) -> None:
        pairs = [(m, c) for m, c in pairs if c > 0]
        if not pairs:
            return
        key = f"ratio:{spec.channel}/{spec.model}"
        prev = self.store.cache_get(key) or {"max": 0.0, "n": 0}
        self.store.cache_put(
            key,
            {
                **prev,  # keeps `last_ts` of `eval learn`'s log reads
                "max": round(max(prev["max"], *(m / c for m, c in pairs)), 5),
                "mean": round(sum(m for m, _ in pairs) / sum(c for _, c in pairs), 5),
                "n": prev["n"] + len(pairs),
            },
        )

    # -------------------------------------------------------------- planning
    def _estimate(
        self, spec: ProviderSpec, n_in_chars: int, calls: int, out_per_call: int | None = None
    ) -> PlanRow:
        b = self.suite.ext.budget
        price = self.price_for(spec)
        if out_per_call is None:
            cap = int(spec.config.get("max_tokens") or spec.config.get("max_completion_tokens") or 0)
            out_per_call = min(cap, b.default_output_tokens) if cap else b.default_output_tokens
        in_tok = int(n_in_chars / b.chars_per_token)
        est = call_cost(
            price, in_tok, out_per_call * calls, requests=calls, units=b.default_units_per_call * calls
        )
        ratio = self.learned_ratio(spec)
        if est is not None:
            est *= (ratio if ratio is not None else 1.0) * b.safety_factor
        return PlanRow(
            provider=spec.id,
            calls=calls,
            est_input_tokens=in_tok,
            est_output_tokens=out_per_call * calls,
            est_cost=round(est, 6) if est is not None else None,
            price=price,
            ratio=ratio,
        )

    def cases(self) -> list[tuple[int, TestCase, Prompt]]:
        return [(i, t, p) for i, t in enumerate(self.suite.tests) for p in self.suite.prompts]

    def plan(self) -> Plan:
        cases = self.cases()
        rows = []
        judge_calls = judge_chars = 0
        if self.judge:
            j_known = self.judge.channel == "mock" or self.judge.channel in self.suite.ext.channels
            j_base = self._base_url(self.judge) if j_known else ""
            j_config = {"temperature": 0, "response_format": JUDGE_SCHEMA, **self.judge.config}

        def count_judge(t: TestCase, output: str | None) -> None:
            """Judge calls this case will make: all rubrics for a fresh answer; for a cached answer only the rubrics
            whose verdict isn't cached (re-runs after adding a model then cost just the new model's judging)."""
            nonlocal judge_calls, judge_chars
            if not self.judge:
                return
            for a in self.suite.all_asserts(t):
                if a.type != "llm-rubric":
                    continue
                rubric = render(str(a.value), self.suite.vars_for(t))
                if output is not None and self.use_cache:
                    key = _cache_key(j_base, self.judge.model, judge_messages(rubric, output), j_config)
                    if self.store.cache_get(key):
                        continue
                judge_calls += 1
                judge_chars += len(rubric) + 3000

        for spec in self.suite.providers:
            known = spec.channel == "mock" or spec.channel in self.suite.ext.channels
            base, chars, cached = (self._base_url(spec) if known else ""), 0, 0
            for _, t, p in cases:
                msgs = _render_messages(p, self.suite.vars_for(t))
                hit = (
                    self.store.cache_get(_cache_key(base, spec.model, msgs, spec.config))
                    if self.use_cache
                    else None
                )
                if hit and not (
                    hit.get("error") and (self.retry_errors or hit["error"].startswith(TRANSIENT_ERRORS))
                ):
                    cached += 1
                    count_judge(t, hit.get("output") or "")
                    continue
                count_judge(t, None)
                chars += len(json.dumps(msgs, ensure_ascii=False))
            row = self._estimate(spec, chars, len(cases) - cached)
            rows.append(row.model_copy(update={"calls": len(cases), "cached": cached}))
        judge_row = None
        if self.judge and judge_calls:
            judge_row = self._estimate(self.judge, judge_chars, judge_calls, out_per_call=400)
        all_rows = [*rows, *([judge_row] if judge_row else [])]
        unpriced = [r.provider for r in all_rows if r.est_cost is None]
        total = round(sum(r.est_cost for r in all_rows if r.est_cost is not None), 6)
        return Plan(rows=rows, judge=judge_row, total=total, cap=self.cap, unpriced=unpriced)

    # -------------------------------------------------------------- budget
    async def _reserve(self, est: float) -> bool:
        async with self._lock:
            if self._spent + self._reserved + est > self.cap:
                self._stopped = True
                return False
            self._reserved += est
            return True

    async def _settle(self, est: float, actual: float) -> None:
        async with self._lock:
            self._reserved -= est
            self._spent += actual

    # -------------------------------------------------------------- calling
    def _expected_ratio(self, spec: ProviderSpec) -> float:
        """Best guess of real/list for in-flight budget accounting until the block is re-based on the log."""
        hit = self.store.cache_get(f"ratio:{spec.channel}/{spec.model}")
        return float(hit["mean"]) if hit else 1.0

    async def _call(
        self,
        spec: ProviderSpec,
        transport: Transport,
        base_url: str,
        messages: list[dict[str, Any]],
        kind: str,
        config: dict[str, Any],
        price: Price | None,
        est: float,
    ) -> tuple[CallResult, float | None]:
        key = _cache_key(base_url, spec.model, messages, config)
        hit = self.store.cache_get(key) if self.use_cache else None
        if hit and not (hit.get("error") and self.retry_errors):
            res = CallResult.model_validate(hit)
            res.cached, res.cache_key = True, key
            u = res.usage
            return res, call_cost(price, u.input_tokens, u.output_tokens, u.cached_tokens, units=u.units)
        if not await self._reserve(est):
            return CallResult(error="budget"), None
        res = await transport.call(spec.model, messages, config)
        res.cache_key = key
        u = res.usage
        cost = call_cost(price, u.input_tokens, u.output_tokens, u.cached_tokens, units=u.units)
        await self._settle(est, (cost or 0.0) * self._expected_ratio(spec))
        if cost:
            self.store.spend(self.run_id, spec.channel or "", spec.model, kind, cost, "computed")
        # errors are outcomes too (timeouts, truncation) — cache them unless transient
        if self.use_cache and not (res.error and res.error.startswith(TRANSIENT_ERRORS)):
            self.store.cache_put(key, res.model_dump(exclude={"cached", "raw", "cache_key"}))
        return res, cost

    def _measure_block(
        self, spec: ProviderSpec, since: float, fresh: list[tuple[CallResult, float | None]], kind: str
    ) -> tuple[float | None, float, list[tuple[float, float]]]:
        """Attach the real charge + route to fresh calls from the channel's log; re-base the budget on it.

        Returns (charged for our successful calls, charged for nothing usable = failed attempts, timeouts,
        retries or other traffic on this key in the window, [(real, list)] pairs for ratio learning).
        """
        probe = self._probe(spec)
        if probe is None or not fresh or not probe.has_call_log:
            return None, 0.0, []
        entries = probe.calls_since(since - 5, spec.model)
        used: set[int] = set()
        pairs: list[tuple[float, float]] = []
        ok = [(r, c) for r, c in fresh if r.error is None]

        def match(res: CallResult, completion_only: bool) -> int | None:
            u = res.usage
            best: tuple[int, int] | None = None
            for i, e in enumerate(entries):
                if i in used:
                    continue
                d = abs(e.completion_tokens - u.output_tokens)
                if not completion_only:
                    d += abs(e.prompt_tokens - u.input_tokens)
                if best is None or d < best[0]:
                    best = (d, i)
            tol = max(20, 0.02 * ((0 if completion_only else u.input_tokens) + u.output_tokens))
            return best[1] if best and best[0] <= tol else None

        def attach(res: CallResult, computed: float | None, i: int) -> None:
            used.add(i)
            e = entries[i]
            res.measured_cost = round(e.cost_usd, 8)
            res.route = f"{e.group} x{e.group_ratio}" if e.group else None
            if computed:
                pairs.append((e.cost_usd, computed))
            if res.cache_key and self.use_cache:
                self.store.cache_put(res.cache_key, res.model_dump(exclude={"cached", "raw", "cache_key"}))

        # pass 1: prompt+completion (not for audio calls: their log tokens come from audio length)
        for res, computed in ok:
            if not res.usage.units and (i := match(res, False)) is not None:
                attach(res, computed, i)
        # pass 2, per-unit calls (audio): the log counts tokens from audio length (new-api: ~1000 per minute), not
        # what the API returned. Pair leftovers by size, longest to longest — only when the counts agree, so a
        # retry or other traffic on the key never gets attributed to a case.
        left = sorted(
            (x for x in ok if x[0].measured_cost is None and x[0].usage.units > 0),
            key=lambda x: x[0].usage.units,
        )
        free = sorted(
            (i for i in range(len(entries)) if i not in used), key=lambda i: entries[i].prompt_tokens
        )
        if left and len(left) == len(free):
            for (res, computed), i in zip(left, free, strict=True):
                attach(res, computed, i)
        # pass 3: completion only (some upstreams log prompt tokens wrongly); meaningless for audio calls
        for res, computed in ok:
            if res.measured_cost is None and not res.usage.units and (i := match(res, True)) is not None:
                attach(res, computed, i)
        matched = round(sum(r.measured_cost or 0.0 for r, _ in ok), 8)
        wasted = round(sum(e.cost_usd for i, e in enumerate(entries) if i not in used), 8)
        n_matched = sum(r.measured_cost is not None for r, _ in ok)
        if n_matched < len(ok):
            log.warning("%s: matched %d of %d calls in the channel log", spec.id, n_matched, len(ok))
        # budget: replace the in-flight guesses for this block by what was really charged
        guessed = sum((c or 0.0) * self._expected_ratio(spec) for _, c in fresh)
        self._spent += matched + wasted - guessed
        self.store.spend(self.run_id, spec.channel or "", spec.model, kind, matched + wasted, "measured")
        return matched, wasted, pairs

    def reconcile(self, report: RunReport, since: float) -> RunReport:
        """After-the-fact measurement of a saved run from the channel's call log (logs are kept by the channel).

        Matches every successful case to a log line, stores real cost + route on the case and in the cache.
        Wasted (unmatched) charges are not recomputed here: the window may contain unrelated traffic.
        """
        for spec in self.suite.providers:
            probe = self._probe(spec)
            if probe is None or not probe.has_call_log:
                continue
            mine = [c for c in report.cases if c.provider == spec.id and not c.skipped]
            rows = [c for c in mine if not c.error]
            fresh: list[tuple[CallResult, float | None]] = []
            for c in mine:
                prompt = next(p for p in self.suite.prompts if p.label == c.prompt)
                msgs = _render_messages(prompt, self.suite.vars_for(self.suite.tests[c.test_idx]))
                key = _cache_key(self._base_url(spec), spec.model, msgs, spec.config)
                res = CallResult(
                    output=c.output, usage=c.usage, latency_ms=c.latency_ms, error=c.error, cache_key=key
                )
                if c.error:  # keep non-transient failures as outcomes so re-runs don't pay for them again
                    if not c.error.startswith(TRANSIENT_ERRORS) and self.store.cache_get(key) is None:
                        self.store.cache_put(key, res.model_dump(exclude={"cached", "raw", "cache_key"}))
                    continue
                fresh.append((res, c.cost))
            saved, self._spent = self._spent, 0.0
            _, _, pairs = self._measure_block(spec, since, fresh, "reconcile")
            self._spent = saved
            self._learn_ratio(spec, pairs)
            for c, (res, _) in zip(rows, fresh, strict=True):
                if res.measured_cost is not None:
                    c.cost_measured, c.route = res.measured_cost, res.route
        return report

    async def pairwise(
        self, report: RunReport, a: str, b: str, max_cases: int | None = None
    ) -> dict[str, Any]:
        """Head-to-head of two providers of a saved run, judged per case in BOTH orders (position bias cancels):
        A wins a case only if the judge prefers it both times; a split or a tie counts as a tie. Uses the suite's
        judge, cache, budget and ledger like `run`."""
        if not self.judge:
            raise BudgetError("the suite has no judge (x-whichapiapi.judge)")
        pick = {p.id: p for p in self.suite.providers}
        ids = {x: next((i for i in pick if x in (i, pick[i].label)), None) for x in (a, b)}
        if None in ids.values():
            raise ValueError(f"unknown provider(s): {[x for x, v in ids.items() if v is None]}")
        by_case: dict[tuple[int, str], dict[str, CaseResult]] = {}
        for c in report.cases:
            if c.provider in ids.values() and not c.error and not c.skipped and c.output:
                by_case.setdefault((c.test_idx, c.prompt), {})[c.provider] = c
        pairs = [(k, v[ids[a]], v[ids[b]]) for k, v in sorted(by_case.items()) if len(v) == 2][:max_cases]
        spec, (transport, base_url) = self.judge, self.transport_factory(self.judge)
        price = self.price_for(spec)
        config = {"temperature": 0, "response_format": PAIRWISE_SCHEMA, **spec.config}
        per_call = (
            (call_cost(price, 6000, 200) or 0.0) * self.suite.ext.budget.safety_factor if price else 0.0
        )
        prompts = {p.label: p for p in self.suite.prompts}
        spent = 0.0
        since = time.time()
        fresh: list[tuple[CallResult, float | None]] = []

        async def verdict(task: str, crit: str | None, first: str, second: str) -> str | None:
            nonlocal spent
            res, cost = await self._call(
                spec,
                transport,
                base_url,
                pairwise_messages(task, first, second, crit),
                "judge",
                config,
                price,
                per_call,
            )
            spent += cost or 0.0
            if not res.cached and res.error != "budget":
                fresh.append((res, cost))
            return None if res.error else parse_winner(res.output)

        rows = []
        for (idx, prompt_label), ca, cb in pairs:
            test = self.suite.tests[idx]
            msgs = _render_messages(prompts[prompt_label], self.suite.vars_for(test))
            task = "\n\n".join(f"[{m['role']}] {m['content']}" for m in msgs)[-12000:]
            rubric = next(
                (str(x.value) for x in self.suite.all_asserts(test) if x.type == "llm-rubric"), None
            )
            crit = render(rubric, self.suite.vars_for(test))[:4000] if rubric else None
            v1, v2 = await asyncio.gather(
                verdict(task, crit, ca.output, cb.output), verdict(task, crit, cb.output, ca.output)
            )
            winner = (
                "a" if (v1, v2) == ("first", "second") else "b" if (v1, v2) == ("second", "first") else "tie"
            )
            rows.append({"case": test.description or idx, "winner": winner, "votes": [v1, v2]})
        measured = None
        if (
            fresh and self._probe(spec) is not None
        ):  # real judge charges into the ledger (global cap counts them)
            await asyncio.sleep(LOG_SETTLE_S)
            measured, wasted, _ = self._measure_block(spec, since, fresh, "judge")
            measured = round((measured or 0.0) + wasted, 6)
        n = len(rows)
        wins_a = sum(r["winner"] == "a" for r in rows)
        wins_b = sum(r["winner"] == "b" for r in rows)
        return {
            "a": ids[a],
            "b": ids[b],
            "cases": n,
            "a_wins": wins_a,
            "b_wins": wins_b,
            "ties": n - wins_a - wins_b,
            "a_win_rate": round((wins_a + 0.5 * (n - wins_a - wins_b)) / n, 3) if n else None,
            "judge": spec.id,
            "judge_cost_computed": round(spent, 6),
            "judge_cost_measured": measured,
            "position_biased": sum(
                1 for r in rows if r["votes"][0] == r["votes"][1] and r["votes"][0] != "tie"
            ),
            "per_case": rows,
        }

    def _base_url(self, spec: ProviderSpec) -> str:
        if spec.channel == "mock":
            return "mock://"
        return self.suite.ext.channels[spec.channel or ""].base_url

    async def _run_provider(self, spec: ProviderSpec, row: PlanRow) -> list[CaseResult]:
        transport, base_url = self.transport_factory(spec)
        price = row.price
        per_call_est = (row.est_cost or 0.0) / max(row.calls, 1)
        sem = asyncio.Semaphore(self.concurrency)
        fresh: list[tuple[CallResult, float | None]] = []
        since = time.time()

        async def one(idx: int, test: TestCase, prompt: Prompt) -> tuple[CaseResult, CallResult]:
            async with sem:
                msgs = _render_messages(prompt, self.suite.vars_for(test))
                res, cost = await self._call(
                    spec, transport, base_url, msgs, "candidate", spec.config, price, per_call_est
                )
                if not res.cached and res.error != "budget":
                    fresh.append((res, cost))
                cr = CaseResult(
                    provider=spec.id,
                    test_idx=idx,
                    prompt=prompt.label,
                    description=test.description,
                    output=res.output,
                    error=res.error,
                    usage=res.usage,
                    latency_ms=res.latency_ms,
                    cost=cost,
                    cached=res.cached,
                )
                if res.error == "budget":
                    cr.error, cr.skipped = None, "budget"
                return cr, res

        pairs = list(await asyncio.gather(*(one(i, t, p) for i, t, p in self.cases())))
        if fresh and self._probe(spec) is not None:
            await asyncio.sleep(LOG_SETTLE_S)  # let the channel write its call log
        measured, wasted, ratio_pairs = self._measure_block(spec, since, fresh, "candidate")
        self._block_wasted[spec.id] = wasted
        self._learn_ratio(spec, ratio_pairs)
        for cr, res in pairs:
            cr.cost_measured, cr.route = res.measured_cost, res.route
        self._block_measured[spec.id] = measured
        return [cr for cr, _ in pairs]

    async def _grade(self, cr: CaseResult, test: TestCase, judge_fn: Any) -> None:
        if cr.skipped:
            return
        if cr.error:
            cr.passed, cr.score = False, 0.0
            return
        ctx = AssertContext(
            vars=self.suite.vars_for(test),
            base_dir=self.suite.base_dir,
            latency_ms=cr.latency_ms,
            cost=cr.cost,
            judge=judge_fn,
            extra={"provider": cr.provider, "prompt": cr.prompt},
        )
        grades = [await run_assertion(a, cr.output, ctx) for a in self.suite.all_asserts(test)]
        cr.grades = grades
        threshold = test.threshold if test.threshold is not None else self.suite.default_test.threshold
        cr.passed, cr.score = combine(grades, threshold)

    def _judge_fn(self) -> tuple[Any, list[tuple[CallResult, float | None]]]:
        fresh: list[tuple[CallResult, float | None]] = []
        if not self.judge:
            return None, fresh
        spec = self.judge
        transport, base_url = self.transport_factory(spec)
        price = self.price_for(spec)
        config = {"temperature": 0, "response_format": JUDGE_SCHEMA, **spec.config}
        per_call = 0.0
        if price and price.is_known:
            per_call = (call_cost(price, 4000, 400) or 0.0) * self.suite.ext.budget.safety_factor
            per_call *= self.learned_ratio(spec) or 1.0
        sem = asyncio.Semaphore(self.concurrency)

        async def fn(rubric: str, output: str, a: Assertion) -> GradingResult:
            async with sem:
                res, cost = await self._call(
                    spec,
                    transport,
                    base_url,
                    judge_messages(rubric, output),
                    "judge",
                    config,
                    price,
                    per_call,
                )
            if not res.cached and res.error != "budget":
                fresh.append((res, cost))
            if res.error == "budget":
                return GradingResult(
                    type=a.type, passed=False, score=0.0, weight=0.0, reason="skipped: budget"
                )
            if res.error:
                return GradingResult(
                    type=a.type, passed=False, score=0.0, weight=a.weight, reason=f"judge error: {res.error}"
                )
            return parse_grade(res.output, a, cost or 0.0)

        return fn, fresh

    def _check_balance_floor(self, plan: Plan) -> None:
        """Refuse to run if the key's balance would drop below the channel's `min_balance` (shared keys)."""
        for name, ch in self.suite.ext.channels.items():
            if ch.min_balance is None or not ch.balance or not self.env.get(ch.key_env):
                continue
            left = BalanceProbe(ch.balance, ch.base_url, self.env[ch.key_env]).remaining_usd()
            planned = sum(
                r.est_cost or 0.0 for r in [*plan.rows, plan.judge] if r and r.provider.startswith(name + ":")
            )
            if left is not None and left - planned < ch.min_balance:
                raise BudgetError(
                    f"{name}: balance ${left:.2f} − planned ${planned:.2f} would go below the ${ch.min_balance:.2f} floor"
                )

    # -------------------------------------------------------------- main
    async def run(self, *, confirm_over_budget: bool = False) -> RunReport:
        plan = self.plan()
        if plan.unpriced and not self.allow_unpriced:
            raise BudgetError(
                f"no price for {plan.unpriced}; pull offers (whichapiapi offers pull) or set x-whichapiapi.price"
            )
        total_cap = self.env.get("WHICHAPIAPI_BUDGET_TOTAL")
        if total_cap is not None:
            already = self.store.real_spent_total()
            if already >= float(total_cap):
                raise BudgetError(f"global budget exhausted: ${already:.4f} real of ${float(total_cap):.2f}")
            self.cap = min(self.cap, float(total_cap) - already)
        self._check_balance_floor(plan)
        if plan.total and plan.total > self.cap and not confirm_over_budget:
            raise BudgetError(
                f"estimated ${plan.total:.4f} exceeds cap ${self.cap:.4f}; lower scope or pass --partial"
            )
        self.store.start_run(self.run_id, self.suite_path, {"plan": plan.model_dump(mode="json")})

        self._block_measured: dict[str, float | None] = {}
        self._block_wasted: dict[str, float] = {}
        results: list[CaseResult] = []
        for spec, row in zip(self.suite.providers, plan.rows, strict=True):
            log.info("run %s: provider %s", self.run_id, spec.id)
            results.extend(await self._run_provider(spec, row))

        judge_fn, judge_fresh = self._judge_fn()
        jsince = time.time()
        tests = self.suite.tests
        await asyncio.gather(*(self._grade(cr, tests[cr.test_idx], judge_fn) for cr in results))
        judge_measured = None
        if self.judge:
            if judge_fresh and self._probe(self.judge) is not None:
                await asyncio.sleep(LOG_SETTLE_S)
            judge_measured, judge_wasted, jpairs = self._measure_block(
                self.judge, jsince, judge_fresh, "judge"
            )
            if judge_measured is not None:
                judge_measured = round(judge_measured + judge_wasted, 8)
            self._learn_ratio(self.judge, jpairs)
        judge_routes = Counter(r.route for r, _ in judge_fresh if r.route)
        self._judge_fresh_count = len(judge_fresh)

        summaries = [
            summarize(
                spec,
                [r for r in results if r.provider == spec.id],
                self._block_measured.get(spec.id),
                plan.rows[i].price,
                self._block_wasted.get(spec.id, 0.0),
            )
            for i, spec in enumerate(self.suite.providers)
        ]
        meas = [v for v in [*self._block_measured.values(), judge_measured] if v is not None]
        meas.append(sum(self._block_wasted.values()))
        report = RunReport(
            run_id=self.run_id,
            suite=self.suite_path,
            description=self.suite.description,
            plan=plan,
            cases=results,
            summaries=summaries,
            judge_label=self.judge.label if self.judge else None,
            judge_cost=round(sum(c or 0.0 for _, c in judge_fresh), 6),
            judge_measured=judge_measured,
            judge_routes=dict(judge_routes),
            spent_computed=round(
                sum(r.cost or 0.0 for r in results if not r.cached) + sum(c or 0.0 for _, c in judge_fresh), 6
            ),
            spent_measured=round(sum(meas), 6) if meas else None,
            stopped_by_budget=self._stopped,
        )
        self.store.finish_run(
            self.run_id,
            "budget_stopped" if self._stopped else "done",
            {"spent_computed": report.spent_computed, "spent_measured": report.spent_measured},
        )
        return report


def _real_per_case(rows: list[CaseResult], list_per_case: float | None) -> float | None:
    """List cost per case x the real/list ratio observed on the measured calls (robust to partial coverage)."""
    pairs = [(r.cost_measured, r.cost) for r in rows if r.cost_measured is not None and r.cost]
    if not pairs or list_per_case is None:
        return None
    return round(list_per_case * sum(m for m, _ in pairs) / sum(c for _, c in pairs), 8)


def summarize(
    spec: ProviderSpec,
    rows: list[CaseResult],
    measured: float | None,
    price: Price | None,
    wasted: float = 0.0,
) -> ProviderSummary:
    done = [r for r in rows if not r.skipped]
    ok = [r for r in done if not r.error]
    lat = sorted(r.latency_ms for r in ok if r.latency_ms > 0)
    costs = [r.cost for r in done if r.cost is not None]
    passes = sum(r.passed for r in done)
    total = round(sum(costs), 6) if costs else None
    real_pc = _real_per_case(done, total / len(costs) if total is not None and costs else None)
    per_assert: dict[str, list[bool]] = {}
    for r in done:
        for g in r.grades:
            if g.weight > 0:
                per_assert.setdefault(g.metric or g.type, []).append(g.passed)
    return ProviderSummary(
        provider=spec.id,
        label=spec.label,
        model=spec.model,
        channel=spec.channel,
        current=spec.current,
        n=len(done),
        errors=len(done) - len(ok),
        skipped=len(rows) - len(done),
        pass_rate=round(passes / len(done), 4) if done else 0.0,
        score=round(statistics.fmean(r.score for r in done), 4) if done else 0.0,
        latency_p50=_pct(lat, 50),
        latency_p95=_pct(lat, 95),
        cost_per_case=round(total / len(costs), 8) if total is not None and costs else None,
        cost_total=total,
        cost_per_success=round(total / passes, 8) if total is not None and passes else None,
        measured_cost=measured,
        measured_per_case=real_pc,
        wasted_cost=round(wasted, 8),
        real_cost_per_success=round((real_pc * len(done) + wasted) / passes, 8)
        if real_pc is not None and passes
        else None,
        routes=dict(Counter(r.route for r in done if r.route)),
        price=price,
        assertion_pass_rates={k: round(sum(v) / len(v), 3) for k, v in per_assert.items()},
    )
