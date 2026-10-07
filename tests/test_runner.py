import json

import pytest

from whichapiapi.evaluator.export import to_promptfoo
from whichapiapi.evaluator.runner import BudgetError, EvalRunner
from whichapiapi.schema.offer import Price
from whichapiapi.schema.suite import load_suite
from whichapiapi.selector.pareto import recommend
from whichapiapi.transports.mock import MockTransport

GRADE = json.dumps({"reason": "fine", "score": 0.9, "pass": True})


def factory(spec):
    return MockTransport({"judge": GRADE}), "mock://"


def make(fix, store, **kw):
    suite = load_suite(fix / "suite/suite.yaml")
    return suite, EvalRunner(suite, store, suite_path="t", transport_factory=factory, env={}, **kw)


async def test_end_to_end(fix, store):
    suite, r = make(fix, store)
    suite.ext.judge = {"provider": {"id": "mock:judge", "x-whichapiapi": {"price": {"input_per_1m": 0.0}}}}
    r.judge = r.judge.model_copy(update={"price": Price(input_per_1m=0.0, output_per_1m=0.0)})
    judge_before = r.plan().judge.calls
    rep = await r.run()
    by = {s.label: s for s in rep.summaries}
    assert by["good"].pass_rate == 1.0 and by["good"].score > 0.9
    assert by["bad"].pass_rate == 0.0  # no JSON, python check fails
    assert by["broken"].errors == 2
    assert by["good"].cost_per_case is not None and by["good"].cost_per_case > by["bad"].cost_per_case
    rec = recommend(rep.summaries)
    assert rec.primary == suite.providers[0].id
    assert rep.spent_computed > 0
    assert rep.judge_cost == 0.0  # mock judge is free, but its calls must be accounted
    assert r._judge_fresh_count > 0
    # second run is fully cached -> no new spend
    _, r2 = make(fix, store)
    r2.judge = r.judge
    replan = r2.plan().judge  # cached verdicts aren't planned again
    assert replan is None or replan.calls < judge_before
    rep2 = await r2.run()
    assert rep2.spent_computed == 0.0
    assert all(c.cached for c in rep2.cases if c.provider != suite.providers[2].id)


async def test_budget_refuses_and_stops(fix, store):
    _, r = make(fix, store, budget=1e-12)
    with pytest.raises(BudgetError):
        await r.run()
    _, r = make(fix, store, budget=1e-9, use_cache=False)
    rep = await r.run(confirm_over_budget=True)
    assert rep.stopped_by_budget
    assert any(c.skipped == "budget" for c in rep.cases)


async def test_unpriced_refused(fix, store):
    suite, r = make(fix, store)
    suite.providers[0].price = None
    suite.providers[0].channel = "nowhere"
    with pytest.raises(BudgetError):
        await r.run()


def test_plan(fix, store):
    _, r = make(fix, store)
    plan = r.plan()
    assert [row.calls for row in plan.rows] == [2, 2, 2]
    assert plan.judge is not None and plan.judge.calls == 6


def test_export(fix):
    out = to_promptfoo(load_suite(fix / "suite/suite.yaml"))
    assert "x-whichapiapi" not in out and out["providers"] == []  # mock providers are dropped


class FakeProbe:
    """Pretends the channel charged 10 % of list price for every call, routed to group 'G1'."""

    has_call_log = True

    def __init__(self, runner):
        self.runner = runner

    def calls_since(self, ts, model=None):
        from whichapiapi.evaluator.balance import CallLogEntry

        out = []
        for _key, value in self.runner.store.engine.connect().execute(
            __import__("sqlalchemy").text("select key, value from cache where key not like 'ratio:%'")
        ):
            v = __import__("json").loads(value) if isinstance(value, str) else value
            u = v["usage"]
            out.append(CallLogEntry(ts=0, model=model or "", cost_usd=0.0, prompt_tokens=u["input_tokens"],
                                    completion_tokens=u["output_tokens"], group="G1", group_ratio=0.1))  # fmt: skip
        return out


async def test_measured_cost_and_learned_ratio(fix, store, monkeypatch):
    import whichapiapi.evaluator.runner as runner_mod

    monkeypatch.setattr(runner_mod, "LOG_SETTLE_S", 0)
    suite, r = make(fix, store, use_judge=False)
    suite.providers = suite.providers[:1]
    monkeypatch.setattr(r, "_probe", lambda spec: FakeProbe(r) if spec else None)
    rep = await r.run()
    s = rep.summaries[0]
    assert s.routes == {"G1 x0.1": 2}
    assert s.measured_per_case == 0.0 and s.effective_cost_per_case == 0.0
    assert rep.spent_measured == 0.0
    assert r.learned_ratio(suite.providers[0]) == 0.0


def test_spearman():
    from whichapiapi.report.compare import spearman

    assert spearman({"a": 3, "b": 2, "c": 1}, {"a": 30, "b": 20, "c": 10}) == 1.0
    assert spearman({"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 3}) == -1.0
    assert spearman({"a": 1, "b": 2}, {"a": 1, "b": 2}) is None


def test_measure_pairs_audio_calls_by_length(fix, store):
    """STT: new-api logs tokens derived from audio length, not the API's usage — pair calls by size instead."""
    from whichapiapi.evaluator.balance import CallLogEntry
    from whichapiapi.transports.base import CallResult, Usage

    suite, r = make(fix, store, use_judge=False)
    spec = suite.providers[0]

    class Probe:
        has_call_log = True

        def calls_since(self, ts, model):
            return [
                CallLogEntry(
                    ts=1, model=spec.model, cost_usd=c, prompt_tokens=p, completion_tokens=0, group="Az"
                )
                for c, p in ((0.5, 500), (0.1, 100), (0.2, 217))
            ]

    r._probe = lambda s: Probe()
    calls = [
        CallResult(output="x", usage=Usage(input_tokens=58, output_tokens=23, units=u))
        for u in (0.2, 0.5, 0.1)
    ]
    matched, wasted, _ = r._measure_block(spec, 0.0, [(c, None) for c in calls], "candidate")
    assert [c.measured_cost for c in calls] == [0.2, 0.5, 0.1]
    assert matched == 0.8 and wasted == 0.0
    # a count mismatch (retry / other traffic on the key) must not be guessed at
    calls = [CallResult(output="x", usage=Usage(units=u)) for u in (0.2, 0.5)]
    r._measure_block(spec, 0.0, [(c, None) for c in calls], "candidate")
    assert all(c.measured_cost is None for c in calls)


def test_vendor_of():
    from whichapiapi.selector.pareto import vendor_of

    assert vendor_of("gpt-4o-transcribe") == vendor_of("whisper-1") == "openai"
    assert vendor_of("deepseek-v4.1-flash") == "deepseek"
    assert vendor_of("nvidia/parakeet-tdt-0.6b-v3") == "nvidia"
    assert vendor_of("advanced", "tavily") == vendor_of("basic", "tavily") == "tavily"
    assert vendor_of("somenewmodel-2") == "somenewmodel"


async def test_pairwise_judges_both_orders(fix, store):
    from whichapiapi.transports.base import CallResult

    class Judge:
        """Prefers whichever answer mentions Paris; a position-biased judge would always say "first"."""

        async def call(self, model, messages, config):
            text = messages[-1]["content"]
            first = text.split("<FIRST>")[1].split("</FIRST>")[0]
            second = text.split("<SECOND>")[1].split("</SECOND>")[0]
            winner = "first" if "Paris" in first else "second" if "Paris" in second else "tie"
            return CallResult(output=json.dumps({"reason": "r", "winner": winner}))

    def fac(spec):
        return (Judge(), "mock://") if spec.model == "judge" else (MockTransport({}), "mock://")

    suite = load_suite(fix / "suite/suite.yaml")
    runner = EvalRunner(suite, store, suite_path="t", transport_factory=fac, env={}, use_judge=False)
    rep = await runner.run()
    suite.ext.judge = {"provider": {"id": "mock:judge", "x-whichapiapi": {"price": {"input_per_1m": 0.0}}}}
    judged = EvalRunner(suite, store, suite_path="t", transport_factory=fac, env={})
    judged.judge = judged.judge.model_copy(update={"price": Price(input_per_1m=0.0, output_per_1m=0.0)})
    res = await judged.pairwise(rep, "good", "bad")
    assert res["cases"] == 2 and res["a_wins"] == 2 and res["a_win_rate"] == 1.0
    assert all(r["votes"] == ["first", "second"] for r in res["per_case"])
