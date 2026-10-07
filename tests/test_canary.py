from whichapiapi import core
from whichapiapi.evaluator.canary import compare_to_baseline, new_baselines
from whichapiapi.evaluator.runner import ProviderSummary


def _summary(**kw) -> ProviderSummary:
    base = dict(
        provider="reseller:gpt-6-luna", label="gpt-6-luna", model="gpt-6-luna", channel="reseller",
        n=5, errors=0, skipped=0, pass_rate=1.0, score=1.0, latency_p50=500.0, latency_p95=600.0,
        cost_per_case=0.001, cost_total=0.005, cost_per_success=0.001,
    )  # fmt: skip
    base.update(kw)
    return ProviderSummary(**base)


def test_new_baselines_skips_empty_providers():
    empty = _summary(n=0, errors=0, pass_rate=0.0, score=0.0)
    good = _summary()
    assert new_baselines([empty, good]) == {
        "reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}
    }


def test_compare_to_baseline_no_baseline_yet_is_not_an_alert():
    assert compare_to_baseline([_summary()], {}) == []


def test_compare_to_baseline_flags_pass_rate_drop():
    baseline = {"reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}}
    current = _summary(pass_rate=0.6, score=0.9)  # 0.4 drop >= 0.2 threshold
    alerts = compare_to_baseline([current], baseline)
    assert len(alerts) == 1 and "pass rate dropped" in alerts[0]["reasons"][0]


def test_compare_to_baseline_flags_score_drop():
    baseline = {"reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}}
    current = _summary(pass_rate=1.0, score=0.8)  # 0.2 drop >= 0.15 threshold
    alerts = compare_to_baseline([current], baseline)
    assert len(alerts) == 1 and "score dropped" in alerts[0]["reasons"][0]


def test_compare_to_baseline_flags_latency_spike():
    baseline = {"reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}}
    current = _summary(latency_p50=1200.0)  # > 2x baseline
    alerts = compare_to_baseline([current], baseline)
    assert len(alerts) == 1 and "latency_p50" in alerts[0]["reasons"][0]


def test_compare_to_baseline_flags_all_errored():
    baseline = {"reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}}
    current = _summary(n=5, errors=5, pass_rate=0.0, score=0.0)
    alerts = compare_to_baseline([current], baseline)
    assert len(alerts) == 1 and alerts[0]["reasons"] == ["every call errored"]


def test_compare_to_baseline_clean_run_no_alert():
    baseline = {"reseller:gpt-6-luna": {"pass_rate": 1.0, "score": 1.0, "latency_p50": 500.0, "n": 5}}
    assert compare_to_baseline([_summary()], baseline) == []


async def test_arun_canary_records_baseline_then_stays_clean(store, fix):
    suite_path = str(fix / "canary_suite" / "suite.yaml")
    res1 = await core.arun_canary(suite_path, budget=1.0, store=store)
    assert res1["baseline_reset"] is True
    stored = store.cache_get(f"canary:{suite_path}")
    assert stored and "mock:const:ok" in stored

    res2 = await core.arun_canary(suite_path, budget=1.0, store=store)
    assert res2["alerts"] == [] and res2["baseline_reset"] is False
    status = core.canary_status(store=store)
    assert status[0]["run_id"] == res2["run_id"] and status[0]["alerts"] == []
    assert status[0]["providers"]["mock:const:ok"]["baseline"]["n"] > 0


async def test_arun_canary_reset_baseline_discards_old_one(store, fix):
    suite_path = str(fix / "canary_suite" / "suite.yaml")
    await core.arun_canary(suite_path, budget=1.0, store=store)
    res = await core.arun_canary(suite_path, budget=1.0, reset_baseline=True, store=store)
    assert res["baseline_reset"] is True
