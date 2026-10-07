import json

from whichapiapi import core
from whichapiapi.adapters.newapi import NewApiAdapter
from whichapiapi.adapters.openrouter import OpenRouterAdapter
from whichapiapi.schema import LoadProfile
from whichapiapi.schema.naming import model_key
from whichapiapi.selector.constraints import Constraints, apply


def test_model_key():
    assert model_key("deepseek/deepseek-v4.1-flash") == "deepseek-v4.1-flash"
    assert model_key("claude-haiku-4-5-20251001") == "claude-haiku-4.5"
    assert model_key("claude-opus-5-5") == model_key("claude-opus-5.5") == "claude-opus-5.5"
    assert (
        model_key("claude-3-5-sonnet-20241022")
        == model_key("anthropic/claude-sonnet-3.5")
        == "claude-sonnet-3.5"
    )
    assert model_key("Claude 4.5 Opus") == model_key("Claude-Opus-4-5-20251101 (FC)") == "claude-opus-4.5"
    assert model_key("Gemini 3 Pro Preview") == "gemini-3-pro"
    assert model_key("claude-sonnet-4-20250514 (32k thinking)") == "claude-sonnet-4"
    assert model_key("gemma-3-4b-it") == "gemma-3-4b-it"  # 4b is a size, not a version
    assert model_key("deepseek-v4-0731") == "deepseek-v4-0731"
    assert model_key("gpt-4o-2024-08-06") == "gpt-4o"
    assert model_key("qwen3-235b-a22b-2507") == "qwen3-235b-a22b-2507"  # snapshot tags stay distinct


def test_openrouter_parse(fix):
    ad = OpenRouterAdapter()
    offers = list(ad.parse_models(json.loads((fix / "openrouter_models.json").read_text())))
    assert {o.model for o in offers} <= {"deepseek/deepseek-v4.1-flash", "openai/gpt-6-luna"}
    assert all(not o.model.startswith("~") for o in offers)
    ds = next(o for o in offers if o.model == "deepseek/deepseek-v4.1-flash")
    assert ds.channel_type == "aggregator" and ds.price.input_per_1m > 0 and "tools" in ds.features
    eps = list(ad.parse_endpoints(json.loads((fix / "openrouter_endpoints.json").read_text())))
    assert eps and all(e.group and e.reliability.uptime_30d is not None for e in eps)


def _seed(store, fix):
    pricing = json.loads((fix / "newapi_pricing.json").read_text())
    status = json.loads((fix / "newapi_status.json").read_text())["data"]
    store.upsert_offers(NewApiAdapter("https://api.reseller.test", "reseller").parse(pricing, status))
    store.upsert_offers(
        OpenRouterAdapter().parse_models(json.loads((fix / "openrouter_models.json").read_text()))
    )
    store.cache_put("ratio:reseller/deepseek-v4.1-flash", {"max": 0.5, "mean": 0.5, "n": 4})


def test_constraints(store, fix):
    _seed(store, fix)
    offers = store.find_offers(model="deepseek-v4.1-flash")
    kept, rejected = apply(offers, Constraints(exclude_user_channels=True))
    assert kept and all(k.offer.channel == "openrouter" for k in kept)
    assert rejected.get("user/BYO channel")
    kept, _ = apply(offers, Constraints(payment_method="crypto"))
    assert any(k.offer.channel == "reseller" for k in kept)  # Reseller lists "Crypto Pay"
    assert any(
        "payment methods unknown" in w for k in kept for w in k.warnings
    )  # OpenRouter: unknown, warned
    kept, _ = apply(offers, Constraints(payment_method="crypto", strict=True))
    assert all(k.offer.channel == "reseller" for k in kept)


def test_compare_prices_uses_learned_ratio(store, fix):
    _seed(store, fix)
    res = core.compare_prices(
        "DeepSeek-V4.1-Flash", LoadProfile(requests_per_month=1000), include_groups=True, store=store
    )
    assert res["model_key"] == "deepseek-v4.1-flash"
    chans = {o["channel"] for o in res["offers"]}
    assert {"reseller", "openrouter"} <= chans
    base = next(o for o in res["offers"] if o["channel"] == "reseller" and o["group"] is None)
    assert base["monthly_real_estimate"] == round(base["monthly_list"] * 0.5, 4)
    assert "risk" in base
    lp = core.learned_prices("reseller", store=store)
    assert lp[0]["channel_model"] == "reseller/deepseek-v4.1-flash"


def test_find_offers_sort_and_limit(store, fix):
    _seed(store, fix)
    res = core.find_offers(query="gpt-6-luna", limit=5, store=store)
    assert res["count"] >= 2 and len(res["offers"]) <= 5
    prices = [o["input_per_1m"] for o in res["offers"]]
    assert prices == sorted(prices)


def test_find_offers_sort_value_uses_benchmarks(store, fix):
    _seed(store, fix)
    core.pull_benchmarks(store=store, adapters=[_FakeLMArena()])
    res = core.find_offers(query="gpt-6-luna", sort="value", limit=5, store=store)
    assert res["count"] >= 2
    assert res["offers"][0]["quality_rating"] == 1250.0
    assert "lmarena/text/overall" in res["note"]
    coding = core.find_offers(query="gpt-6-luna", sort="value", benchmark="lmarena/text/coding", store=store)
    assert coding["offers"][0]["benchmark"]["score"] == 1300.0

    lb = core.leaderboard(store=store)
    assert lb[0]["model_key"] == "gpt-6-luna" and lb[0]["score"] == 1250.0


class _FakeLMArena:
    id, license, attribution, errors = "lmarena", "CC BY 4.0", "test", {}

    def fetch(self):
        from whichapiapi.schema.benchmark import BenchmarkScore

        def sc(board, name, score):
            return BenchmarkScore(
                source="lmarena", board=board, model_name=name, score=score, rank=30, votes=10
            )

        return [
            sc("text/overall", "gpt-6-luna", 1250.0),
            sc("text/overall", "GPT-6-Luna", 1200.0),  # same key, lower: dropped
            sc("text/coding", "gpt-6-luna", 1300.0),
        ]


def test_pull_benchmarks_writes_cache(store):
    res = core.pull_benchmarks(store=store, adapters=[_FakeLMArena()])
    assert res == {"lmarena": {"scores": 3, "models": 1, "boards": 2}}
    assert (
        core.model_benchmarks("gpt-6-luna", store=store)["boards"]["lmarena/text/coding"]["score"] == 1300.0
    )
    assert {b["benchmark"] for b in core.benchmark_boards(store=store)} == {
        "lmarena/text/overall",
        "lmarena/text/coding",
    }


class _FakeLiteLLM:
    id, license, attribution = "litellm", "MIT", "test"

    def fetch(self):
        return {
            "gpt-6-luna": {
                "batch_discount": 0.5,
                "off_peak": {"window": "00:00-06:00", "tz": "UTC", "discount": 0.3},
            }
        }


def test_pull_conditions_writes_cache(store):
    res = core.pull_conditions(store=store, adapter=_FakeLiteLLM())
    assert res == {"litellm": {"models": 1}}
    assert store.cache_get("conditions:gpt-6-luna")["batch_discount"] == 0.5

    cp = core.conditions_priors(store=store)
    assert cp[0]["model_key"] == "gpt-6-luna" and cp[0]["batch_discount"] == 0.5


def test_estimate_cost_applies_litellm_batch_discount(store, fix):
    # gpt-6-luna (from _seed's newapi fixture) has no conditions of its own — confirms the merge, not a
    # hand-built Conditions() like tests/test_cost.py's synthetic cases.
    _seed(store, fix)
    store.cache_put("conditions:gpt-6-luna", {"batch_discount": 0.4, "source": "litellm"})
    profile = LoadProfile(requests_per_month=1000, input_tokens=5000, output_tokens=1000, batch_share=1.0)
    res = core.estimate_cost("reseller/gpt-6-luna", profile=profile, store=store)
    b = res["breakdown"]
    assert b is not None and b["list_per_month"] > 0
    assert b["per_month"] < b["list_per_month"]
    assert any("batch" in n for n in b["notes"])
    assert res["offer"]["batch_discount"] == 0.4


class _FakeInferIndex:
    attribution = "test"

    def __init__(self, result):
        self._result = result

    def cheapest(self, model):
        return self._result


def test_cross_check_price_flags_large_delta(store, fix):
    _seed(store, fix)  # reseller/gpt-6-luna input_per_1m == 0.1
    res = core.cross_check_price(
        "gpt-6-luna", store=store, adapter=_FakeInferIndex({"input_per_1m": 0.5, "promo": False})
    )
    assert res["ours"]["input_per_1m"] == 0.1
    assert res["inferindex"]["input_per_1m"] == 0.5
    assert res["input_price_delta_vs_inferindex"] == round((0.1 - 0.5) / 0.5, 4)
    assert "differs from InferIndex's by more than 15%" in res["note"]


def test_cross_check_price_flags_promo(store, fix):
    _seed(store, fix)
    res = core.cross_check_price(
        "gpt-6-luna", store=store, adapter=_FakeInferIndex({"input_per_1m": 0.1, "promo": True})
    )
    assert "active promo" in res["note"]


def test_cross_check_price_no_match(store):
    res = core.cross_check_price("nonexistent-model", store=store, adapter=_FakeInferIndex(None))
    assert res["ours"] is None and res["inferindex"] is None


def test_ingest_price_change_notification_matches_channel(store, fix):
    _seed(store, fix)  # reseller offers have provenance.source == "https://api.reseller.test/api/pricing"
    rec = core.ingest_price_change_notification(
        {
            "watch_url": "https://api.reseller.test/api/pricing",
            "watch_uuid": "abc-123",
            "watch_title": "Reseller pricing",
            "diff_added": "+ new line",
            "diff_removed": "",
        },
        store=store,
    )
    assert rec["matched_channels"] == ["reseller"]
    assert rec["watch_title"] == "Reseller pricing"

    alerts = core.price_change_alerts(store=store)
    assert len(alerts) == 1 and alerts[0]["watch_url"] == "https://api.reseller.test/api/pricing"


def test_ingest_price_change_notification_unwraps_apprise_body(store):
    # some Apprise notification URL schemes nest the templated JSON as a string under "message"/"body"
    inner = json.dumps({"watch_url": "https://example.com/pricing", "watch_title": "Example"})
    rec = core.ingest_price_change_notification({"message": inner}, store=store)
    assert rec["watch_url"] == "https://example.com/pricing" and rec["watch_title"] == "Example"


def test_ingest_price_change_notification_no_domain_match(store):
    rec = core.ingest_price_change_notification({"watch_url": "https://unrelated.example/x"}, store=store)
    assert rec["matched_channels"] == []


async def test_mcp_tools(monkeypatch, store, fix):
    from fastmcp import Client

    from whichapiapi.surfaces import mcp as mcp_mod

    _seed(store, fix)
    monkeypatch.setattr(core, "_store", lambda s: store)
    async with Client(mcp_mod.mcp) as c:
        names = {t.name for t in await c.list_tools()}
        assert {"find_offers", "compare_prices", "estimate_cost", "run_eval", "eval_plan"} <= names
        r = await c.call_tool("compare_prices", {"model": "deepseek-v4.1-flash"})
        assert r.data["model_key"] == "deepseek-v4.1-flash"
        tax = await c.read_resource("whichapiapi://taxonomy")
        assert "llm.chat" in tax[0].text


def test_group_hints_and_usage_learning(store, fix):
    from whichapiapi.evaluator.balance import CallLogEntry
    from whichapiapi.evaluator.learn import learn_group_usage

    _seed(store, fix)
    res = core.compare_prices("deepseek-v4.1-flash", store=store)
    base = next(o for o in res["offers"] if o["channel"] == "reseller")
    hint = base["cheapest_group"]
    assert hint["group"] and "caveat" in hint and "observed_share" not in hint
    assert hint["input_per_1m"] <= base["input_per_1m"]

    def entry(group):
        return CallLogEntry(
            ts=1, model="deepseek-v4.1-flash", cost_usd=0.001, prompt_tokens=1, completion_tokens=1,
            group=group, retries=1 if group == "other" else 0,
        )  # fmt: skip

    learn_group_usage(store, [entry("other")] * 3 + [entry(None)], "reseller")
    usage = core.group_usage("reseller", store=store)
    assert usage[0]["n"] == 4 and usage[0]["groups"]["other"]["share"] == 0.75
    assert usage[0]["retried_share"] == 0.75

    res = core.compare_prices("deepseek-v4.1-flash", store=store, include_groups=True)
    base = next(o for o in res["offers"] if o["channel"] == "reseller" and o["group"] is None)
    assert base["cheapest_group"]["observed_share"] == 0.0
    assert base["cheapest_group"]["served_by"][0]["group"] == "other"
    grouped = next(o for o in res["offers"] if o["group"] == hint["group"])
    assert grouped["observed_share"] == 0.0


def test_repull_refreshes_verified_at_without_new_version(store):
    from datetime import UTC, datetime, timedelta

    from whichapiapi.schema.offer import Offer, Provenance

    old = datetime.now(UTC) - timedelta(days=30)
    o = Offer(model="m", channel="c", provenance=Provenance(source="s", verified_at=old))
    assert store.upsert_offers([o]) == (1, 0)
    again = o.model_copy(update={"provenance": Provenance(source="s")})
    assert store.upsert_offers([again]) == (0, 0)  # unchanged: no version row
    assert datetime.now(UTC) - store.get_offer("c/m").provenance.verified_at < timedelta(minutes=1)


def test_learn_from_log_does_not_double_count_overlapping_windows(store):
    from whichapiapi.evaluator.balance import CallLogEntry
    from whichapiapi.evaluator.learn import learn_from_log

    class Probe:
        def __init__(self, entries):
            self.entries = entries

        def calls_since(self, ts, model=None, max_pages=20):
            return self.entries

    def e(ts, ratio):
        return CallLogEntry(
            ts=ts, model="m", cost_usd=0.1, prompt_tokens=1, completion_tokens=1, group_ratio=ratio
        )

    learn_from_log(store, Probe([e(1, 0.5), e(2, 0.3)]), "c")
    val = learn_from_log(store, Probe([e(2, 0.3), e(3, 0.1)]), "c")["m"]
    assert val["n"] == 3 and val["max"] == 0.5 and val["last_ts"] == 3


def test_value_sort_lower_is_better_in_capability_unit(store):
    from whichapiapi.schema.benchmark import BenchmarkScore
    from whichapiapi.schema.offer import Offer, Price, Provenance

    def stt(model, per_min=None, per_1m=None):
        price = Price(per_unit=per_min, unit="minute") if per_min else Price(input_per_1m=per_1m)
        return Offer(
            capability="speech.stt", model=model, channel="c", price=price, provenance=Provenance(source="t")
        )

    store.upsert_offers(
        [
            stt("whisper-large-v3", 0.00045),
            stt("whisper-large-v3-turbo", 0.0002),
            stt("x/whisper-large-v3", None, 0.1),
        ]
    )

    class FakeASR:
        id, license, attribution, errors = "open_asr", "t", "t", {}

        def fetch(self):
            return [
                BenchmarkScore(
                    source="open_asr",
                    board="avg_wer",
                    model_name=m,
                    score=w,
                    unit="wer",
                    higher_is_better=False,
                )
                for m, w in (("openai/whisper-large-v3", 5.8), ("openai/whisper-large-v3-turbo", 6.4))
            ]

    core.pull_benchmarks(store=store, adapters=[FakeASR()])
    res = core.find_offers(capability="speech.stt", sort="value", benchmark="open_asr/avg_wer", store=store)
    # WER x $/min: turbo 6.4*0.0002 < 5.8*0.00045; the token-priced offer isn't comparable and goes last
    assert [o["offer_id"] for o in res["offers"]] == [
        "c/whisper-large-v3-turbo",
        "c/whisper-large-v3",
        "c/x/whisper-large-v3",
    ]


def test_failed_benchmark_pull_keeps_previous_scores(store):
    from whichapiapi.schema.benchmark import BenchmarkScore

    class Arena:
        id, license, attribution = "lmarena", "t", "t"

        def __init__(self, scores, errors=None):
            self.scores, self.errors = scores, errors or {}

        def fetch(self):
            return self.scores

    def sc(board, name, score):
        return BenchmarkScore(source="lmarena", board=board, model_name=name, score=score)

    core.pull_benchmarks(
        store=store, adapters=[Arena([sc("text/overall", "a-1", 1300), sc("vision/overall", "a-1", 1200)])]
    )
    core.pull_benchmarks(store=store, adapters=[Arena([], {"text": "429"})])  # everything rate-limited
    assert set(core.model_benchmarks("a-1", store=store)["boards"]) == {
        "lmarena/text/overall",
        "lmarena/vision/overall",
    }
    # vision fetched again without the model: only that board goes, text (failed this time) stays
    core.pull_benchmarks(store=store, adapters=[Arena([sc("vision/overall", "b-2", 1100)], {"text": "429"})])
    assert set(core.model_benchmarks("a-1", store=store)["boards"]) == {"lmarena/text/overall"}


def test_suggest_candidates_only_reachable_and_one_per_creator(store, monkeypatch):
    from whichapiapi.schema.offer import Offer, Price, Provenance

    monkeypatch.setattr(core, "model_cards", lambda st=None: [
        {"id": "a-1", "name": "A 1", "creator": "Acme", "price_in": 1, "price_out": 1, "think": 0, "tps": 100,
         "ttft": 0.5, "ttfa": 0.5, "q": {"intelligence": 95, "arena_overall": 95, "arena_hard": 95}},
        {"id": "a-2", "name": "A 2", "creator": "Acme", "price_in": 1, "price_out": 1, "think": 0, "tps": 100,
         "ttft": 0.5, "ttfa": 0.5, "q": {"intelligence": 90, "arena_overall": 90, "arena_hard": 90}},
        {"id": "b-1", "name": "B 1", "creator": "Beta", "price_in": 1, "price_out": 1, "think": 0, "tps": 100,
         "ttft": 0.5, "ttfa": 0.5, "q": {"intelligence": 80, "arena_overall": 80, "arena_hard": 80}},
        {"id": "c-1", "name": "C 1", "creator": "Gamma", "price_in": 1, "price_out": 1, "think": 0, "tps": 100,
         "ttft": 0.5, "ttfa": 0.5, "q": {"intelligence": 99, "arena_overall": 99, "arena_hard": 99}},
    ])  # fmt: skip

    def offer(ch, model):
        return Offer(
            model=model,
            channel=ch,
            price=Price(input_per_1m=1, output_per_1m=1),
            provenance=Provenance(source="t"),
        )

    store.upsert_offers(
        [
            offer("reseller", "a-1"),
            offer("reseller", "a-2"),
            offer("openrouter", "b/b-1"),
            offer("elsewhere", "c-1"),
        ]
    )
    got = core.suggest_candidates(channels=["reseller", "openrouter"], store=store)
    assert got == [
        "reseller:a-1",
        "openrouter:b/b-1",
    ]  # c-1 isn't callable with these keys; one Acme model only

    # a pair that fails on real traffic goes last; the same creator's next model takes its place
    import json as _json
    import os as _os
    import tempfile

    store_dir = tempfile.mkdtemp()
    log = _os.path.join(store_dir, "usage.jsonl")
    with open(log, "w") as f:
        for i in range(6):
            f.write(_json.dumps({"ts": "2099-01-01T00:00:00Z", "type": "call", "provider": "reseller", "model": "a-1",
                                 "status": "ok" if i == 0 else "timeout", "ms": 1000}) + "\n")  # fmt: skip
    monkeypatch.setenv("WHICHAPIAPI_FIELD_LOGS", log)
    got = core.suggest_candidates(channels=["reseller", "openrouter"], n=3, store=store)
    assert got == ["reseller:a-2", "openrouter:b/b-1", "reseller:a-1"]
    scored = core.suggest_candidates(
        channels=["reseller"], n=1, store=store, compare=["reseller:a-1", "x:nope"]
    )
    top, (cmp_a1, cmp_none) = scored["candidates"][0], scored["compare"]
    assert top["candidate"] == "reseller:a-2" and top["cost_per_task"] > 0 and "quality" in top["why"]
    assert cmp_a1["field"]["ok_rate"] == round(1 / 6, 3) and cmp_a1["score"] is not None
    assert cmp_none["why"] == "no benchmark data or not sold on your channels"

    # the channel's real price (learned mean ratio) is what ranks, and the age rule drops old releases
    store.cache_put("ratio:reseller/a-2", {"max": 0.2, "mean": 0.1, "n": 9})
    cheap = core.suggest_candidates(channels=["reseller"], n=1, store=store, with_scores=True)["candidates"][
        0
    ]
    assert cheap["price_source"] == "measured ×0.1 of list" and cheap["cost_per_task"] < top["cost_per_task"]
    monkeypatch.setattr(core, "model_cards", lambda st=None: [
        {"id": "b-1", "name": "B 1", "creator": "Beta", "released": "2020-01-01", "price_in": 1, "price_out": 1,
         "think": 0, "tps": 100, "ttft": 0.5, "ttfa": 0.5, "q": {"intelligence": 80}}])  # fmt: skip
    assert core.suggest_candidates(channels=["openrouter"], store=store) == []
    assert core.suggest_candidates(channels=["openrouter"], store=store, max_age_months=None) == [
        "openrouter:b/b-1"
    ]
