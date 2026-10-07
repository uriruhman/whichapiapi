"""Public façade (ARCHITECTURE §2). Every surface — CLI, MCP, REST — calls these functions and nothing deeper.

All functions return plain JSON-serialisable data so surfaces stay thin.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from whichapiapi import channels
from whichapiapi.adapters.artificial_analysis import KEY_ENV as AA_KEY_ENV
from whichapiapi.adapters.artificial_analysis import ArtificialAnalysisAdapter
from whichapiapi.adapters.bfcl import BFCLAdapter
from whichapiapi.adapters.deepinfra import DeepInfraAdapter
from whichapiapi.adapters.edenai import EdenAIAdapter
from whichapiapi.adapters.freellmapi import FreeLLMAPIAdapter
from whichapiapi.adapters.hf_router import HFRouterAdapter
from whichapiapi.adapters.inferindex import InferIndexAdapter
from whichapiapi.adapters.litellm_conditions import LiteLLMConditionsAdapter
from whichapiapi.adapters.lmarena import LMArenaAdapter
from whichapiapi.adapters.models_dev import ModelsDevAdapter
from whichapiapi.adapters.newapi import NewApiAdapter
from whichapiapi.adapters.open_asr import OpenASRAdapter
from whichapiapi.adapters.openrouter import OpenRouterAdapter
from whichapiapi.adapters.openrouter_usage import OpenRouterUsageAdapter
from whichapiapi.adapters.swebench import SWEBenchAdapter
from whichapiapi.cost.engine import monthly_cost
from whichapiapi.evaluator.canary import compare_to_baseline, new_baselines
from whichapiapi.evaluator.runner import EvalRunner
from whichapiapi.report.render import to_markdown
from whichapiapi.schema.capability import TAXONOMY
from whichapiapi.schema.load_profile import LoadProfile
from whichapiapi.schema.naming import model_key
from whichapiapi.schema.offer import Offer, OffPeak
from whichapiapi.schema.suite import load_suite
from whichapiapi.selector.cards import COMPONENTS, PRESETS, PROFILES, TASKS, build_cards
from whichapiapi.selector.cards import rank as rank_cards
from whichapiapi.selector.constraints import Constraints, apply
from whichapiapi.selector.pareto import recommend
from whichapiapi.store.db import Store

STALE_AFTER = timedelta(days=14)

ATTRIBUTION = {
    "models_dev": "models.dev (MIT)",
    "openrouter": "OpenRouter public API",
    "newapi": "reseller's public /api/pricing",
    "litellm": "LiteLLM model_prices_and_context_window.json (MIT, HTTP data only, ADR-0003)",
}


def _store(store: Store | None) -> Store:
    return store or Store()


def _is_zero_priced(o: Offer) -> bool:
    """0/0 prices mean a free tier or access bundled into a subscription plan — not comparable per token."""
    p = o.price
    return not (p.input_per_1m or p.output_per_1m or p.per_request or p.per_unit)


def learned_ratio(store: Store, channel: str, model: str) -> dict[str, Any] | None:
    return store.cache_get(f"ratio:{channel}/{model}")


def group_usage(channel: str | None = None, store: Store | None = None) -> list[dict[str, Any]]:
    """Which reseller price group actually answered the key's calls (from `eval learn`), per model."""
    items = _store(store).cache_items(f"groups:{channel}/" if channel else "groups:")
    return sorted(
        ({"channel_model": k.removeprefix("groups:"), **v} for k, v in items.items()),
        key=lambda d: d["channel_model"],
    )


def _group_hints(offers: list[Offer], views: list[dict[str, Any]], store: Store) -> None:
    """Annotate reseller price-group offers with how often the key was really served by that group, and give the
    ungrouped view the cheapest group as an unverified best case (list price of a group != what you pay)."""
    grouped = [o for o in offers if o.group and o.price.is_known and not _is_zero_priced(o)]
    for v in views:
        stats = store.cache_get(f"groups:{v['channel']}/{v['model']}")
        if v["group"] is not None:
            if stats:
                g = stats["groups"].get(v["group"])
                v["observed_calls"] = g["n"] if g else 0
                v["observed_share"] = g["share"] if g else 0.0
            continue
        cands = [o for o in grouped if o.channel == v["channel"] and o.model == v["model"]]
        if not cands:
            continue
        best = min(cands, key=lambda o: (o.price.input_per_1m or 0) + (o.price.output_per_1m or 0))
        hint: dict[str, Any] = {
            "group": best.group,
            "input_per_1m": best.price.input_per_1m,
            "output_per_1m": best.price.output_per_1m,
            "caveat": "best case from the price list; the key's routing may never reach this group",
        }
        if stats:
            g = stats["groups"].get(best.group)
            hint["observed_share"] = g["share"] if g else 0.0
            hint["observed_calls_total"] = stats["n"]
            hint["served_by"] = sorted(
                ({"group": k, **x} for k, x in stats["groups"].items()), key=lambda x: -x["share"]
            )[:3]
        v["cheapest_group"] = hint


def _with_conditions(o: Offer, store: Store) -> Offer:
    """Fill in `Conditions` fields the offer's own adapter left unset from cached LiteLLM-derived priors
    (adapter-populated conditions always win). This is what makes `monthly_cost`'s batch/off-peak discount
    toggles apply to real data instead of only to hand-built test offers."""
    cached = conditions_prior(store, o.model)
    if not cached:
        return o
    c = o.conditions
    updates: dict[str, Any] = {}
    if c.batch_discount is None and cached.get("batch_discount") is not None:
        updates["batch_discount"] = cached["batch_discount"]
    if c.off_peak is None and cached.get("off_peak") is not None:
        updates["off_peak"] = OffPeak(**cached["off_peak"])
    if not updates:
        return o
    return o.model_copy(update={"conditions": c.model_copy(update=updates)})


def offer_view(o: Offer, store: Store, profile: LoadProfile | None = None) -> dict[str, Any]:
    """Compact, agent-friendly view of one offer, with real-price hints and staleness."""
    o = _with_conditions(o, store)
    p = o.price
    verified = o.provenance.verified_at
    if verified.tzinfo is None:
        verified = verified.replace(tzinfo=UTC)
    stale = datetime.now(UTC) - verified > STALE_AFTER
    view: dict[str, Any] = {
        "offer_id": o.id,
        "model": o.model,
        "channel": o.channel,
        "channel_type": o.channel_type,
        "group": o.group,
        "input_per_1m": p.input_per_1m,
        "output_per_1m": p.output_per_1m,
        "cache_read_per_1m": p.cache_read_per_1m,
        "per_request": p.per_request,
        "currency": p.currency,
        "context_window": o.context_window,
        "features": o.features,
        "uptime": o.reliability.uptime_30d,
        "source": o.provenance.source,
        "confidence": o.provenance.confidence,
        "verified_at": verified.date().isoformat(),
        "stale": stale,
    }
    if p.per_unit is not None:
        view["per_unit"], view["unit"] = p.per_unit, p.unit
    if o.reliability.ttft_ms is not None or o.reliability.throughput_tps is not None:
        view["ttft_ms"], view["throughput_tps"] = o.reliability.ttft_ms, o.reliability.throughput_tps
    if o.provenance.note:
        view["source_note"] = o.provenance.note
    if o.conditions.batch_discount is not None:
        view["batch_discount"] = o.conditions.batch_discount
    if o.conditions.off_peak is not None:
        view["off_peak_discount"] = o.conditions.off_peak.discount
    if o.conditions.free_tier is not None:
        view["free_tier"] = o.conditions.free_tier.model_dump(exclude_none=True, exclude_defaults=True)
    if o.visibility == "user":
        view["risk"] = (
            "user/BYO channel: unverified reseller — retention policy, SLA and key stability unknown"
        )
    ratio = learned_ratio(store, o.channel, o.model) if o.group is None else None
    if ratio:
        view["real_price_multiplier"] = {"mean": ratio["mean"], "max": ratio["max"], "calls_seen": ratio["n"]}
    if profile:
        b = monthly_cost(o, profile)
        if b:
            view["monthly_list"] = b.per_month
            if ratio:
                view["monthly_real_estimate"] = round(b.per_month * ratio["mean"], 4)
            if b.notes:
                view["cost_notes"] = b.notes
    return view


# ---------------------------------------------------------------- catalog


# Public catalogs that need no key. Dedicated adapters for aggregators (live prices, non-LLM capabilities) plus
# models.dev for the long tail; models.dev skips the channels these cover (adapters/models_dev.FIRST_PARTY_ADAPTERS).
_SIMPLE_CATALOGS = {
    "deepinfra": DeepInfraAdapter,
    "huggingface": HFRouterAdapter,
    "edenai": EdenAIAdapter,
    "freellmapi": FreeLLMAPIAdapter,  # free tiers with limits (zero-priced, hidden unless include_free)
}
CATALOGS = ["models-dev", "openrouter", *_SIMPLE_CATALOGS]


def pull_offers(
    sources: list[str] | None = None,
    newapi: list[dict[str, str]] | None = None,
    openrouter_endpoints_for: list[str] | None = None,
    store: Store | None = None,
) -> dict[str, Any]:
    """Refresh offers from public sources. `newapi`: [{"url", "channel", "key_env"?}] for reseller channels."""
    st = _store(store)
    out: dict[str, Any] = {}
    adapters: list[tuple[str, Any]] = []
    for s in sources or CATALOGS:
        if s == "models-dev":
            adapters.append((s, ModelsDevAdapter()))
        elif s == "openrouter":
            adapters.append((s, OpenRouterAdapter(endpoints_for=openrouter_endpoints_for)))
        elif s in _SIMPLE_CATALOGS:
            adapters.append((s, _SIMPLE_CATALOGS[s]()))
    for n in newapi or []:
        adapters.append(
            (f"newapi:{n['channel']}", NewApiAdapter(n["url"], n["channel"], key_env=n.get("key_env")))
        )
    for name, fetched in zip(
        (n for n, _ in adapters), _fetch_all([ad for _, ad in adapters]), strict=True
    ):  # one broken source must not block the others
        if isinstance(fetched, Exception):
            out[name] = {"error": f"{type(fetched).__name__}: {str(fetched)[:200]}"}
            continue
        new, changed = st.upsert_offers(fetched)
        out[name] = {"offers": len(fetched), "new": new, "changed": changed}
    return out


def _fetch_all(adapters: list[Any]) -> list[Any]:
    """`list(ad.fetch())` for every adapter, concurrently (independent HTTP sources); exceptions are returned."""

    def one(ad: Any) -> Any:
        try:
            return list(ad.fetch())
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=max(1, len(adapters))) as pool:
        return list(pool.map(one, adapters))


DEFAULT_BENCHMARK = "lmarena/text/overall"


def _benchmark_adapters() -> list[Any]:
    adapters: list[Any] = [
        LMArenaAdapter(), OpenASRAdapter(), SWEBenchAdapter(), BFCLAdapter(), OpenRouterUsageAdapter(),
    ]  # fmt: skip
    if os.environ.get(AA_KEY_ENV):  # needs a (free) key; skipped silently without one
        adapters.append(ArtificialAnalysisAdapter())
    return adapters


def pull_benchmarks(store: Store | None = None, adapters: list[Any] | None = None) -> dict[str, Any]:
    """Refresh third-party benchmark scores (LMArena arenas and categories, ...) into `bench:{model_key}`.
    Each source replaces only its own boards. Powers `find_offers(sort="value", benchmark=...)` and
    `model_benchmarks`."""
    st = _store(store)
    out: dict[str, Any] = {}
    adapters = adapters or _benchmark_adapters()
    for ad, scores in zip(adapters, _fetch_all(adapters), strict=True):
        if isinstance(scores, Exception):  # mirror pull_offers: one broken source must not raise
            out[ad.id] = {"error": f"{type(scores).__name__}: {str(scores)[:200]}"}
            continue
        by_model: dict[str, dict[str, Any]] = {}
        for sc in scores:
            boards = by_model.setdefault(model_key(sc.model_name), {})
            prev = boards.get(sc.id)
            better = prev is None or (
                sc.score > prev["score"] if sc.higher_is_better else sc.score < prev["score"]
            )
            if better:  # names that collapse to one key: keep the best variant
                boards[sc.id] = sc.model_dump(exclude={"source", "board"})
        board_counts: dict[str, int] = {}
        for sc in scores:
            board_counts[sc.id] = board_counts.get(sc.id, 0) + 1
        # Replace exactly the boards fetched in this pull: a model a board no longer lists loses it; boards of an
        # arena that failed this time (429, renamed config) keep their previous scores.
        for old_key, boards in st.cache_items("bench:").items():
            key = old_key.removeprefix("bench:")
            if key not in by_model and not board_counts.keys().isdisjoint(boards):
                st.cache_put(old_key, {k: v for k, v in boards.items() if k not in board_counts})
        for key, boards in by_model.items():
            cur = {k: v for k, v in (st.cache_get(f"bench:{key}") or {}).items() if k not in board_counts}
            st.cache_put(f"bench:{key}", cur | boards)
        prev_meta = st.cache_get(f"benchsrc:{ad.id}") or {}
        st.cache_put(
            f"benchsrc:{ad.id}",
            {
                "license": ad.license,
                "attribution": ad.attribution,
                "boards": (prev_meta.get("boards") or {}) | board_counts,
                "pulled_at": datetime.now(UTC).isoformat(timespec="seconds")
                if board_counts
                else prev_meta.get("pulled_at"),
            },
        )
        if raw := getattr(ad, "raw", None):  # per-variant source rows, for model cards
            st.cache_put(f"benchraw:{ad.id}", raw)
        out[ad.id] = {"scores": len(scores), "models": len(by_model), "boards": len(board_counts)}
        if errors := getattr(ad, "errors", None):
            out[ad.id]["errors"] = errors
    return out


# Leaderboards list reasoning-effort variants ("deepseek-v4.1-flash-max"); catalogs sell the bare model. Without an
# exact match, take the variant closest to a default setting, in this order, and say which one was used.
_VARIANT_SUFFIXES = ("", "-medium", "-high", "-thinking", "-reasoning", "-max", "-xhigh", "-low", "-minimal")


def _bench_index(store: Store) -> dict[str, dict[str, Any]]:
    return {k.removeprefix("bench:"): v for k, v in store.cache_items("bench:").items()}


def _resolve_bench(index: dict[str, dict[str, Any]] | Store, model: str) -> tuple[str | None, dict[str, Any]]:
    """Merged boards for a model: `index` is a preloaded `_bench_index` (many lookups) or the store (one model:
    reads only the candidate keys)."""
    get = index.get if isinstance(index, dict) else (lambda k: index.cache_get(f"bench:{k}"))
    key = model_key(model)
    if "-" not in key and not any(c.isdigit() for c in key):
        return None, {}  # "enhanced", "default": generic tier names, not model ids — would match anything
    first, merged = None, {}
    for suffix in _VARIANT_SUFFIXES:  # exact name first; variants fill boards the exact name lacks
        for board, b in (get(key + suffix) or {}).items():
            if board not in merged:
                merged[board] = {**b, "matched_model": key + suffix}
                first = first or key + suffix
    return first, merged


def benchmark_score(store: Store, model: str, benchmark: str = DEFAULT_BENCHMARK) -> dict[str, Any] | None:
    _, boards = _resolve_bench(store, model)
    return boards.get(benchmark)


def model_benchmarks(model: str, store: Store | None = None) -> dict[str, Any]:
    """Every benchmark score we hold for a model (all sources, boards), with attribution. Falls back to the
    leaderboard's reasoning-effort variant (`matched_model`) when the bare name isn't listed."""
    st = _store(store)
    key = model_key(model)
    matched, boards = _resolve_bench(st, model)
    sources = {b.split("/", 1)[0] for b in boards}
    return {
        "model_key": key,
        "matched_model": matched,
        "boards": dict(sorted(boards.items())),
        "attribution": [(st.cache_get(f"benchsrc:{s}") or {}).get("attribution") for s in sorted(sources)],
        "note": "Scores come from public leaderboards on their own tasks (Elo = human preference votes). They are "
        "priors, not a verdict on your task: validate with run_eval on your own examples.",
    }


def benchmark_boards(store: Store | None = None) -> list[dict[str, Any]]:
    """Available benchmark boards (source/board) with how many models each covers."""
    st = _store(store)
    out = []
    for meta in st.cache_items("benchsrc:").values():
        for board, n in sorted(meta.get("boards", {}).items()):
            out.append(
                {
                    "benchmark": board,
                    "models": n,
                    "license": meta.get("license"),
                    "pulled_at": meta.get("pulled_at"),
                }
            )
    return out


def leaderboard(
    benchmark: str = DEFAULT_BENCHMARK, limit: int = 40, store: Store | None = None
) -> list[dict[str, Any]]:
    """Models on one benchmark board, best first."""
    rows = [
        {"model_key": k.removeprefix("bench:"), **boards[benchmark]}
        for k, boards in _store(store).cache_items("bench:").items()
        if benchmark in boards
    ]
    higher = rows[0].get("higher_is_better", True) if rows else True
    return sorted(rows, key=lambda r: -r["score"] if higher else r["score"])[:limit]


# channels trusted enough to quote as "where to buy" in model cards: first-party catalogs, official vendor APIs and
# the user's own channels (small resellers known only from models.dev are left out of this one number)
_BUY_CHANNELS = {
    "openrouter", "deepinfra", "huggingface", "vercel",  # first-party catalogs
    "openai", "anthropic", "google", "deepseek", "mistral", "xai", "zai", "moonshotai", "alibaba", "minimax",
    "xiaomi", "groq", "cerebras", "togetherai", "fireworks-ai", "novita-ai", "nebius", "siliconflow", "amazon-bedrock",
    "azure", "google-vertex",  # vendors' own APIs and large, well-known inference providers
}  # fmt: skip


def model_cards(store: Store | None = None) -> list[dict[str, Any]]:
    """One card per LLM variant (Artificial Analysis rows, needs `pull_benchmarks` with its key) with benchmark
    components, measured speed, estimated thinking tokens and the cheapest trusted offer. See selector/cards.py."""
    st = _store(store)
    raw = st.cache_get("benchraw:artificial_analysis") or {}
    index = _bench_index(st)
    best: dict[str, dict[str, Any]] = {}
    vision: set[str] = set()
    for o in st.find_offers(capability="llm.chat", limit=50000):
        key = model_key(o.model)
        if "vision" in o.features:
            vision.add(key)
        p = o.price
        if o.group or p.input_per_1m is None or p.output_per_1m is None or _is_zero_priced(o):
            continue
        if not (o.channel in _BUY_CHANNELS or o.visibility == "user") or o.model.endswith(
            (":batch", ":free")
        ):
            continue
        total = p.input_per_1m + p.output_per_1m
        if key not in best or total < best[key]["_t"]:
            best[key] = {
                "offer_id": o.id,
                "input_per_1m": p.input_per_1m,
                "output_per_1m": p.output_per_1m,
                "_t": total,
            }

    def boards_for(name: str) -> dict[str, Any]:
        return _resolve_bench(index, name)[1]

    def cheapest(name: str) -> dict[str, Any] | None:
        b = best.get(model_key(name))
        return {k: v for k, v in b.items() if k != "_t"} if b else None

    def license_of(name: str) -> str | None:
        return next((b.get("license") for b in boards_for(name).values() if b.get("license")), None)

    return build_cards(
        raw.get("data") or [],
        boards_for,
        cheapest,
        has_vision=lambda n: model_key(n) in vision or "lmarena/vision/overall" in boards_for(n),
        license_of=license_of,
    )


def rank_models(
    preset: str | None = "optimal",
    task: str | None = None,
    weights: tuple[float, float, float] | None = None,
    profile: str | tuple[int, int] | None = None,
    limit: int = 10,
    store: Store | None = None,
    **filters: Any,
) -> dict[str, Any]:
    """Best model variants for a task: a preset (see selector.cards.PRESETS) optionally overridden by task, weights
    (quality, cost, speed), a token profile name or (in, out) tokens, and filters (min_quality, min_tps, max_ttfa,
    max_cost, vision, open_only)."""
    p = dict(PRESETS.get(preset or "", {}))
    if task and task not in TASKS:
        raise ValueError(
            f"unknown task {task!r}; one of {sorted(TASKS)}. Media models (image/video/TTS/STT) are not ranked here: "
            "use find_offers(capability=..., sort='value')"
        )
    prof = profile or p.get("profile") or "chat"
    if isinstance(prof, str) and prof not in PROFILES:
        raise ValueError(f"unknown profile {prof!r}; one of {sorted(PROFILES)}")
    tokens = PROFILES[prof] if isinstance(prof, str) else tuple(prof)
    opts = {
        k: p[k] for k in ("min_quality", "min_tps", "max_ttfa", "vision", "open_only") if k in p
    } | filters
    opts.setdefault("released_after", released_after(max_model_age_months()))
    rows = rank_cards(
        model_cards(store),
        task or p.get("task", "general"),
        weights or p.get("w", (0.55, 0.3, 0.15)),
        tokens,
        limit=limit,
        **opts,
    )
    st = _store(store)
    or_ids = {
        model_key(o.model): o.model
        for o in st.find_offers(channel="openrouter", limit=5000)
        if o.group is None and not o.model.endswith((":free", ":batch")) and not o.model.startswith("~")
    }
    picks = []  # primary + next best from another creator, both on OpenRouter: ready for its `models` fallback list
    for r in rows:
        oid = or_ids.get(model_key(r["card"]["id"]))
        if oid and all(r["card"].get("creator") != c for _, c in picks):
            picks.append((oid, r["card"].get("creator")))
        if len(picks) == 2:
            break
    return {
        "preset": preset,
        "openrouter_models": [oid for oid, _ in picks],
        "task": task or p.get("task", "general"),
        "profile": {"in_tokens": tokens[0], "out_tokens": tokens[1]},
        "models": [
            {
                "model": r["card"]["name"],
                "id": r["card"]["id"],
                "score": r["score"],
                "quality": r["quality"],
                "coverage": r["coverage"],
                "cost_per_task": round(r["cost"], 6),
                "cost_per_1k_tasks": round(1000 * r["cost"], 4),
                "seconds_per_task": round(r["seconds"], 2) if r["seconds"] else None,
                "thinking_tokens": r["card"].get("think"),
                "tokens_per_second": r["card"].get("tps"),
                "buy": r["card"].get("buy"),
            }
            for r in rows
        ],
        "note": "Quality blends benchmark percentiles for the task (Artificial Analysis — attribution required, "
        "LMArena CC BY 4.0, SWE-bench, BFCL); cost per task includes estimated thinking tokens. Validate the top "
        "picks with run_eval on your own examples.",
    }


def max_model_age_months() -> float | None:
    """Owner rule (2026-09-30): don't suggest models released more than N months ago (default 6; 0 = no limit)."""
    v = float(os.environ.get("WHICHAPIAPI_MAX_MODEL_AGE_MONTHS", "6") or 0)
    return v or None


def released_after(months: float | None) -> str | None:
    return (datetime.now(UTC) - timedelta(days=30.44 * months)).date().isoformat() if months else None


def _channel_price(offer: Offer, st: Store) -> tuple[float, float, str]:
    """(input, output USD per 1M, source) that this channel really charges: its list price × the mean real/list ratio
    learned from its own call log (resellers discount models very differently); list price when never measured."""
    p = offer.price
    ratio = (st.cache_get(f"ratio:{offer.channel}/{offer.model}") or {}).get("mean")
    if ratio:
        return p.input_per_1m * ratio, p.output_per_1m * ratio, f"measured ×{ratio:.3g} of list"
    return p.input_per_1m, p.output_per_1m, "list (not measured yet)"


def reachable_channels() -> list[str]:
    """Channels the user can call (a candidate is only suggested where it can really be called): custom and built-in
    channels whose key env variable is set (whichapiapi.channels), then providers in the key pool."""
    from whichapiapi import keypool

    env = channels.with_key()
    return env + sorted(keypool.providers() - set(env))


def free_candidates(task: str = "general", n: int = 3, store: Store | None = None) -> list[str]:
    """The best models for `task` that a free tier offers on a provider you have a pool key for, as
    "provider:model" (one per provider, ranked by benchmark quality — they cost nothing until the quota runs out;
    the router then cools that route until the quota resets and moves on)."""
    from whichapiapi import keypool

    st = _store(store)
    pool = keypool.providers()
    free: dict[str, Offer] = {}
    for prov in pool:
        for o in st.find_offers(channel=prov, capability="llm.chat", limit=5000):
            if o.group == "free":
                free.setdefault(model_key(o.model), o)
    cards = [{**c, "price_in": 0.0, "price_out": 0.0, "buy": None} for c in model_cards(st)
             if model_key(c["id"]) in free]  # fmt: skip
    ranked = rank_cards(cards, task, (1.0, 0.0, 0.0), PROFILES["chat"], limit=200)
    out, used = [], set()
    for r in ranked:
        o = free[model_key(r["card"]["id"])]
        if o.channel in used:
            continue
        out.append(f"{o.channel}:{o.endpoint.model_id or o.model}")
        used.add(o.channel)
        if len(out) >= n:
            break
    return out


def suggest_candidates(
    task: str = "general",
    n: int = 4,
    preset: str = "optimal",
    channels: list[str] | None = None,
    store: Store | None = None,
    with_scores: bool = False,
    compare: list[str] | None = None,
    max_age_months: float | None = -1,
) -> Any:
    """Top model variants for a task that are callable with the user's own keys, as "channel:model" ids ready for a
    suite — at most one per creator so the comparison is broad (Optima's "run it on the leading models", limited to
    what you can buy). Cost per task uses what the channel really charges (learned real/list ratio), not the
    cheapest price elsewhere. Models released more than `max_age_months` ago are skipped (default: the owner's
    rule, 6; None = no limit). Pairs that fail often on the owner's real traffic (field logs) go to the end.

    `with_scores` or `compare` → {"candidates": [rows], "compare": [rows]} where a row is {candidate, variant,
    released, quality, coverage, task_coverage, cost_per_task, price_source, seconds_per_task, score, why, field};
    `compare` scores the given ids (e.g. the chain in use) on the same scale."""
    from whichapiapi.evaluator import field as field_log
    from whichapiapi.selector.cards import task_coverage

    st = _store(store)
    usable = channels or reachable_channels()
    sold: dict[str, dict[str, Offer]] = {}
    for ch in usable:
        for o in st.find_offers(channel=ch, capability="llm.chat", limit=20000):
            if o.group is None and o.price.is_known and not _is_zero_priced(o):
                sold.setdefault(model_key(o.model), {}).setdefault(ch, o)
    months = max_model_age_months() if max_age_months == -1 else max_age_months
    after = released_after(months)
    # one card per variant sold on a reachable channel, priced at what that channel really charges
    cards, where_of = [], {}
    for c in model_cards(st):
        where = sold.get(model_key(c["id"])) or {}
        ch = next((x for x in usable if x in where), None)
        if ch:
            pin, pout, source = _channel_price(where[ch], st)
            cards.append({**c, "price_in": pin, "price_out": pout, "buy": None, "price_source": source})
            where_of[c["id"]] = f"{ch}:{where[ch].model}"
    p = PRESETS.get(preset, PRESETS["optimal"])
    tokens = PROFILES[p.get("profile") or "chat"]
    filters = {k: p[k] for k in ("min_quality", "min_tps", "max_ttfa", "vision", "open_only") if k in p}
    ranked = rank_cards(cards, task, p["w"], tokens, limit=300, released_after=after, **filters)
    scale = {r["card"]["id"]: r for r in rank_cards(cards, task, p["w"], tokens, limit=100000)}
    fstats = field_log.stats()

    def row(cand: str, r: dict[str, Any]) -> dict[str, Any]:
        card, full = r["card"], scale.get(r["card"]["id"], r)
        fs, tcov = fstats.get(cand), task_coverage(card, task)
        why = (
            f"{card.get('name')}: quality {r['quality']:.0f}/100 for {task}, "
            f"${r['cost'] * 1000:.3f} per 1000 tasks on this channel ({card.get('price_source')})"
            + (f", {r['seconds']:.1f} s/task" if r["seconds"] else "")
        )
        if task != "general" and tcov < 0.5:
            why += f"; only {tcov:.0%} of the {task} benchmarks cover it, general quality used as a proxy"
        if not card.get("released"):
            why += "; release date unknown"
        elif after and str(card["released"]) < after:
            why += f"; released {card['released']}, older than {months:g} months"
        if fs and fs["reliable"] is False:
            why += f"; field: only {fs['ok_rate']:.0%} of {fs['calls']} real calls answered"
        return {
            "candidate": cand,
            "variant": card["id"],
            "released": card.get("released"),
            "quality": round(r["quality"], 1),
            "coverage": r["coverage"],
            "task_coverage": tcov,
            "cost_per_task": round(r["cost"], 6),
            "price_source": card.get("price_source"),
            "seconds_per_task": round(r["seconds"], 2) if r["seconds"] else None,
            "score": full.get("score"),
            "why": why,
            "field": fs,
        }

    picked: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    creators: set[Any] = set()
    for r in ranked:
        creator = r["card"].get("creator")
        if creator in creators:
            continue
        item = row(where_of[r["card"]["id"]], r)
        if (item["field"] or {}).get("reliable") is False:
            demoted.append(item)  # another model of this creator may still take its place
        else:
            picked.append(item)
            creators.add(creator)
        if len(picked) >= n:
            break
    rows = (picked + demoted)[:n]
    if not with_scores and not compare:
        return [r["candidate"] for r in rows]
    compared = []
    for cid in compare or []:
        key = model_key(cid.split(":", 1)[-1].split("/")[-1])
        matches = [r for r in scale.values() if model_key(r["card"]["id"]) == key] or [
            r for r in scale.values() if model_key(r["card"]["id"]).startswith(key + "-")
        ]
        best = max(matches, key=lambda r: r.get("score") or 0, default=None)
        compared.append(
            row(cid, best)
            if best
            else {"candidate": cid, "why": "no benchmark data or not sold on your channels"}
        )
    return {"task": task, "preset": preset, "max_age_months": months, "candidates": rows, "compare": compared}


def export_model_data(store: Store | None = None) -> dict[str, Any]:
    """Everything the interactive model picker needs, in one JSON: model cards (compact), the scoring tables, and
    media/STT leaderboards. The page recomputes rankings in the browser with the same formulas as selector.cards."""
    st = _store(store)
    cards = model_cards(st)
    keep = ("id", "name", "creator", "released", "price_in", "price_out", "buy", "tps", "ttft", "ttfa", "think",
            "think_source", "vision", "open", "q", "popular")  # fmt: skip
    boards = {}
    for board in ("artificial_analysis/media/text-to-image", "artificial_analysis/media/image-editing",
                  "artificial_analysis/media/text-to-speech", "artificial_analysis/media/text-to-video",
                  "open_asr/avg_wer"):  # fmt: skip
        boards[board] = [
            {k: r.get(k) for k in ("model_key", "model_name", "score", "rank", "organization", "votes")}
            for r in leaderboard(board, 40, st)
        ]
    stt_prices = [
        {"offer_id": v["offer_id"], "per_minute": v.get("per_unit"), "model": v["model"]}
        for v in find_offers(capability="speech.stt", limit=60, store=st)["offers"]
        if v.get("unit") == "minute"
    ]
    srcs = st.cache_items("benchsrc:")
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cards": [{k: c.get(k) for k in keep} for c in cards],
        "components": {k: v[0] for k, v in COMPONENTS.items()},
        "tasks": TASKS,
        "profiles": PROFILES,
        "presets": PRESETS,
        "media": boards,
        "stt_prices": stt_prices,
        "attribution": sorted({m.get("attribution") for m in srcs.values() if m.get("attribution")}),
    }


def pull_conditions(
    store: Store | None = None, adapter: LiteLLMConditionsAdapter | None = None
) -> dict[str, Any]:
    """Refresh per-model batch/off-peak discount conditions from LiteLLM's public pricing JSON (MIT; read
    over HTTP, the `litellm` package itself is never installed — ADR-0003). Makes the batch/off-peak
    discount toggles in `estimate_cost`/`monthly_cost` apply to real offers instead of only test data."""
    st = _store(store)
    ad = adapter or LiteLLMConditionsAdapter()
    try:
        conds = ad.fetch()
    except Exception as e:  # mirror pull_offers/pull_benchmarks: one broken source must not raise
        return {"litellm": {"error": f"{type(e).__name__}: {str(e)[:200]}"}}
    for key, val in conds.items():
        st.cache_put(
            f"conditions:{key}",
            {**val, "source": ad.id, "license": ad.license, "attribution": ad.attribution},
        )
    return {"litellm": {"models": len(conds)}}


def conditions_prior(store: Store, model: str) -> dict[str, Any] | None:
    return store.cache_get(f"conditions:{model_key(model)}")


def conditions_priors(store: Store | None = None) -> list[dict[str, Any]]:
    """All cached LiteLLM-derived conditions, alphabetical by model key."""
    st = _store(store)
    items = st.cache_items("conditions:")
    return sorted(
        ({"model_key": k.removeprefix("conditions:"), **v} for k, v in items.items()),
        key=lambda x: x["model_key"],
    )


def find_offers(
    query: str | None = None,
    capability: str | None = None,
    channel: str | None = None,
    constraints: Constraints | None = None,
    profile: LoadProfile | None = None,
    sort: str = "input",
    limit: int = 25,
    include_groups: bool = False,
    include_free: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    store: Store | None = None,
) -> dict[str, Any]:
    """Search offers. `query` matches model names (normalised). Sort: input | output | monthly | value
    (benchmark score / price; `benchmark` picks the board, e.g. "lmarena/text/coding" — see `benchmark_boards`;
    needs `pull_benchmarks` to have run at least once).
    Zero-priced offers (free tiers, subscription plans) are hidden unless `include_free`."""
    st = _store(store)
    raw = st.find_offers(
        model=model_key(query) if query else None, channel=channel, capability=capability, limit=20000
    )
    every = [o for o in raw if o.price.is_known]
    raw = [o for o in every if include_groups or o.group is None or (include_free and o.group == "free")]
    hidden_free = sum(_is_zero_priced(o) for o in raw) if not include_free else 0
    if not include_free:
        raw = [o for o in raw if not _is_zero_priced(o)]
    kept, rejected = apply(raw, constraints or Constraints())
    views = [
        offer_view(k.offer, st, profile) | ({"warnings": k.warnings} if k.warnings else {}) for k in kept
    ]
    _group_hints(every, views, st)
    note = (
        "Ranked by price only — quality is NOT considered yet. Don't present the cheapest as the best: "
        "shortlist by capability/brand knowledge, then run_eval on the user's own examples."
    )
    if sort == "monthly" and profile:
        views.sort(key=lambda v: v.get("monthly_real_estimate", v.get("monthly_list", float("inf"))))
    elif sort == "value":
        higher = True
        index = _bench_index(st)
        for v in views:
            _, boards = _resolve_bench(index, v["model"])
            b = boards.get(benchmark)
            v["benchmark"] = {"id": benchmark, **b} if b else None
            v["quality_rating"] = b["score"] if b else None
            if b:
                higher = b.get("higher_is_better", True)

        unit = TAXONOMY[capability].unit if capability in TAXONOMY else "token"

        def value(v: dict[str, Any]) -> tuple[bool, float]:
            q = v["quality_rating"]
            price = (
                v["input_per_1m"]
                if unit == "token"
                else (v.get("per_unit") if v.get("unit") == unit else None)
            )
            if (
                q is None or price is None
            ):  # no score, or priced in another unit: not comparable, listed after
                return True, 0.0
            return False, (-q / max(price, 1e-9)) if higher else q * max(price, 1e-9)

        views.sort(key=value)
        note = (
            f"Ranked by {benchmark} score per unit of price where a score exists — offers with no "
            "`quality_rating` are ranked last, not treated as bad. Leaderboards measure their own tasks "
            "(Elo = human preference votes), not yours: still validate with run_eval on your own examples."
        )
    elif sort == "input" and capability in TAXONOMY and TAXONOMY[capability].unit != "token":
        # non-token capability (STT per minute, TTS per 1k chars, ...): rank by the price in the capability's own
        # unit; offers priced in other units (GPU seconds, tokens) follow, unranked against each other
        unit = TAXONOMY[capability].unit
        views.sort(key=lambda v: (v.get("unit") != unit, v.get("per_unit") is None, v.get("per_unit") or 0))
        note = (
            f"Ranked by price per {unit}; offers priced in other units (GPU seconds, tokens) are listed after "
            "and are not comparable without a load estimate. Quality is NOT considered: run_eval on your own data."
        )
    else:
        field = "output_per_1m" if sort == "output" else "input_per_1m"
        views.sort(key=lambda v: (v[field] is None, v[field] or 0))
    return {
        "count": len(views),
        "offers": views[:limit],
        "rejected_by_constraints": rejected,
        "hidden_zero_priced": hidden_free,
        "note": note,
    }


def compare_prices(
    model: str,
    profile: LoadProfile | None = None,
    constraints: Constraints | None = None,
    include_groups: bool = False,
    include_free: bool = False,
    store: Store | None = None,
) -> dict[str, Any]:
    """Every way to buy one model across channels, cheapest real cost first (learned multipliers applied).
    Reseller price groups (`include_groups`) are informational: a key is usually routed automatically, and the
    cheapest group is often unreachable. Each ungrouped view carries `cheapest_group` as an unverified best case
    (with observed_share once `eval learn` has read the call log); `real_price_multiplier` is the measured truth."""
    st = _store(store)
    key = model_key(model)
    offers = [
        o for o in st.find_offers(model=key, limit=20000) if model_key(o.model) == key and o.price.is_known
    ]
    every = offers
    if not include_groups:
        offers = [o for o in offers if o.group is None]
    free = [o for o in offers if _is_zero_priced(o)]
    if not include_free:
        offers = [o for o in offers if not _is_zero_priced(o)]
    kept, rejected = apply(offers, constraints or Constraints())
    prof = profile or LoadProfile()
    views = [offer_view(k.offer, st, prof) | ({"warnings": k.warnings} if k.warnings else {}) for k in kept]
    _group_hints(every, views, st)
    views.sort(key=lambda v: v.get("monthly_real_estimate", v.get("monthly_list", float("inf"))))
    return {
        "model_key": key,
        "profile": prof.model_dump(),
        "offers": views,
        "rejected_by_constraints": rejected,
        "zero_priced_elsewhere": [o.id for o in free][:20] if not include_free else [],
        "note": "monthly_real_estimate uses real/list multipliers learned from your own channel call logs; "
        "list prices elsewhere. Confirm on the provider's page before committing spend.",
    }


def cross_check_price(
    model: str, store: Store | None = None, adapter: InferIndexAdapter | None = None
) -> dict[str, Any]:
    """Compare our cheapest known offer for a model against a live InferIndex lookup: price sanity + promo
    detection. Informational only — InferIndex is the closest competitor's data, not a source of truth.
    Makes one live network call (InferIndex is public, unauthenticated, rate-limited to 60 req/min/IP)."""
    st = _store(store)
    key = model_key(model)
    ours = [
        o
        for o in st.find_offers(model=key, limit=20000)
        if model_key(o.model) == key and o.price.is_known and o.group is None
    ]
    cheapest = min(ours, key=lambda o: o.price.input_per_1m or float("inf")) if ours else None
    ad = adapter or InferIndexAdapter()
    theirs = ad.cheapest(model) or {}
    out: dict[str, Any] = {
        "model_key": key,
        "ours": {
            "offer_id": cheapest.id,
            "channel": cheapest.channel,
            "input_per_1m": cheapest.price.input_per_1m,
            "output_per_1m": cheapest.price.output_per_1m,
        }
        if cheapest
        else None,
        "inferindex": theirs or None,
        "attribution": ad.attribution,
        "note": "Informational cross-check only — InferIndex is the closest competitor's data, not a source "
        "of truth. Confirm any discrepancy on the provider's own pricing page.",
    }
    our_input = cheapest.price.input_per_1m if cheapest else None
    their_input = theirs.get("input_per_1m")
    if our_input and their_input:
        delta = (our_input - their_input) / their_input
        out["input_price_delta_vs_inferindex"] = round(delta, 4)
        if abs(delta) > 0.15:
            out["note"] += " Our cheapest input price differs from InferIndex's by more than 15%."
    if theirs.get("promo"):
        out["note"] += " InferIndex flags an active promo on this model."
    return out


def _domain(url: str | None) -> str:
    if not url:
        return ""
    return (urlparse(url).netloc or "").removeprefix("www.").lower()


def ingest_price_change_notification(payload: dict[str, Any], store: Store | None = None) -> dict[str, Any]:
    """Record a changedetection.io "a watched pricing page changed" webhook and best-effort match it to
    offers whose `provenance.source` shares the changed page's domain, so an agent/owner knows which
    `offers pull`/`offers pull-conditions` to re-run. Never re-fetches automatically — changedetection.io
    only reports that a page changed, not what it changed to. See the self-host recipe in `docs/ARCHITECTURE.md`.

    `payload` is the notification body changedetection.io/Apprise posted, already parsed from JSON. Its exact
    wrapping depends on the configured notification URL scheme, so this defensively accepts either the
    templated fields directly at the top level, or nested as a JSON string under a common Apprise wrapper
    key ("body"/"message")."""
    if "watch_url" not in payload and "watch_uuid" not in payload:
        for k in ("body", "message"):
            v = payload.get(k)
            if isinstance(v, str):
                try:
                    inner = json.loads(v)
                except json.JSONDecodeError:
                    continue
                if isinstance(inner, dict):
                    payload = inner
                    break
    st = _store(store)
    watch_url = payload.get("watch_url") or ""
    domain = _domain(watch_url)
    matched_channels = (
        sorted({o.channel for o in st.find_offers(limit=20000) if _domain(o.provenance.source) == domain})
        if domain
        else []
    )
    record = {
        "watch_url": watch_url,
        "watch_title": payload.get("watch_title") or "",
        "diff_added": str(payload.get("diff_added") or "")[:2000],
        "diff_removed": str(payload.get("diff_removed") or "")[:2000],
        "matched_channels": matched_channels,
        "received_at": datetime.now(UTC).isoformat(),
    }
    uid = payload.get("watch_uuid") or payload.get("uuid") or watch_url or "unknown"
    st.cache_put(f"price_change:{uid}:{record['received_at']}", record)
    return record


def price_change_alerts(store: Store | None = None, limit: int = 40) -> list[dict[str, Any]]:
    """Recent changedetection.io notifications recorded by `ingest_price_change_notification`, newest first."""
    st = _store(store)
    rows = sorted(
        st.cache_items("price_change:").values(), key=lambda r: r.get("received_at", ""), reverse=True
    )
    return rows[:limit]


def estimate_cost(
    offer_id: str, profile: LoadProfile | None = None, store: Store | None = None
) -> dict[str, Any]:
    st = _store(store)
    o = st.get_offer(offer_id)
    if o is None:
        return {"error": f"unknown offer {offer_id}"}
    prof = profile or LoadProfile()
    b = monthly_cost(_with_conditions(o, st), prof)
    return {
        "offer": offer_view(o, st, prof),
        "breakdown": b.model_dump() if b else None,
        "profile": prof.model_dump(),
    }


def learned_prices(channel: str | None = None, store: Store | None = None) -> list[dict[str, Any]]:
    st = _store(store)
    items = st.cache_items(f"ratio:{channel}/" if channel else "ratio:")
    return sorted(
        ({"channel_model": k.removeprefix("ratio:"), **v} for k, v in items.items()),
        key=lambda d: d["channel_model"],
    )


def taxonomy() -> list[dict[str, Any]]:
    return [c.model_dump() for c in TAXONOMY.values()]


# ---------------------------------------------------------------- evaluation


def eval_plan(suite_path: str, only: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    suite = load_suite(suite_path)
    if only:
        suite.providers = [p for p in suite.providers if any(o in p.id or o in p.label for o in only)]
    if limit:
        suite.tests = suite.tests[:limit]
    return EvalRunner(suite, Store(), suite_path=suite_path).plan().model_dump(mode="json")


async def arun_eval(
    suite_path: str,
    budget: float = 0.25,
    only: list[str] | None = None,
    limit: int | None = None,
    judge: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Run a suite (spends money within `budget` and WHICHAPIAPI_BUDGET_TOTAL). Returns summary + recommendation."""
    suite = load_suite(suite_path)
    if only:
        suite.providers = [p for p in suite.providers if any(o in p.id or o in p.label for o in only)]
    if limit:
        suite.tests = suite.tests[:limit]
    if judge:
        suite.ext.judge = {"provider": {"id": judge, "label": f"judge {judge.split(':', 1)[-1]}"}}
    runner = EvalRunner(suite, Store(), suite_path=suite_path, budget=budget, env=dict(os.environ))
    rep = await runner.run()
    rec = recommend(rep.summaries, suite.ext.weights, suite.ext.min_pass_rate)
    if out_dir:
        from whichapiapi.report.render import save

        save(rep, rec, Path(out_dir))
    return {
        "run_id": rep.run_id,
        "recommendation": rec.model_dump(),
        "summaries": [s.model_dump(exclude={"price"}) for s in rep.summaries],
        "spent_list": rep.spent_computed,
        "spent_real": rep.spent_measured,
        "stopped_by_budget": rep.stopped_by_budget,
        "markdown": to_markdown(rep, rec),
    }


def run_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(arun_eval(*args, **kwargs))


async def arun_canary(
    suite_path: str, budget: float = 0.05, reset_baseline: bool = False, store: Store | None = None
) -> dict[str, Any]:
    """Run a small canary suite (spends real money within `budget`) and diff it against a stored per-provider
    baseline to catch a channel silently substituting or degrading the model behind a model name. First run
    for a given `suite_path`, or `reset_baseline=True`, just records the baseline. A clean run (no alerts)
    refreshes it; a run that raises an alert does not, so a real regression can't get silently absorbed —
    call again with `reset_baseline=True` once you've reviewed it."""
    st = _store(store)
    suite = load_suite(suite_path)
    runner = EvalRunner(suite, st, suite_path=suite_path, budget=budget, env=dict(os.environ))
    rep = await runner.run()
    key = f"canary:{suite_path}"
    baselines = {} if reset_baseline else (st.cache_get(key) or {})
    alerts = compare_to_baseline(rep.summaries, baselines)
    baseline_reset = not baselines or reset_baseline
    if baseline_reset or not alerts:
        st.cache_put(key, new_baselines(rep.summaries))
    st.cache_put(
        f"canarylast:{suite_path}",
        {
            "suite": suite_path,
            "run_id": rep.run_id,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "alerts": alerts,
            "baseline_reset": baseline_reset,
            "providers": {
                s.provider: {"pass_rate": s.pass_rate, "score": s.score, "latency_p50": s.latency_p50, "n": s.n,
                             "errors": s.errors, "baseline": (baselines.get(s.provider) or None)}
                for s in rep.summaries
            },
        },
    )  # fmt: skip
    return {
        "run_id": rep.run_id,
        "alerts": alerts,
        "baseline_reset": baseline_reset,
        "summaries": [s.model_dump(exclude={"price"}) for s in rep.summaries],
        "spent_list": rep.spent_computed,
        "spent_real": rep.spent_measured,
        "stopped_by_budget": rep.stopped_by_budget,
    }


def run_canary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(arun_canary(*args, **kwargs))


def canary_status(store: Store | None = None) -> list[dict[str, Any]]:
    """The last canary run of every suite (time, per-provider pass rate / score vs baseline, alerts). Read-only: no
    API calls, no spend — for tools that want to demote a model a channel silently swapped or degraded."""
    return sorted(
        _store(store).cache_items("canarylast:").values(), key=lambda r: r.get("at", ""), reverse=True
    )


# ---------------------------------------------------------------- model integrity (substitution checks)


def _parse_chat_body(resp: Any) -> Any:
    """A chat completion body; some channels answer SSE even without `stream` (seen on a reseller) — fold it."""
    text = resp.text
    if not text.lstrip().startswith("data:"):
        try:
            return resp.json()
        except ValueError:
            return {"error": {"message": text[:200]}}
    content, usage, rid, fp = [], None, None, None
    for line in text.splitlines():
        if not line.startswith("data: {"):
            continue
        try:
            ev = json.loads(line[6:])
        except ValueError:
            continue
        rid, fp = ev.get("id") or rid, ev.get("system_fingerprint") or fp
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            content.append((ch.get("delta") or ch.get("message") or {}).get("content") or "")
    return {"id": rid, "system_fingerprint": fp, "usage": usage,
            "choices": [{"message": {"content": "".join(content)}}]}  # fmt: skip


async def arun_integrity(
    watch_path: str,
    reset_baseline: bool = False,
    only: list[str] | None = None,
    store: Store | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Run the model-integrity checks (evaluator/integrity.py) for the models in a watch file. Spends real money:
    ~20 short calls per model, refused when `budget_per_model` × models would cross WHICHAPIAPI_BUDGET_TOTAL. Real
    cost is read back from new-api call logs by request id and written to the ledger. Baselines follow the canary
    policy: first run (or `reset_baseline`) records them; a run without alerts refreshes them; a run with alerts
    does not."""
    import time as _time

    import httpx

    from whichapiapi.evaluator import integrity as ig
    from whichapiapi.evaluator import tokenizers as tk
    from whichapiapi.evaluator.balance import BalanceProbe

    st = _store(store)
    cfg = yaml.safe_load(Path(watch_path).read_text())
    channels = cfg.get("channels") or {}
    watched = [m for m in cfg.get("models") or [] if not only or m["model"] in only]
    samples = int(cfg.get("samples", 4))
    est = float(cfg.get("budget_per_model", 0.01)) * len(watched)
    if cap := os.environ.get("WHICHAPIAPI_BUDGET_TOTAL"):
        already = st.real_spent_total()
        if already + est > float(cap):
            raise ValueError(
                f"integrity run needs ~${est:.3f}; global budget ${already:.4f} of ${float(cap):.2f} used"
            )
    local = {f: tk.deltas(ig.PREFIX, ig.CALIBRATION, f) for f in tk.LOCAL_FAMILIES}
    run_id = "integrity-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    t_start = _time.time()
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 6)))

    async def one(channel: str, model: str, probe: str, text: str, max_tokens: int) -> dict[str, Any]:
        ch = channels[channel]
        key = os.environ.get(ch["key_env"], "")
        async with sem:
            t0 = _time.perf_counter()
            try:
                r = await http.post(
                    ch["base_url"].rstrip("/") + "/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": text}],
                        "max_tokens": max_tokens,
                    },
                    headers={"Authorization": f"Bearer {key}"},
                )
            except httpx.HTTPError as e:
                return ig.observation(probe, 0, {"error": type(e).__name__}, None, _time.perf_counter() - t0)
            return ig.observation(
                probe, r.status_code, _parse_chat_body(r), r.headers, _time.perf_counter() - t0
            )

    async def model_obs(
        channel: str, model: str, probe_list: list[tuple[str, str, int]]
    ) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(one(channel, model, *p) for p in probe_list)))

    jobs = {(m["channel"], m["model"]): ig.probes(samples) for m in watched}
    for m in watched:  # a sibling that isn't watched itself still needs its fingerprint
        if m.get("sibling") and (m["channel"], m["sibling"]) not in jobs:
            jobs[(m["channel"], m["sibling"])] = ig.fingerprint_probes(samples)
    keys = list(jobs)
    results = await asyncio.gather(*(model_obs(c, mdl, jobs[(c, mdl)]) for c, mdl in keys))
    obs = dict(zip(keys, results, strict=True))
    if client is None:
        await http.aclose()

    report: dict[str, Any] = {
        "run_id": run_id,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": [],
    }
    alerts = []
    for m in watched:
        ch, model, family = m["channel"], m["model"], m["family"]
        bkey = f"integrity:{ch}/{model}"
        baseline = None if reset_baseline else st.cache_get(bkey)
        sib = m.get("sibling")
        sib_fp = ig.fingerprint(obs[(ch, sib)]) if sib else None
        count = tk.counter(family)
        res = ig.analyze(
            family, obs[(ch, model)], local, baseline, sib_fp, sib, count(ig.PREFIX) if count else None
        )
        if res["state"] and (baseline is None or res["status"] != "alert"):
            keep_tok = (baseline or {}).get("tok") if baseline else None
            st.cache_put(bkey, {**res["state"], **({"tok": keep_tok} if keep_tok else {})})
        entry = {"channel": ch, "model": model, "family": family, "sibling": sib, "status": res["status"],
                 "checks": res["checks"], "groups": (res["state"] or {}).get("groups"),
                 "baseline_reset": baseline is None}  # fmt: skip
        report["models"].append(entry)
        alerts += [
            f"{ch}:{model} {c['check']}: {c['detail']}" for c in res["checks"] if c["status"] == "alert"
        ]
        st.cache_put(f"integritylast:{ch}/{model}", {"at": report["at"], "run_id": run_id, **entry})

    # real cost from the channels' own call logs (exact per request id); fall back to the estimate
    spent = 0.0
    for name, ch in channels.items():
        ids = {o["request_id"] for (c, _), os_ in obs.items() if c == name for o in os_ if o["request_id"]}
        if not ids or ch.get("log") != "newapi":
            continue
        probe = BalanceProbe("newapi", ch["base_url"].removesuffix("/v1"), os.environ.get(ch["key_env"], ""))
        found: dict[str, float] = {}
        for _ in range(12):  # the log lags by up to a minute
            found = {e.request_id: e.cost_usd for e in probe.calls_since(t_start - 5) if e.request_id in ids}
            if len(found) >= len(ids):
                break
            await asyncio.sleep(10)
        cost = sum(found.values())
        for (c, mdl), os_ in obs.items():
            if c == name:
                part = sum(found.get(o["request_id"], 0.0) for o in os_)
                if part:
                    st.spend(run_id, c, mdl, "integrity", part, "measured")
        spent += cost
        report.setdefault("unmatched_calls", 0)
        report["unmatched_calls"] += len(ids) - len(found)
    report["spent_real"] = round(spent, 6)
    report["alerts"] = alerts
    report["status"] = max(
        (e["status"] for e in report["models"]), key=lambda s: ig.SEVERITY[s], default="ok"
    )
    return report


def run_integrity(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(arun_integrity(*args, **kwargs))


def integrity_status(store: Store | None = None) -> list[dict[str, Any]]:
    """The last integrity verdict per watched model (read-only: no calls, no spend)."""
    return sorted(
        _store(store).cache_items("integritylast:").values(), key=lambda r: (r["channel"], r["model"])
    )


def balance_forecast(channel: str | None = None) -> dict[str, Any]:
    """Remaining balance of a new-api custom channel (default: the primary one) and when it runs out at the 7-day
    average spending rate (from its call log). Warns when it would run out within 3 days."""
    from whichapiapi.evaluator.balance import BalanceProbe

    channel = channel or channels.primary()
    env = channels.key_env(channel) if channel else None
    key = os.environ.get(env or "")
    if not channel or not key or (channels.custom().get(channel) or {}).get("kind") != "newapi":
        return {"error": "no new-api custom channel with a key (see whichapiapi.channels)"}
    probe = BalanceProbe("newapi", channels.root_url(channel) or "", key)
    left = probe.remaining_usd()
    now = datetime.now(UTC).timestamp()
    calls = probe.calls_since(now - 7 * 86400, max_pages=60)
    day = sum(c.cost_usd for c in calls if c.ts >= now - 86400)
    week = sum(c.cost_usd for c in calls)
    rate = week / 7  # USD per day: the weekly average (a single busy day — e.g. test runs — would mislead)
    days_left = (left / rate) if left is not None and rate > 0 else None
    out = {"channel": channel, "remaining_usd": left, "spent_24h": round(day, 4), "spent_7d": round(week, 4),
           "days_left": round(days_left, 1) if days_left is not None else None}  # fmt: skip
    if days_left is not None and days_left < 3:
        out["warning"] = f"runs out in ~{days_left:.1f} days at the current rate"
    return out


def dashboard_data(days: float = 7.0, store: Store | None = None) -> dict[str, Any]:
    """Everything the owner's dashboard shows, as one JSON-able dict: real vs list spend on the reseller key (from its
    call log), router traffic and recent attempt trails, route health, integrity verdicts, balance forecast, invited
    keys and the provider key pool. Read-only, no paid calls."""
    from collections import defaultdict

    from whichapiapi import activity, keypool, keys, route_health
    from whichapiapi.cost.engine import call_cost
    from whichapiapi.evaluator.balance import BalanceProbe

    st = _store(store)
    now = datetime.now(UTC)
    spend: dict[str, Any] = {"by_model": [], "by_day": []}
    ch = channels.primary()
    key = os.environ.get(channels.key_env(ch) or "") if ch else None
    if ch and key and (channels.custom().get(ch) or {}).get("kind") == "newapi":
        calls = BalanceProbe("newapi", channels.root_url(ch) or "", key).calls_since(
            now.timestamp() - days * 86400, max_pages=80
        )
        by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "real_usd": 0.0, "list_usd": 0.0,
                                                                    "unpriced": 0, "groups": defaultdict(int)})  # fmt: skip
        by_day: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "real_usd": 0.0, "list_usd": 0.0}
        )
        for c in calls:
            offer = st.get_offer(f"{ch}/{c.model}")
            listed = (
                call_cost(offer.price, c.prompt_tokens, c.completion_tokens, c.cached_tokens)
                if offer
                else None
            )
            m, d = by_model[c.model], by_day[datetime.fromtimestamp(c.ts, UTC).strftime("%Y-%m-%d")]
            for b in (m, d):
                b["calls"] += 1
                b["real_usd"] += c.cost_usd
                b["list_usd"] += listed or 0.0
            m["unpriced"] += listed is None
            m["groups"][c.group or "?"] += 1
        spend["by_model"] = sorted(
            ({"model": k, **v, "real_usd": round(v["real_usd"], 5), "list_usd": round(v["list_usd"], 5),
              "groups": dict(v["groups"])} for k, v in by_model.items()),
            key=lambda r: -r["real_usd"],
        )  # fmt: skip
        spend["by_day"] = [{"day": k, **{x: round(y, 5) if isinstance(y, float) else y for x, y in v.items()}}
                           for k, v in sorted(by_day.items())]  # fmt: skip
    events = [e for e in activity.read_events(days) if e.get("surface") == "router"]
    recent = [
        {"ts": e["ts"], "policy": e.get("action"), "task": (e.get("args") or {}).get("task"),
         "key": (e.get("args") or {}).get("caller"), "trail": (e.get("args") or {}).get("trail") or [],
         "route": (e.get("result") or {}).get("route"), "status": (e.get("result") or {}).get("status"),
         "cost_usd": (e.get("result") or {}).get("cost_usd"), "ms": e.get("ms"), "ok": e.get("ok")}
        for e in events[-100:]
    ][::-1]  # fmt: skip
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "days": days,
        "balance": balance_forecast(),
        "spend": spend,
        "router": {**activity.router_usage(days), "recent": recent},
        "health": route_health.snapshot(),
        "integrity": integrity_status(st),
        "keys": [
            {"name": n, **{k: v for k, v in r.items() if k != "hash"}} for n, r in keys.list_keys().items()
        ],
        "pool": keypool.listing(),
        "errors": activity.report(days).get("recent_errors", [])[:20],
    }
