"""Model cards and task-aware ranking: one comparable record per model *variant* (reasoning effort included) and one
scoring function the CLI, MCP and the interactive page all share. Pure functions — `core.model_cards` feeds them.

Why variants: "Claude Opus 5.5 (Max Effort)" and "(High Effort)" are the same weights at very different prices per
task, because the max variant writes tens of thousands of hidden reasoning tokens. So cost is modelled **per task**:

    thinking_tokens ≈ max(0, time_to_first_answer_token − min(ttft, 1 s)) × output_tokens_per_second
    cost_per_task   = in_tokens × $in + (thinking_tokens + answer_tokens) × $out
    seconds_per_task = time_to_first_answer_token + answer_tokens / output_tokens_per_second

(time-to-first-answer and speed are Artificial Analysis measurements on a ~1k-token prompt; the thinking estimate
scales with how hard your task is, so treat it as a relative, not absolute, figure.)

Quality is task-dependent: each task blends normalized components (each source converted to a 0–100 percentile over
all models it covers, so an Elo, an index and a pass rate become comparable). A missing component is dropped and the
remaining weights renormalized; `coverage` says how much of the blend had data. Combining an independently measured
index (Artificial Analysis) with human-preference votes (LMArena) and task benchmarks (SWE-bench, BFCL, tau2, ...)
dampens any single benchmark's contamination or style bias.
"""

from __future__ import annotations

import math
from typing import Any

# component -> (benchmark board, higher is better)
COMPONENTS: dict[str, tuple[str, bool]] = {
    "intelligence": ("artificial_analysis/intelligence_index", True),
    "coding": ("artificial_analysis/coding_index", True),
    "math": ("artificial_analysis/math_index", True),
    "hle": ("artificial_analysis/bench/hle", True),
    "long_context": ("artificial_analysis/bench/lcr", True),
    "instructions": ("artificial_analysis/bench/ifbench", True),
    "agentic_tau2": ("artificial_analysis/bench/tau2", True),
    "agentic_terminal": ("artificial_analysis/bench/terminalbench_hard", True),
    "tool_calling": ("bfcl/overall/fc", True),
    "swe": ("swebench/verified", True),
    "arena_overall": ("lmarena/text/overall", True),
    "arena_coding": ("lmarena/text/coding", True),
    "arena_math": ("lmarena/text/math", True),
    "arena_hard": ("lmarena/text/hard_prompts", True),
    "arena_russian": ("lmarena/text/russian", True),
    "arena_creative": ("lmarena/text/creative_writing", True),
    "arena_webdev": ("lmarena/webdev/overall", True),
    "arena_vision": ("lmarena/vision/overall", True),
    "arena_document": ("lmarena/document/overall", True),
}

# task -> quality blend (weights over COMPONENTS)
TASKS: dict[str, dict[str, float]] = {
    "general": {"intelligence": 0.45, "arena_overall": 0.35, "arena_hard": 0.2},
    "coding": {
        "coding": 0.35,
        "swe": 0.2,
        "arena_coding": 0.2,
        "arena_webdev": 0.15,
        "agentic_terminal": 0.1,
    },
    "math": {"math": 0.45, "hle": 0.25, "arena_math": 0.3},
    "agents": {"agentic_tau2": 0.3, "agentic_terminal": 0.25, "tool_calling": 0.25, "instructions": 0.2},
    "russian": {"arena_russian": 0.7, "intelligence": 0.3},
    "writing": {"arena_creative": 0.6, "arena_overall": 0.25, "instructions": 0.15},
    "long_context": {"long_context": 0.6, "arena_document": 0.2, "intelligence": 0.2},
    "vision": {"arena_vision": 0.8, "intelligence": 0.2},
}

# typical token shapes per task: (input tokens, answer tokens)
PROFILES: dict[str, tuple[int, int]] = {
    "chat": (1500, 400),
    "classify": (800, 20),
    "summarize": (8000, 500),
    "rag": (12000, 600),
    "agent_step": (6000, 300),
    "code": (4000, 1500),
    "long_doc": (60000, 1000),
}

# ready-made choices: task, weights (quality, cost, speed), hard filters
PRESETS: dict[str, dict[str, Any]] = {
    "best": {"title": "Лучшее качество", "task": "general", "w": (1.0, 0.04, 0.0)},  # tiny cost weight: ties
    "optimal": {"title": "Оптимум", "task": "general", "w": (0.55, 0.3, 0.15)},
    "value": {"title": "Дёшево и сердито", "task": "general", "w": (0.35, 0.65, 0.0), "min_quality": 50},
    "fast": {"title": "Быстрая (≥150 ток/с)", "task": "general", "w": (0.6, 0.2, 0.2), "min_tps": 150},
    "ultrafast": {
        "title": "Очень быстрая (≥400 ток/с)",
        "task": "general",
        "w": (0.5, 0.2, 0.3),
        "min_tps": 400,
    },
    "realtime": {
        "title": "Мгновенный ответ (<1.5 с до ответа)",
        "task": "general",
        "w": (0.6, 0.3, 0.1),
        "max_ttfa": 1.5,
    },
    "coding": {"title": "Код", "task": "coding", "w": (0.7, 0.2, 0.1), "profile": "code"},
    "math": {"title": "Математика и рассуждения", "task": "math", "w": (0.75, 0.2, 0.05)},
    "agents": {
        "title": "Агенты и инструменты",
        "task": "agents",
        "w": (0.6, 0.25, 0.15),
        "profile": "agent_step",
    },
    "russian": {"title": "Русский язык", "task": "russian", "w": (0.6, 0.3, 0.1)},
    "writing": {"title": "Тексты", "task": "writing", "w": (0.6, 0.3, 0.1)},
    "vision": {"title": "Зрение (картинки на входе)", "task": "vision", "w": (0.6, 0.3, 0.1), "vision": True},
    "long_context": {
        "title": "Длинные документы",
        "task": "long_context",
        "w": (0.6, 0.3, 0.1),
        "profile": "long_doc",
    },
    "open": {"title": "Открытые веса", "task": "general", "w": (0.6, 0.3, 0.1), "open_only": True},
}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) and v == v else None


_EFFORTS = ("non-reasoning", "minimal", "xhigh", "max", "high", "medium", "low")


def effort_class(name: str) -> str:
    """Reasoning setting from an Artificial Analysis variant name: "(Reasoning, Max Effort)" → "max"."""
    n = name.lower()
    return next((e for e in _EFFORTS if e in n), "other")


def thinking_tokens(
    tps: float | None, ttft: float | None, ttfa: float | None, name: str = ""
) -> float | None:
    """Reasoning before the answer, from measured timings. A streamed reasoning phase shows as answer-start later
    than first-token: (ttfa − ttft) × speed. Hidden reasoning (first token = answer) shows as a long first token,
    but so does a slow queue — so a long first token only counts as thinking for variants that say they reason."""
    if not tps or ttfa is None:
        return None
    ttft = ttft or 0.0
    if ttfa - ttft > 0.5:
        return round((ttfa - ttft) * tps)
    n = name.lower()
    reasons = effort_class(name) not in ("other", "non-reasoning") or "reasoning" in n.replace(
        "non-reasoning", ""
    )
    return round(max(0.0, ttft - 1.0) * tps) if reasons or "thinking" in n else 0


def impute_thinking(cards: list[dict[str, Any]]) -> None:
    """Variants without a speed measurement get the median thinking of measured variants of the same effort class
    (non-reasoning → 0); otherwise they would look like they answer for free. Marked `think_source: "imputed"`."""
    measured: dict[str, list[float]] = {}
    for c in cards:
        if c.get("think") is not None:
            c["think_source"] = "measured"
            measured.setdefault(effort_class(c["name"]), []).append(c["think"])
    everything = sorted(v for vs in measured.values() for v in vs)
    fallback = everything[len(everything) // 2] if everything else 0
    for c in cards:
        if c.get("think") is None:
            cls = effort_class(c["name"])
            vals = sorted(measured.get(cls) or [])
            c["think"] = 0 if cls == "non-reasoning" else (vals[len(vals) // 2] if vals else fallback)
            c["think_source"] = "imputed"


def percentiles(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    """Model -> 0..100 percentile (ties share the average rank)."""
    items = sorted(values.items(), key=lambda kv: kv[1] if higher_is_better else -kv[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 100.0}
    out, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        pct = 100.0 * ((i + j) / 2) / (n - 1)
        for k in range(i, j + 1):
            out[items[k][0]] = round(pct, 1)
        i = j + 1
    return out


def build_cards(
    aa_models: list[dict[str, Any]],
    boards_for: Any,
    cheapest_offer: Any = None,
    has_vision: Any = None,
    license_of: Any = None,
) -> list[dict[str, Any]]:
    """One card per Artificial Analysis model variant that has a price. `boards_for(name) -> {board: score row}`
    resolves benchmark boards for a model name; `cheapest_offer(name) -> view|None` the cheapest real catalog offer;
    `has_vision(name) -> bool|None`; `license_of(name) -> str|None`."""
    cards: list[dict[str, Any]] = []
    for m in aa_models:
        p = m.get("pricing") or {}
        pin, pout = _num(p.get("price_1m_input_tokens")), _num(p.get("price_1m_output_tokens"))
        if pin is None or pout is None:
            continue
        slug = m.get("slug") or ""
        tps = _num(m.get("median_output_tokens_per_second")) or None
        ttft = _num(m.get("median_time_to_first_token_seconds")) or None
        ttfa = _num(m.get("median_time_to_first_answer_token")) or ttft
        boards = boards_for(slug) or {}
        raw = {c: (boards.get(b) or {}).get("score") for c, (b, _h) in COMPONENTS.items()}
        ev = m.get("evaluations") or {}  # the variant's own AA numbers beat the slug-level merge
        for comp, field in (("intelligence", "artificial_analysis_intelligence_index"),
                            ("coding", "artificial_analysis_coding_index"),
                            ("math", "artificial_analysis_math_index")):  # fmt: skip
            if (v := _num(ev.get(field))) is not None:
                raw[comp] = v
        offer = cheapest_offer(slug) if cheapest_offer else None
        lic = license_of(slug) if license_of else None
        cards.append(
            {
                "id": slug,
                "name": m.get("name") or slug,
                "creator": (m.get("model_creator") or {}).get("name"),
                "released": m.get("release_date"),
                "price_in": pin,
                "price_out": pout,
                "buy": offer,  # cheapest real offer in our catalog, if any: {"offer_id", "input_per_1m", ...}
                "tps": tps,
                "ttft": ttft,
                "ttfa": ttfa,
                "think": thinking_tokens(tps, ttft, ttfa, m.get("name") or ""),
                "vision": bool(has_vision(slug)) if has_vision else raw.get("arena_vision") is not None,
                "open": bool(lic) and lic.lower() not in ("proprietary", "unknown"),
                "raw": {k: v for k, v in raw.items() if v is not None},
                # usage rank per OpenRouter category (1 = most used); shown, never blended into quality
                "popular": {
                    b.split("/", 1)[1]: row.get("rank")
                    for b, row in boards.items()
                    if b.startswith("openrouter_usage/") and row.get("rank")
                },
            }
        )
    # normalize every component to percentiles over the cards that have it
    for comp, (_b, higher) in COMPONENTS.items():
        pct = percentiles({c["id"]: c["raw"][comp] for c in cards if comp in c["raw"]}, higher)
        for c in cards:
            if c["id"] in pct:
                c.setdefault("q", {})[comp] = pct[c["id"]]
    for c in cards:
        c.setdefault("q", {})
    impute_thinking(cards)
    return cards


PROXY_SHRINK = 0.75  # general quality standing in for missing task boards keeps 3/4 of its distance from 50


def _blend(card: dict[str, Any], blend: dict[str, float]) -> tuple[float | None, float]:
    have = {k: w for k, w in blend.items() if k in card["q"]}
    cover = sum(have.values()) / sum(blend.values())
    raw = sum(card["q"][k] * w for k, w in have.items()) / sum(have.values()) if have else None
    return raw, cover


def task_coverage(card: dict[str, Any], task: str) -> float:
    """Share of the task's own benchmark blend this model was measured on (0..1)."""
    return round(_blend(card, TASKS[task])[1], 2)


def quality(card: dict[str, Any], task: str) -> tuple[float | None, float]:
    """(0..100 task quality, coverage 0..1). Partial data is shrunk toward the median (50): a model measured on one
    benchmark only can't outrank a model that is good on all of them by luck. The part of a task blend a model has
    no boards for is filled with its general quality, pulled toward 50 (PROXY_SHRINK), so a brand-new flagship that
    task boards haven't covered yet isn't dropped or beaten by old fully-covered models; it counts as half coverage."""
    raw, cover = _blend(card, TASKS[task])
    graw, gcover = _blend(card, TASKS["general"]) if task != "general" else (None, 0.0)
    if raw is None and graw is None:
        return None, 0.0
    general = gcover * graw + (1 - gcover) * 50.0 if graw is not None else 50.0
    fill = 50.0 + PROXY_SHRINK * (general - 50.0)
    q = (cover * raw if raw is not None else 0.0) + (1 - cover) * fill
    effective = cover + (1 - cover) * 0.5 * gcover
    return round(q, 1), round(effective, 2)


def task_cost(card: dict[str, Any], in_tokens: int, out_tokens: int, use_buy_price: bool = True) -> float:
    pin, pout = card["price_in"], card["price_out"]
    buy = card.get("buy") or {}
    if use_buy_price and buy.get("input_per_1m") is not None and buy.get("output_per_1m") is not None:
        pin, pout = min(pin, buy["input_per_1m"]), min(pout, buy["output_per_1m"])
    return (in_tokens * pin + ((card.get("think") or 0) + out_tokens) * pout) / 1e6


def task_seconds(card: dict[str, Any], out_tokens: int) -> float | None:
    if not card.get("tps"):
        return None
    return (card.get("ttfa") or card.get("ttft") or 0.0) + out_tokens / card["tps"]


def _log_scale(values: dict[str, float], lower_is_better: bool) -> dict[str, float]:
    """0..1 on a log scale across candidates (1 = best)."""
    pos = {k: v for k, v in values.items() if v and v > 0}
    if not pos:
        return {}
    lo, hi = math.log(min(pos.values())), math.log(max(pos.values()))
    span = (hi - lo) or 1.0
    return {
        k: (hi - math.log(v)) / span if lower_is_better else (math.log(v) - lo) / span for k, v in pos.items()
    }


def rank(
    cards: list[dict[str, Any]],
    task: str = "general",
    weights: tuple[float, float, float] = (0.55, 0.3, 0.15),
    profile: tuple[int, int] = PROFILES["chat"],
    min_quality: float | None = None,
    min_tps: float | None = None,
    max_ttfa: float | None = None,
    max_cost: float | None = None,
    vision: bool = False,
    open_only: bool = False,
    min_coverage: float = 0.3,
    released_after: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Utility = wq·quality + wc·cheapness + ws·speed, each 0..1 over the candidates that pass the hard filters.
    Cheapness and speed are log-scaled (a 10× price gap matters the same at any price level)."""
    rows = []
    for c in cards:
        q, cover = quality(c, task)
        if q is None or cover < min_coverage:
            continue
        cost = task_cost(c, *profile)
        secs = task_seconds(c, profile[1])
        if (min_quality is not None and q < min_quality) or (min_tps and (c.get("tps") or 0) < min_tps):
            continue
        if (max_ttfa and (c.get("ttfa") is None or c["ttfa"] > max_ttfa)) or (max_cost and cost > max_cost):
            continue
        if (vision and not c.get("vision")) or (open_only and not c.get("open")):
            continue
        if released_after and c.get("released") and str(c["released"]) < released_after:
            continue  # unknown release dates are kept (callers flag them)
        rows.append({"card": c, "quality": q, "coverage": cover, "cost": cost, "seconds": secs})
    cheap = _log_scale({r["card"]["id"]: r["cost"] for r in rows}, lower_is_better=True)
    fast = _log_scale({r["card"]["id"]: r["seconds"] for r in rows if r["seconds"]}, lower_is_better=True)
    wq, wc, ws = weights
    total = (wq + wc + ws) or 1.0
    for r in rows:
        cid = r["card"]["id"]
        r["score"] = round(
            100 * (wq * r["quality"] / 100 + wc * cheap.get(cid, 0.0) + ws * fast.get(cid, 0.0)) / total, 1
        )
    rows.sort(key=lambda r: -r["score"])
    return rows[:limit]
