from whichapiapi.selector.cards import (
    build_cards,
    effort_class,
    percentiles,
    quality,
    rank,
    task_cost,
    thinking_tokens,
)


def _aa(slug, name, pin, pout, tps=None, ttft=None, ttfa=None, ii=None):
    return {
        "slug": slug,
        "name": name,
        "pricing": {"price_1m_input_tokens": pin, "price_1m_output_tokens": pout},
        "median_output_tokens_per_second": tps,
        "median_time_to_first_token_seconds": ttft,
        "median_time_to_first_answer_token": ttfa,
        "evaluations": {"artificial_analysis_intelligence_index": ii},
    }


def test_thinking_estimate():
    assert thinking_tokens(100, 1.0, 11.0) == 1000  # streamed reasoning phase
    assert thinking_tokens(100, 30.0, 30.0, "Model X (Reasoning, Max Effort)") == 2900  # hidden reasoning
    assert thinking_tokens(700, 2.8, 2.8, "Mercury 2.5") == 0  # slow queue, not thinking
    assert effort_class("Claude (Adaptive Reasoning, Xhigh Effort)") == "xhigh"


def test_percentiles_and_quality_shrink():
    assert percentiles({"a": 1, "b": 2, "c": 3}) == {"a": 0.0, "b": 50.0, "c": 100.0}
    assert percentiles({"a": 5, "b": 1}, higher_is_better=False) == {"a": 0.0, "b": 100.0}
    card = {"q": {"intelligence": 100.0}}
    q, cover = quality(card, "general")
    assert cover == 0.45 and q == round(0.45 * 100 + 0.55 * 50, 1)  # partial data pulled toward the median


def test_rank_prefers_cheap_thinker_only_when_quality_holds():
    arena = {"lmarena/text/overall": {"score": 1400}, "lmarena/text/hard_prompts": {"score": 1400}}
    models = [
        _aa(
            "big-max", "Big (Reasoning, Max Effort)", 1.0, 10.0, tps=100, ttft=0.5, ttfa=100.5, ii=60
        ),  # 10k think
        _aa("big-low", "Big (Reasoning, Low Effort)", 1.0, 10.0, tps=100, ttft=0.5, ttfa=2.5, ii=58),
        _aa("tiny", "Tiny (Non-reasoning)", 0.1, 0.1, tps=500, ttft=0.2, ttfa=0.2, ii=10),
        _aa("unmeasured", "Other (Reasoning, Max Effort)", 1.0, 10.0, ii=59),
    ]
    cards = build_cards(models, lambda n: arena)
    by = {c["id"]: c for c in cards}
    assert (
        by["unmeasured"]["think_source"] == "imputed" and by["unmeasured"]["think"] == by["big-max"]["think"]
    )
    assert task_cost(by["big-max"], 1000, 500) > 10 * task_cost(by["big-low"], 1000, 500)
    top = rank(cards, "general", (0.5, 0.5, 0.0))
    assert top[0]["card"]["id"] in ("big-low", "tiny") and top[-1]["card"]["id"] in ("big-max", "unmeasured")
    assert [r["card"]["id"] for r in rank(cards, min_tps=400)] == ["tiny"]


def test_missing_task_boards_fall_back_to_general_quality():
    from whichapiapi.selector.cards import quality

    flagship = {"q": {"intelligence": 95, "arena_overall": 95, "arena_hard": 95}}  # no agentic boards yet
    old = {"q": {"agentic_tau2": 83, "agentic_terminal": 83, "tool_calling": 83, "instructions": 83}}
    q_new, cov_new = quality(flagship, "agents")
    assert q_new == round(50 + 0.75 * 45, 1) and cov_new == 0.5  # proxy: general quality, pulled toward 50
    assert quality(old, "agents") == (83.0, 1.0)
    assert quality({"q": {}}, "agents") == (None, 0.0)


def test_rank_released_after_keeps_unknown_dates():
    from whichapiapi.selector.cards import rank

    base = {"price_in": 1, "price_out": 1, "think": 0, "tps": 100, "ttft": 0.5, "ttfa": 0.5,
            "q": {"intelligence": 90, "arena_overall": 90, "arena_hard": 90}}  # fmt: skip
    cards = [{**base, "id": "old", "released": "2025-01-01"}, {**base, "id": "new", "released": "2026-09-01"},
             {**base, "id": "undated"}]  # fmt: skip
    assert {r["card"]["id"] for r in rank(cards, released_after="2026-04-01")} == {"new", "undated"}


def test_rank_models_rejects_unknown_task_and_profile():
    import pytest

    from whichapiapi import core

    with pytest.raises(ValueError, match=r"unknown task 'image generation'.*find_offers"):
        core.rank_models(task="image generation")
    with pytest.raises(ValueError, match="unknown profile"):
        core.rank_models(profile="essay")
