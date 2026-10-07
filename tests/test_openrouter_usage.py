from whichapiapi.adapters.openrouter_usage import OpenRouterUsageAdapter
from whichapiapi.selector.cards import build_cards


def test_parse_ranks_by_order():
    body = {"data": [{"id": "deepseek/deepseek-v4.1-flash"}, {"id": "z-ai/glm-5.3-flash"}, {"id": "x/y"}]}
    rows = OpenRouterUsageAdapter.parse("programming", body)
    assert [(r.model_name, r.rank, r.score) for r in rows] == [
        ("deepseek-v4.1-flash", 1, 3.0),
        ("glm-5.3-flash", 2, 2.0),
        ("y", 3, 1.0),
    ]
    assert rows[0].id == "openrouter_usage/programming" and rows[0].organization == "deepseek"


def test_usage_is_shown_on_cards_but_not_in_quality():
    aa = [{"slug": "m", "name": "M", "pricing": {"price_1m_input_tokens": 1, "price_1m_output_tokens": 2}}]
    boards = {"openrouter_usage/programming": {"score": 20, "rank": 1}}
    card = build_cards(aa, lambda _slug: boards)[0]
    assert card["popular"] == {"programming": 1}
    assert card["q"] == {}
