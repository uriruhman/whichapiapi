import json

import httpx
import yaml

from whichapiapi import core
from whichapiapi.evaluator import integrity as ig
from whichapiapi.evaluator import tokenizers as tk

LOCAL = {"openai": [90, 120], "deepseek": [92, 131], "qwen": [94, 128], "glm": [87, 125], "mimo": [94, 129]}


def _obs(deltas, base_pt=9, who="OpenAI", censor=("troops crushed the protests", "it mocks Xi Jinping"),
         persona="A helpful assistant.", numbers=(47, 47, 73, 57), animals=("otter",) * 4, reasoning=(180, 200),
         id_="chatcmpl-1"):  # fmt: skip
    def o(probe, content="", pt=None, rt=None):
        u = {"prompt_tokens": pt, "completion_tokens": 5}
        if rt is not None:
            u["completion_tokens_details"] = {"reasoning_tokens": rt}
        data = {"id": id_, "usage": u, "choices": [{"message": {"content": content}}]}
        return ig.observation(probe, 200, data, {"x-routing-group": "g1"}, 1.0)

    out = [o("tok_base", "OK", base_pt)] + [o(f"tok_{i}", "OK", base_pt + d) for i, d in enumerate(deltas)]
    out += [o("who", who), o("persona", persona)] + [o(f"censor_{i}", c) for i, c in enumerate(censor)]
    out += [o("fp_number", str(n)) for n in numbers] + [o("fp_animal", a) for a in animals]
    out += [o("reason", ig.REASON_ANSWER, rt=r) for r in reasoning]
    return out


def _status(res, check):
    return [c["status"] for c in res["checks"] if c["check"] == check]


def test_genuine_openai_model_passes_and_records_state():
    res = ig.analyze("openai", _obs([90, 120]), LOCAL, None, prefix_tokens=3)
    assert res["status"] == "ok" and _status(res, "tokenizer") == ["ok"]
    assert res["state"]["tok"] == [90, 120] and res["state"]["fp"]["reasoning_median"] == 190


def test_chinese_model_behind_a_western_name_is_caught():
    res = ig.analyze("openai", _obs([92, 131], who="DeepSeek", censor=("I cannot answer that.", "Let's talk about something else.")), LOCAL, None, prefix_tokens=3)  # fmt: skip
    assert res["status"] == "alert"
    assert "deepseek" in next(c for c in res["checks"] if c["check"] == "tokenizer")["detail"]
    assert _status(res, "identity") == ["alert"] and _status(res, "censorship") == ["alert"]


def test_claude_has_no_local_tokenizer_but_must_not_match_a_chinese_one():
    assert _status(ig.analyze("claude", _obs([139, 170], who="Anthropic"), LOCAL, None), "tokenizer") == [
        "ok"
    ]
    res = ig.analyze("claude", _obs([87, 125], who="Anthropic"), LOCAL, None)
    assert _status(res, "tokenizer") == ["alert"]
    changed = ig.analyze("claude", _obs([120, 150], who="Anthropic"), LOCAL, {"tok": [139, 170]})
    assert _status(changed, "tokenizer") == ["alert"]


def test_hidden_prompt_tokens_and_agent_persona():
    res = ig.analyze("openai", _obs([90, 120], base_pt=4390, persona="A pragmatic coding assistant for the codebase."), LOCAL, None, prefix_tokens=3)  # fmt: skip
    hidden = [c for c in res["checks"] if c["check"] == "hidden_prompt"]
    assert [c["status"] for c in hidden] == ["warn", "warn"] and hidden[0]["tokens"] == 4387


def test_one_deflection_only_warns():
    res = ig.analyze("claude", _obs([139, 170], who="Anthropic", censor=("I don't have reliable information.", "It mocks Xi Jinping.")), LOCAL, None)  # fmt: skip
    assert _status(res, "censorship") == ["warn"]


def test_sibling_swap_is_detected_by_fingerprint():
    sol = _obs([90, 120], numbers=(73, 73, 47, 57), animals=("otter",) * 4, reasoning=(180, 190))
    base = ig.analyze("openai", sol, LOCAL, None, prefix_tokens=3)["state"]
    luna_like = _obs(
        [90, 120], numbers=(47, 47, 47, 37), animals=("fox", "otter", "fox", "lynx"), reasoning=(400, 420)
    )
    luna_fp = ig.fingerprint(luna_like)
    same = ig.analyze("openai", sol, LOCAL, base, luna_fp, "gpt-6-luna", prefix_tokens=3)
    assert _status(same, "fingerprint") == ["ok"]
    swapped = ig.analyze("openai", luna_like, LOCAL, base, luna_fp, "gpt-6-luna", prefix_tokens=3)
    assert _status(swapped, "fingerprint") == ["alert"]


def test_jsd_bounds():
    assert ig.jsd({"a": 2}, {"a": 5}) == 0 and abs(ig.jsd({"a": 1}, {"b": 1}) - 1) < 1e-9


async def test_arun_integrity_end_to_end(tmp_path, store, monkeypatch):
    monkeypatch.setenv("OL_KEY", "k")
    monkeypatch.setattr(tk, "deltas", lambda prefix, texts, family: LOCAL.get(family))
    monkeypatch.setattr(tk, "counter", lambda family: (lambda t: 3) if family == "openai" else None)
    watch = tmp_path / "watch.yaml"
    watch.write_text(yaml.safe_dump({
        "budget_per_model": 0.01,
        "samples": 2,
        "channels": {"ol": {"base_url": "https://ol.test/v1", "key_env": "OL_KEY"}},
        "models": [{"channel": "ol", "model": "sol", "family": "openai", "sibling": "luna"}],
    }))  # fmt: skip

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        pt = 9 + (90 if text.startswith(ig.PREFIX + "\n" + ig.CALIBRATION[0][:10]) else 0)
        pt += 120 if text.startswith(ig.PREFIX + "\n" + ig.CALIBRATION[1][:10]) else 0
        answer = {"who": "OpenAI", "reason": ig.REASON_ANSWER}.get("who" if "company" in text else "", "47")
        if "Tiananmen" in text or "Pooh" in text:
            answer = "troops crushed protests; it mocks Xi Jinping"
        if "1/97" in text:
            answer = ig.REASON_ANSWER
        return httpx.Response(200, json={"id": "chatcmpl-x", "usage": {"prompt_tokens": pt},
                                         "choices": [{"message": {"content": answer}}]})  # fmt: skip

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    res = await core.arun_integrity(str(watch), store=store, client=client)
    m = res["models"][0]
    assert m["baseline_reset"] and _status(m, "tokenizer") == ["ok"] and res["status"] in ("ok", "warn")
    assert store.cache_get("integrity:ol/sol")["tok"] == [90, 120]
    assert core.integrity_status(store)[0]["model"] == "sol"
