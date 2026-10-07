import json

import httpx

from whichapiapi import route_health as rh
from whichapiapi.surfaces import rescue, router


def test_classify_error_classes():
    assert rh.classify(429, "Rate limit reached") == "rate"
    assert rh.classify(429, "You exceeded your daily request quota") == "daily"
    assert rh.classify(402) == "payment" and rh.classify(403, "insufficient balance") == "payment"
    assert (
        rh.classify(403, "account suspended") == "suspended" and rh.classify(403, "no access") == "forbidden"
    )
    assert rh.classify(404) == "gone" and rh.classify(400, "model gpt-x does not exist") == "gone"
    assert rh.classify(400, "maximum context length is 8192") == "too_large"
    assert rh.classify(400, "invalid temperature") == "client"
    assert rh.classify(0) == rh.classify(503) == "transient"


def test_rate_ladder_retry_after_and_daily(monkeypatch):
    now = 1_800_000_000.0
    rh.record("ol:a", 429, "slow down", now=now)
    assert 89 <= rh.cooling("ol:a", now=now) <= 90
    rh.record("ol:a", 429, "slow down", now=now + 100)
    assert 299 <= rh.cooling("ol:a", now=now + 100) <= 300
    rh.record("ol:b", 429, "slow down", headers={"retry-after": "600"}, now=now)
    assert rh.cooling("ol:b", now=now) == 600  # the upstream asked for longer than the ladder
    rh.record("ol:c", 429, "daily quota exceeded", now=now)
    assert 0 < rh.cooling("ol:c", now=now) <= 86400
    rh.record("ol:d", 402, "", now=now)  # payment problem cools the whole channel
    assert rh.cooling("ol:zzz", now=now) > 80000


def test_transient_streak_penalty_decay_and_order():
    now = 1_800_000_000.0
    for k in range(2):
        rh.record("x:a", 503, now=now + k)
    assert rh.cooling("x:a", now=now + 2) == 0  # two transient failures: penalty only
    assert rh.order(["x:a", "x:b"], now=now + 2) == ["x:b", "x:a"]  # penalty 2 → 1 position down
    rh.record("x:a", 503, now=now + 3)
    assert rh.cooling("x:a", now=now + 3) > 0  # third in a row → short cooldown
    assert rh.order(["x:a", "x:b"], now=now + 3) == ["x:b", "x:a"]
    assert rh.order(["x:a", "x:b"], now=now + 3 + 61 + 10 * rh.DECAY_S) == ["x:a", "x:b"]  # healed


def test_parse_duration():
    assert rh.parse_duration("2m59.5s") == 179.5 and rh.parse_duration("30") == 30.0
    assert rh.parse_duration("1h2m") == 3720 and rh.parse_duration("junk") is None


def _sse(*events):
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


async def test_stream_fails_over_before_first_token(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:a", "openrouter:b"])

    def handler(request):
        if request.url.host == "api.reseller.test":  # 200 but an in-band error before any token
            return httpx.Response(200, content=_sse({"error": {"message": "upstream saturated"}}))
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "hi"}}]}))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, it = await router.complete(
        {"model": "auto:coding", "stream": True, "messages": [{"role": "user", "content": "x"}]}, client
    )
    body = b"".join([c async for c in it])
    assert status == 200 and headers["x-whichapiapi-route"] == "openrouter:b"
    assert b'"hi"' in body and b"saturated" not in body
    assert "reseller:a=502:transient" in headers["x-whichapiapi-trail"]


async def test_200_with_error_body_falls_back_and_client_error_does_not(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:a", "openrouter:b"])

    def handler(request):
        if request.url.host == "api.reseller.test":
            return httpx.Response(200, json={"error": {"message": "model service temporarily unavailable"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hm</think>ok"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, data = await router.complete({"model": "auto", "messages": []}, client)
    assert status == 200 and headers["x-whichapiapi-route"] == "openrouter:b"
    assert data["choices"][0]["message"] == {"content": "ok", "reasoning_content": "hm"}
    assert headers["x-whichapiapi-repaired"] == "think_tags"


def test_rescue_text_tool_calls_and_double_encoding():
    body = {"tools": [{"type": "function", "function": {"name": "read_file"}}]}
    for text in (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>',
        '<function=read_file>{"path": "a.py"}</function>',
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
        '{"path": "a.py"}<|tool_call_end|><|tool_calls_section_end|>',
        '```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
    ):
        data = {"choices": [{"message": {"content": text}}]}
        assert rescue.repair(body, data) == ["text_tool_calls"], text
        msg = data["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.py"}'}
        assert data["choices"][0]["finish_reason"] == "tool_calls" and not msg["content"]
    # an unknown tool name is left alone
    data = {"choices": [{"message": {"content": '{"name": "rm_rf", "arguments": {}}'}}]}
    assert rescue.repair(body, data) == [] and "tool_calls" not in data["choices"][0]["message"]
    data = {
        "choices": [{"message": {"tool_calls": [{"function": {"name": "f", "arguments": '"{\\"a\\": 1}"'}}]}}]
    }
    assert rescue.repair({}, data) == ["double_encoded_args"]
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


async def test_cache_idempotency_and_stickiness(monkeypatch):
    from whichapiapi.surfaces import router_memory as memory

    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:a", "openrouter:b"])
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"n{len(calls)}"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    a = await router.complete(body, client, request_headers={"x-whichapiapi-cache": "1"})
    b = await router.complete(body, client, request_headers={"x-whichapiapi-cache": "1"})
    assert len(calls) == 1 and b[1]["x-whichapiapi-cache"] == "hit" and b[2] == a[2]
    h = {"idempotency-key": "k-1"}
    first = await router.complete({**body, "seed": 1}, client, request_headers=h)
    again = await router.complete({**body, "seed": 1}, client, request_headers=h)
    assert again[2] == first[2] and again[1]["x-whichapiapi-idempotent-replay"] == "1" and len(calls) == 2
    assert (await router.complete({**body, "seed": 2}, client, request_headers=h))[0] == 409
    # a conversation answered by openrouter:b stays there while it is healthy
    memory.remember_route(
        {"model": "auto", "messages": [{"role": "user", "content": "q1"}]}, None, "openrouter:b"
    )
    convo = {"model": "auto", "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                                           {"role": "user", "content": "q2"}]}  # fmt: skip
    _, headers, _ = await router.complete(convo, client)
    assert headers["x-whichapiapi-route"] == "openrouter:b" and headers["x-whichapiapi-sticky"] == "1"


async def test_embeddings_fall_back_only_to_the_same_model(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setattr(router, "embedding_targets", lambda m: ["reseller:emb-1", "openrouter:emb-1"])
    hosts = []

    def handler(request):
        hosts.append((request.url.host, json.loads(request.content)["model"]))
        if request.url.host == "api.reseller.test":
            return httpx.Response(503, text="down")
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 2}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, data = await router.embed({"model": "emb-1", "input": "x"}, client)
    assert status == 200 and hosts == [("api.reseller.test", "emb-1"), ("openrouter.ai", "emb-1")]
    assert headers["x-whichapiapi-route"] == "openrouter:emb-1" and data["data"][0]["embedding"] == [0.1, 0.2]


async def test_fusion_synthesizes_from_parallel_answers(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:a", "reseller:b", "reseller:c"])
    prompts = []

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        prompts.append((body["model"], text[:20]))
        if "Several assistants" in text:
            assert "# Answer 1" in text and "# Answer 2" in text
            return httpx.Response(200, json={"choices": [{"message": {"content": "merged"}}]})
        if body["model"] == "c":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": f"from {body['model']}"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, data = await router.complete(
        {"model": "auto:general@fusion", "messages": [{"role": "user", "content": "q"}]}, client
    )
    assert status == 200 and data["choices"][0]["message"]["content"] == "merged"
    assert headers["x-whichapiapi-fusion"] == "reseller:a,reseller:b"
    guest = await router.complete({"model": "auto@fusion", "messages": []}, client, guest="friend")
    assert guest[0] == 400
