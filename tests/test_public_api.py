import json

import httpx
from starlette.testclient import TestClient

from whichapiapi import keys, tracing
from whichapiapi.surfaces import public_api, router

MSG = {"messages": [{"role": "user", "content": "hi"}]}


def test_keys_lifecycle():
    key = keys.add("ann", 0.5)
    assert key.startswith("wa-") and keys.find(key)[0] == "ann"
    assert "hash" not in keys.list_keys()["ann"] and keys.list_keys()["ann"]["hint"] == key[:7] + "…"
    keys.charge("ann", 0.2)
    assert abs(keys.remaining(keys.find(key)[1]) - 0.3) < 1e-9
    assert keys.set_limit("ann", 0.1) and keys.remaining(keys.find(key)[1]) < 0
    assert keys.revoke("ann") and keys.find(key) is None
    assert keys.find("wa-nope") is None and keys.find("sk-other") is None


async def test_guest_route_caps_tokens_and_charges(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:glm-5.3-flash"])
    monkeypatch.setattr(router, "route_cost", lambda cand, usage: 0.002)
    key = keys.add("bob", 0.5)
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, _ = await router.complete({"model": "auto", **MSG}, client, guest="bob", trace=True)
    assert status == 200 and sent["max_tokens"] == router.GUEST_MAX_TOKENS
    assert headers["x-whichapiapi-cost-usd"] == "0.002000"
    assert keys.find(key)[1]["spent_usd"] == 0.002 and keys.find(key)[1]["calls"] == 1
    (row,) = tracing.read("bob")
    assert (
        row["messages"] == MSG["messages"]
        and row["output"] == "ok"
        and row["route"] == "reseller:glm-5.3-flash"
    )
    assert row["cost_usd"] == 0.002 and row["type"] == "chat"
    status, _, data = await router.complete({"model": "claude-opus-5-5", **MSG}, client, guest="bob")
    assert status == 400 and "auto" in data["error"]["message"]


def _client(monkeypatch, balance=5.0):
    monkeypatch.setenv("WHICHAPIAPI_MCP_TOKEN", "owner-token")
    monkeypatch.setattr(public_api, "shared_balance_usd", lambda: balance)
    public_api._hits.clear()
    seen = []

    async def fake_complete(body, client=None, guest=None, trace=False, request_headers=None):
        seen.append(guest)
        return 200, {"x-whichapiapi-route": "reseller:m"}, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(router, "complete", fake_complete)
    return TestClient(public_api.app()), seen


def test_public_auth_limits_and_both_prefixes(monkeypatch):
    c, seen = _client(monkeypatch)
    assert c.post("/v1/chat/completions", json={"model": "auto", **MSG}).status_code == 401
    key = keys.add("cat", 0.1)
    auth = {"Authorization": f"Bearer {key}"}
    for path in ("/v1/chat/completions", "/chat/completions"):
        r = c.post(path, json={"model": "auto", **MSG}, headers=auth)
        assert r.status_code == 200 and r.headers["x-whichapiapi-route"] == "reseller:m"
    owner = c.post(
        "/chat/completions", json={"model": "gpt-x", **MSG}, headers={"Authorization": "Bearer owner-token"}
    )
    assert owner.status_code == 200 and seen == ["cat", "cat", None]
    assert c.get("/models", headers=auth).json()["data"][0]["id"] == "auto"
    assert c.get("/v1/usage", headers=auth).json()["remaining_usd"] == 0.1
    keys.charge("cat", 0.1)
    assert c.post("/v1/chat/completions", json={"model": "auto", **MSG}, headers=auth).status_code == 402
    page = c.get("/")
    assert page.status_code == 200 and "https://testserver/v1" in page.text


def test_public_rate_limit_and_balance_guard(monkeypatch):
    c, _ = _client(monkeypatch, balance=0.35)
    auth = {"Authorization": f"Bearer {keys.add('dan', 1.0)}"}
    assert c.post("/v1/chat/completions", json={"model": "auto", **MSG}, headers=auth).status_code == 503
    c, _ = _client(monkeypatch, balance=5.0)
    monkeypatch.setattr(public_api, "RATE_PER_MIN", 2)
    codes = [
        c.post("/v1/chat/completions", json={"model": "auto", **MSG}, headers=auth).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


async def test_tester_mcp_auth_and_tools(monkeypatch):
    from fastmcp import Client

    from whichapiapi.surfaces import tester_mcp

    monkeypatch.setenv("WHICHAPIAPI_MCP_TOKEN", "owner-token")
    v = tester_mcp.KeyVerifier()
    tester, guest = keys.add("tess", 0.1), keys.add("gus", 0.1, role="guest")
    assert (await v.verify_token(tester)).client_id == "tess"
    assert await v.verify_token(guest) is None  # guests get the router only
    assert (await v.verify_token("owner-token")).client_id == "owner"
    assert await v.verify_token("wa-wrong") is None
    async with Client(tester_mcp.build()) as c:
        names = {t.name for t in await c.list_tools()}
    assert {"suggest_candidates", "route_chat", "report_feedback", "audit_n8n", "my_usage"} <= names
    assert not names & {
        "run_eval",
        "pull_offers",
        "create_suite",
        "field_stats",
        "suite_from_traces",
        "group_usage",
    }
