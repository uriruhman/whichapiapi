import json

import httpx

from whichapiapi.surfaces import router


def test_classify():
    msg = lambda t: {"messages": [{"role": "user", "content": t}]}  # noqa: E731
    assert router.classify(msg("Fix this:\n```python\ndef f(): pass\n```")) == "coding"
    assert router.classify(msg("Напиши письмо клиенту о задержке поставки")) == "russian"
    assert router.classify(msg("solve the equation x^2 = 4")) == "math"
    assert router.classify({**msg("book a flight"), "tools": [{"type": "function"}]}) == "agents"
    img = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            }
        ]
    }
    assert router.classify(img) == "vision" and router.classify(msg("hello there")) == "general"
    assert router.parse_model("auto:coding@value", {}) == ("coding", "value")
    assert router.parse_model("auto", msg("привет, как дела у проекта")) == ("russian", "optimal")
    assert router.parse_model("gpt-6-luna", {}) == (None, "")


async def test_complete_falls_back_to_the_next_vendor(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setattr(
        router,
        "candidates",
        lambda task, preset: ["reseller:glm-5.3-flash", "openrouter:deepseek/deepseek-v4.1-flash"],
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body["model"], request.headers["authorization"]))
        if request.url.host == "api.reseller.test":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, data = await router.complete(
        {"model": "auto:coding", "messages": [{"role": "user", "content": "x"}]}, client
    )
    assert status == 200 and data["choices"][0]["message"]["content"] == "ok"
    assert calls == [
        ("api.reseller.test", "glm-5.3-flash", "Bearer k1"),
        ("openrouter.ai", "deepseek/deepseek-v4.1-flash", "Bearer k2"),
    ]
    assert (
        headers["x-whichapiapi-route"] == "openrouter:deepseek/deepseek-v4.1-flash"
        and headers["x-whichapiapi-task"] == "coding"
    )


async def test_non_retryable_error_is_returned_not_retried(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:a", "reseller:b"])
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(400, json={"error": {"message": "bad request"}})
        )
    )
    status, headers, _ = await router.complete({"model": "auto", "messages": []}, client)
    assert status == 400 and headers["x-whichapiapi-tried"] == "reseller:a"


async def test_tiny_max_tokens_raised_for_auto_routes(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["reseller:m"])
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _, headers, _ = await router.complete({"model": "auto", "messages": [], "max_tokens": 20}, client)
    assert seen["max_tokens"] == router.MIN_OUTPUT_TOKENS and "raised" in headers["x-whichapiapi-note"]
