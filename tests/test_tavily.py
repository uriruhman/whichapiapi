import asyncio

import httpx

from whichapiapi.schema.suite import load_suite
from whichapiapi.transports.tavily import TavilyTransport

OK = {
    "results": [
        {"title": "T1", "url": "https://a.example", "content": "Da Nang sunny", "score": 0.9},
        {"title": "T2", "url": "https://b.example", "content": "more", "score": 0.5},
    ],
    "response_time": 0.9,
}


def _transport(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TavilyTransport("https://api.tavily.com/", "k", client=client)


def test_call_builds_request_and_formats_results():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], req.read()
        return httpx.Response(200, json=OK)

    res = asyncio.run(
        _transport(handler).call(
            "advanced", [{"role": "user", "content": "weather"}], {"max_results": 2, "temperature": 1}
        )
    )
    assert seen["url"] == "https://api.tavily.com/search" and seen["auth"] == "Bearer k"
    assert b'"search_depth":"advanced"' in seen["body"].replace(b" ", b"")
    assert b"temperature" not in seen["body"]  # only Tavily params are passed through
    assert res.error is None and res.output.startswith("1. T1 - https://a.example\n   Da Nang sunny")
    assert res.raw["n_results"] == 2


def test_errors_use_transient_prefixes():
    def status(code):
        return asyncio.run(
            _transport(lambda r: httpx.Response(code, text="nope")).call(
                "basic", [{"role": "user", "content": "q"}], {}
            )
        )

    assert status(429).error.startswith("RateLimitError")
    assert status(503).error.startswith("InternalServerError")
    assert status(401).error.startswith("APIStatusError")
    assert status(200).error.startswith("APIStatusError")  # body has no `results`


def test_live_suite_loads_with_tavily_channel():
    suite = load_suite("examples/search/suite.live.yaml")
    ch = suite.ext.channels["tavily"]
    assert ch.protocol == "tavily" and [p.model for p in suite.providers] == ["basic", "advanced"]
    assert suite.providers[0].price.per_request == 0.008
