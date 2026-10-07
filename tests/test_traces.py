import json

import httpx

from whichapiapi import core
from whichapiapi.implementer import traces as tr
from whichapiapi.schema.suite import load_suite

SYSTEM = "You classify support tickets."
LANGFUSE = {
    "data": [
        {
            "id": f"obs{i}",
            "traceId": f"t{i}",
            "type": "GENERATION",
            "name": "classify",
            "model": "gpt-4o-mini",
            "input": json.dumps(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"ticket {i}"}]
            ),
            "output": json.dumps({"role": "assistant", "content": f"label {i}"}),
            "usageDetails": {"input": 50, "output": 5, "total": 55},
            "totalCost": 0.0001,
            "latency": 0.8,
        }
        for i in range(6)
    ]
    + [{"id": "s", "traceId": "t", "type": "SPAN", "input": "x"}],
    "meta": {"cursor": None},
}


def _attrs(**kv):
    out = []
    for k, v in kv.items():
        val = {"intValue": v} if isinstance(v, int) else {"stringValue": v}
        out.append({"key": k, "value": val})
    return out


OTLP = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        {  # OpenInference (Phoenix)
                            "spanId": "a",
                            "startTimeUnixNano": "1000000000",
                            "endTimeUnixNano": "1500000000",
                            "attributes": _attrs(
                                **{
                                    "openinference.span.kind": "LLM",
                                    "llm.model_name": "gpt-4o-mini",
                                    "llm.input_messages.0.message.role": "system",
                                    "llm.input_messages.0.message.content": SYSTEM,
                                    "llm.input_messages.1.message.role": "user",
                                    "llm.input_messages.1.message.content": "phoenix ticket",
                                    "llm.output_messages.0.message.content": "billing",
                                    "llm.token_count.prompt": 40,
                                }
                            ),
                        },
                        {  # OTel GenAI
                            "spanId": "b",
                            "attributes": _attrs(
                                **{
                                    "gen_ai.request.model": "gpt-4o-mini",
                                    "gen_ai.input.messages": json.dumps(
                                        [
                                            {
                                                "role": "user",
                                                "parts": [{"type": "text", "content": "genai ticket"}],
                                            }
                                        ]
                                    ),
                                    "gen_ai.output.messages": json.dumps(
                                        [
                                            {
                                                "role": "assistant",
                                                "parts": [{"type": "text", "content": "tech"}],
                                            }
                                        ]
                                    ),
                                    "gen_ai.usage.output_tokens": 3,
                                }
                            ),
                        },
                        {"spanId": "c", "attributes": _attrs(**{"http.method": "GET"})},
                    ]
                }
            ]
        }
    ]
}


def test_langfuse_observations():
    calls = tr.parse(LANGFUSE)
    assert len(calls) == 6
    c = calls[0]
    assert c.system == SYSTEM and c.last_user == "ticket 0" and c.output == "label 0"
    assert (c.input_tokens, c.output_tokens, c.cost_usd, c.latency_ms) == (50, 5, 0.0001, 800.0)


def test_otlp_openinference_and_genai():
    calls = tr.parse(json.dumps(OTLP))
    assert [c.last_user for c in calls] == ["phoenix ticket", "genai ticket"]
    assert [c.output for c in calls] == ["billing", "tech"]
    assert calls[0].latency_ms == 500 and calls[1].output_tokens == 3


def test_jsonl_records():
    text = "\n".join(
        json.dumps({"input": [{"role": "user", "content": f"q{i}"}], "output": f"a{i}", "model": "m"})
        for i in range(3)
    )
    assert [c.output for c in tr.parse(text)] == ["a0", "a1", "a2"]


def test_build_suite_from_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "suggest_candidates", lambda task, n=4, **_: ["reseller:glm-5.3-flash"])
    result = tr.build_suite(tr.parse(LANGFUSE), tmp_path / "s", "tickets", sample=3)
    assert result["cases"] == 3 and result["traffic"]["models"] == {"gpt-4o-mini": 6}
    suite = load_suite(tmp_path / "s" / "suite.yaml")
    assert suite.ext.contains_pii
    assert suite.prompts[0].messages[0] == {"role": "system", "content": SYSTEM}
    assert suite.tests[0].vars["reference"] == "label 0"
    assert "{{reference}}" in str(suite.default_test.assert_[0].value)


def test_fetch_langfuse_pages_with_cursor():
    pages = iter([{"data": [{"id": 1}], "meta": {"cursor": "c1"}}, {"data": [{"id": 2}], "meta": {}}])
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=next(pages))

    rows = tr.fetch_langfuse(
        "https://lf", "pk", "sk", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert [r["id"] for r in rows] == [1, 2]
    assert seen[0].url.path == "/api/public/v2/observations" and seen[1].url.params["cursor"] == "c1"
    assert seen[0].headers["authorization"].startswith("Basic ")
