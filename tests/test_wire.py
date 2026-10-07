import json

import httpx

from whichapiapi.surfaces import router, wire


def _sse(*events):
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


async def _aiter(data: bytes):
    for i in range(0, len(data), 37):  # split mid-line on purpose
        yield data[i : i + 37]


def _events(raw: bytes) -> list[tuple[str, dict]]:
    out = []
    for block in raw.decode().split("\n\n"):
        if block.strip():
            ev, data = block.split("\n", 1)
            out.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_anthropic_request_translation():
    body = {
        "model": "claude-sonnet-5-5",
        "max_tokens": 1000,
        "system": [{"type": "text", "text": "Be brief."}],
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "tools": [{"name": "read", "description": "read a file", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "open a.py"},
                                         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                              {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"p": "a.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "print(1)"}]}]},
        ],
    }  # fmt: skip
    chat = wire.anthropic_to_chat(body)
    assert chat["messages"][0] == {"role": "system", "content": "Be brief."}
    assert chat["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA=="},
    }
    assert chat["messages"][2]["tool_calls"][0]["function"] == {"name": "read", "arguments": '{"p": "a.py"}'}
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"}
    assert chat["tool_choice"] == "required" and chat["reasoning_effort"] == "medium"
    assert chat["tools"][0]["function"]["parameters"] == {"type": "object"}


def test_chat_to_anthropic_with_tool_call():
    data = {"choices": [{"finish_reason": "tool_calls", "message": {"content": "reading",
            "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": '{"p": 1}'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 40}}}  # fmt: skip
    out = wire.chat_to_anthropic(data, "auto")
    assert out["stop_reason"] == "tool_use" and out["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read",
        "input": {"p": 1},
    }
    assert out["usage"] == {"input_tokens": 60, "output_tokens": 7, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 0}  # fmt: skip


async def test_anthropic_stream_text_then_tool():
    raw = _sse(
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": '{"p"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': 1}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"completion_tokens": 9}},
    )  # fmt: skip
    out = b"".join([c async for c in wire.anthropic_stream(_aiter(raw), "auto")])
    kinds = [k for k, _ in _events(out)]
    assert kinds == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"]  # fmt: skip
    evs = _events(out)
    assert (
        evs[4][1]["content_block"]["type"] == "tool_use" and evs[8][1]["delta"]["stop_reason"] == "tool_use"
    )
    assert evs[8][1]["usage"]["output_tokens"] == 9


def test_responses_request_translation():
    body = {
        "model": "auto:coding",
        "instructions": "You are Codex.",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
            {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": '{"cmd": "ls"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.py"},
            {"type": "reasoning", "summary": []},
        ],
        "tools": [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {"type": "web_search"},
        ],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 500,
    }
    chat = wire.responses_to_chat(body)
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "system", "user", "assistant", "tool"]
    assert chat["messages"][3]["tool_calls"][0]["id"] == "call_1" and chat["messages"][4]["content"] == "a.py"
    assert [t["function"]["name"] for t in chat["tools"]] == ["shell"] and chat["reasoning_effort"] == "high"
    assert chat["max_tokens"] == 500


async def test_responses_stream_events():
    raw = _sse(
        {"choices": [{"delta": {"content": "Do"}}]},
        {"choices": [{"delta": {"content": "ne"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "shell", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    )  # fmt: skip
    evs = _events(b"".join([c async for c in wire.responses_stream(_aiter(raw), "auto")]))
    kinds = [k for k, _ in evs]
    assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed"
    assert "".join(d["delta"] for k, d in evs if k == "response.output_text.delta") == "Done"
    done = evs[-1][1]["response"]
    assert [o["type"] for o in done["output"]] == ["message", "function_call"]
    assert done["output"][1]["call_id"] == "call_9" and done["usage"]["input_tokens"] == 5
    assert [d["sequence_number"] for _, d in evs] == list(range(1, len(evs) + 1))


async def test_handle_routes_guests_as_auto(monkeypatch):
    monkeypatch.setenv("RESELLER_API_KEY", "k1")
    seen = {}
    monkeypatch.setattr(
        router, "candidates", lambda task, preset: seen.setdefault("task", task) and ["reseller:m"]
    )
    monkeypatch.setattr(router, "_charge", lambda guest, cand, usage: 0.0)

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(router.httpx, "AsyncClient", lambda **kw: client)
    status, headers, out = await wire.handle(
        "anthropic", {"model": "claude-opus-5-5", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]},
        guest="friend",
    )  # fmt: skip
    assert status == 200 and out["type"] == "message" and out["content"][0]["text"] == "hello"
    assert "routed as 'auto'" in headers["x-whichapiapi-note"] and seen["body"]["model"] == "m"
