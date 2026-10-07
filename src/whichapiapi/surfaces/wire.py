"""Other wire protocols on top of the chat-completions router, so coding agents can use it directly:

- Anthropic Messages (`POST /v1/messages`, `/v1/messages/count_tokens`) — Claude Code via `ANTHROPIC_BASE_URL`.
- OpenAI Responses (`POST /v1/responses`) — Codex CLI via a provider with `wire_api = "responses"`.

Requests are translated to chat completions, routed by `router.complete`, and the answer (JSON or SSE) is translated
back. Stateless: Responses' `previous_response_id` is not supported (Codex sends the full input with `store=false`).
Thinking/reasoning blocks are not echoed back as Anthropic `thinking` blocks (their signatures can't be produced).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

# ------------------------------------------------------------------ Anthropic Messages → chat


def _image_url(source: dict[str, Any]) -> str | None:
    if source.get("type") == "base64" and source.get("data"):
        return f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
    if source.get("type") == "url":
        return source.get("url")
    return None


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def anthropic_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """An Anthropic Messages request as an OpenAI chat-completions request."""
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _blocks_text(body["system"])})
    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                url = _image_url(b.get("source") or {})
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif t == "document":
                src = b.get("source") or {}
                if src.get("type") == "text":
                    parts.append({"type": "text", "text": src.get("data", "")})
            elif t == "tool_use":
                calls.append({"id": b.get("id"), "type": "function",
                              "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})  # fmt: skip
            elif t == "tool_result":
                out = b.get("content")
                text = _blocks_text(out) if not isinstance(out, str) else out
                if b.get("is_error"):
                    text = f"[error] {text}"
                messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": text})
        if role == "assistant":
            text = "".join(p["text"] for p in parts if p["type"] == "text")
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            if text or calls:
                messages.append(msg)
        elif parts:
            only_text = all(p["type"] == "text" for p in parts)
            messages.append(
                {"role": role, "content": "\n".join(p["text"] for p in parts) if only_text else parts}
            )
    out: dict[str, Any] = {"model": body.get("model") or "auto", "messages": messages}
    if body.get("max_tokens"):
        out["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p", "stream"):
        if k in body:
            out[k] = body[k]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if body.get("tools"):
        out["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema") or {"type": "object"}}}
            for t in body["tools"] if t.get("name")
        ]  # fmt: skip
    tc = body.get("tool_choice") or {}
    if tc.get("type") == "any":
        out["tool_choice"] = "required"
    elif tc.get("type") == "tool" and tc.get("name"):
        out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    elif tc.get("type") == "none":
        out["tool_choice"] = "none"
    th = body.get("thinking") or {}
    if th.get("type") == "enabled":
        b = int(th.get("budget_tokens") or 0)
        out["reasoning_effort"] = "low" if b <= 4096 else "medium" if b <= 16384 else "high"
    return out


_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use",
         "content_filter": "end_turn"}  # fmt: skip


def _args_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
    except ValueError:
        return {"_raw": raw}
    return val if isinstance(val, dict) else {"value": val}


def chat_to_anthropic(data: dict[str, Any], model: str) -> dict[str, Any]:
    """A chat.completion as an Anthropic Messages response."""
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        content.append({"type": "tool_use", "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name"), "input": _args_obj(fn.get("arguments"))})  # fmt: skip
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": "tool_use"
        if msg.get("tool_calls")
        else _STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(0, (u.get("prompt_tokens") or 0) - cached),
            "output_tokens": u.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        },
    }


def anthropic_error(status: int, data: Any) -> dict[str, Any]:
    msg = ((data or {}).get("error") or {}).get("message") if isinstance(data, dict) else None
    kind = {400: "invalid_request_error", 401: "authentication_error", 402: "billing_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error"}.get(status, "api_error")  # fmt: skip
    return {"type": "error", "error": {"type": kind, "message": msg or f"upstream error {status}"}}


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def _chat_events(stream: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Parsed chat-completion chunks from an SSE byte stream."""
    buf = b""
    async for chunk in stream:
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            line = line.strip()
            if line.startswith(b"data: {"):
                try:
                    yield json.loads(line[6:])
                except ValueError:
                    continue


async def anthropic_stream(stream: AsyncIterator[bytes], model: str) -> AsyncIterator[bytes]:
    """Chat-completion SSE → Anthropic Messages SSE (message_start … message_stop)."""
    mid = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})  # fmt: skip
    index, open_kind, tools, stop, usage = -1, None, {}, "end_turn", {}
    async for ev in _chat_events(stream):
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                if open_kind != "text":
                    if open_kind:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
                    index, open_kind = index + 1, "text"
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "text", "text": ""}})  # fmt: skip
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "text_delta", "text": d["content"]}})  # fmt: skip
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if i not in tools:
                    if open_kind:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
                    index, open_kind = index + 1, f"tool{i}"
                    tools[i] = index
                    yield _sse("content_block_start", {"type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                          "name": fn.get("name"), "input": {}}})  # fmt: skip
                if fn.get("arguments"):
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": tools[i],
                               "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})  # fmt: skip
            if ch.get("finish_reason"):
                stop = "tool_use" if tools else _STOP.get(ch["finish_reason"], "end_turn")
    if open_kind:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                 "usage": {"output_tokens": (usage or {}).get("completion_tokens") or 0}})  # fmt: skip
    yield _sse("message_stop", {"type": "message_stop"})


def count_tokens_estimate(body: dict[str, Any]) -> int:
    """`/v1/messages/count_tokens`: a local o200k estimate of the request (the real model may count differently)."""
    from whichapiapi.evaluator.tokenizers import counter

    chat = anthropic_to_chat({**body, "max_tokens": 1})
    text = json.dumps(chat["messages"], ensure_ascii=False) + json.dumps(chat.get("tools") or [])
    count = counter("openai")
    return count(text) if count else len(text) // 4


# ------------------------------------------------------------------ OpenAI Responses → chat


def _resp_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for p in content or []:
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t == "input_image" and (p.get("image_url") or p.get("url")):
            parts.append({"type": "image_url", "image_url": {"url": p.get("image_url") or p.get("url")}})
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI Responses request as a chat-completions request."""
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending)})
            pending.clear()

    for it in items or []:
        t = it.get("type") or ("message" if it.get("role") else None)
        if t == "function_call":
            pending.append({"id": it.get("call_id") or it.get("id"), "type": "function",
                            "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}})  # fmt: skip
            continue
        flush()
        if t == "function_call_output":
            out = it.get("output")
            messages.append({"role": "tool", "tool_call_id": it.get("call_id"),
                             "content": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)})  # fmt: skip
        elif t == "message":
            role = {"developer": "system"}.get(it.get("role"), it.get("role") or "user")
            messages.append({"role": role, "content": _resp_content(it.get("content"))})
    flush()
    out: dict[str, Any] = {"model": body.get("model") or "auto", "messages": messages}
    if body.get("max_output_tokens"):
        out["max_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "stream", "parallel_tool_calls"):
        if k in body:
            out[k] = body[k]
    tools = [t for t in body.get("tools") or [] if t.get("type") == "function" and t.get("name")]
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                          "parameters": t.get("parameters") or {"type": "object"}}}
                        for t in tools]  # fmt: skip
    tc = body.get("tool_choice")
    if tc in ("auto", "none", "required"):
        out["tool_choice"] = tc
    elif isinstance(tc, dict) and tc.get("name"):
        out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        out["reasoning_effort"] = effort
    return out


def _usage(u: dict[str, Any] | None) -> dict[str, Any]:
    u = u or {}
    return {
        "input_tokens": u.get("prompt_tokens") or 0,
        "output_tokens": u.get("completion_tokens") or 0,
        "total_tokens": u.get("total_tokens")
        or (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0),
        "input_tokens_details": {
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        },
        "output_tokens_details": {
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        },
    }


def _response(rid: str, model: str, output: list[dict[str, Any]], usage: Any, status: str) -> dict[str, Any]:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output, "usage": _usage(usage) if usage is not None else None, "error": None,
            "incomplete_details": None}  # fmt: skip


def chat_to_responses(data: dict[str, Any], model: str) -> dict[str, Any]:
    msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
    output: list[dict[str, Any]] = []
    if msg.get("content"):
        output.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "status": "completed",
                       "role": "assistant",
                       "content": [{"type": "output_text", "text": msg["content"], "annotations": []}]})  # fmt: skip
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        output.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}",
                       "call_id": c.get("id") or f"call_{uuid.uuid4().hex[:24]}", "name": fn.get("name"),
                       "arguments": fn.get("arguments") or "{}", "status": "completed"})  # fmt: skip
    return _response(f"resp_{uuid.uuid4().hex[:24]}", model, output, data.get("usage") or {}, "completed")


async def responses_stream(stream: AsyncIterator[bytes], model: str) -> AsyncIterator[bytes]:
    """Chat-completion SSE → Responses SSE (response.created … response.completed)."""
    rid = f"resp_{uuid.uuid4().hex[:24]}"
    seq = 0

    def ev(kind: str, **data: Any) -> bytes:
        nonlocal seq
        seq += 1
        return _sse(kind, {"type": kind, "sequence_number": seq, **data})

    yield ev("response.created", response=_response(rid, model, [], None, "in_progress"))
    yield ev("response.in_progress", response=_response(rid, model, [], None, "in_progress"))
    output: list[dict[str, Any]] = []
    text_item: dict[str, Any] | None = None
    calls: dict[int, dict[str, Any]] = {}
    usage: Any = None

    def close_text() -> list[bytes]:
        nonlocal text_item
        if text_item is None:
            return []
        i, item = output.index(text_item), text_item
        txt = item["content"][0]["text"]
        item["status"] = "completed"
        text_item = None
        return [ev("response.output_text.done", item_id=item["id"], output_index=i, content_index=0, text=txt),
                ev("response.content_part.done", item_id=item["id"], output_index=i, content_index=0,
                   part=item["content"][0]),
                ev("response.output_item.done", output_index=i, item=item)]  # fmt: skip

    async for chunk in _chat_events(stream):
        usage = chunk.get("usage") or usage
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                if text_item is None:
                    text_item = {"type": "message", "id": f"msg_{uuid.uuid4().hex[:24]}", "status": "in_progress",
                                 "role": "assistant", "content": [{"type": "output_text", "text": "", "annotations": []}]}  # fmt: skip
                    output.append(text_item)
                    i = len(output) - 1
                    yield ev("response.output_item.added", output_index=i,
                             item={**text_item, "content": []})  # fmt: skip
                    yield ev("response.content_part.added", item_id=text_item["id"], output_index=i, content_index=0,
                             part={"type": "output_text", "text": "", "annotations": []})  # fmt: skip
                text_item["content"][0]["text"] += d["content"]
                yield ev("response.output_text.delta", item_id=text_item["id"], output_index=output.index(text_item),
                         content_index=0, delta=d["content"])  # fmt: skip
            for tc in d.get("tool_calls") or []:
                k = tc.get("index", 0)
                fn = tc.get("function") or {}
                if k not in calls:
                    for b in close_text():
                        yield b
                    item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:24]}",
                            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", "name": fn.get("name"),
                            "arguments": "", "status": "in_progress"}  # fmt: skip
                    calls[k] = item
                    output.append(item)
                    yield ev("response.output_item.added", output_index=len(output) - 1, item=dict(item))
                if fn.get("arguments"):
                    calls[k]["arguments"] += fn["arguments"]
                    yield ev("response.function_call_arguments.delta", item_id=calls[k]["id"],
                             output_index=output.index(calls[k]), delta=fn["arguments"])  # fmt: skip
    for b in close_text():
        yield b
    for item in calls.values():
        item["status"] = "completed"
        i = output.index(item)
        yield ev("response.function_call_arguments.done", item_id=item["id"], output_index=i,
                 arguments=item["arguments"])  # fmt: skip
        yield ev("response.output_item.done", output_index=i, item=item)
    yield ev("response.completed", response=_response(rid, model, output, usage or {}, "completed"))


# ------------------------------------------------------------------ one handler for both servers


async def handle(
    kind: str,
    body: dict[str, Any],
    guest: str | None = None,
    trace: bool = False,
    request_headers: Any = None,
) -> tuple[int, dict[str, str], Any]:
    """kind = "anthropic" | "responses": translate, route, translate back. Invited keys may only use `auto` policies,
    so a concrete model name from an agent's config (e.g. Claude Code's default) is routed as `auto` for them."""
    from whichapiapi.surfaces import router

    chat = anthropic_to_chat(body) if kind == "anthropic" else responses_to_chat(body)
    requested = str(chat.get("model") or "auto")
    note = ""
    if guest and not requested.startswith("auto"):
        chat["model"], note = "auto", f"model {requested!r} routed as 'auto' (invited keys)"
    status, headers, data = await router.complete(
        chat, guest=guest, trace=trace, request_headers=request_headers
    )
    if note:
        headers = {**headers, "x-whichapiapi-note": note}
    if hasattr(data, "__aiter__"):
        conv = anthropic_stream if kind == "anthropic" else responses_stream
        return status, headers, conv(data, requested)
    if status >= 400 or not isinstance(data, dict) or not data.get("choices"):
        err = (
            anthropic_error(status, data)
            if kind == "anthropic"
            else {
                "error": (data or {}).get("error")
                if isinstance(data, dict)
                else {"message": "upstream error"}
            }
        )
        return (status if status >= 400 else 502), headers, err
    out = chat_to_anthropic(data, requested) if kind == "anthropic" else chat_to_responses(data, requested)
    return status, headers, out
