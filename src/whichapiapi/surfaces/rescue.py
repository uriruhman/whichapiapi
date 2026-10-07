"""Repairs applied to a non-streamed chat answer before it goes back to the client, so a coding agent sees the same
shape whichever model answered (mechanisms borrowed from freellmapi's tool-call rescue, MIT):

- `<think>…</think>` in `content` → `reasoning_content` (DeepSeek/Qwen-style models served raw);
- tool calls written as text when the request offered tools → structured `tool_calls`: Kimi/DeepSeek
  `<|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>{…}<|tool_call_end|>`, Llama/Groq
  `<function=NAME>{…}</function>`, Qwen/Hermes `<tool_call>{"name":…,"arguments":…}</tool_call>`, and a bare/fenced
  JSON object `{"name":…,"arguments":…}` — only for tool names the request actually offered;
- arguments double-encoded as a JSON string inside JSON (GLM) → decoded once.
Every repair is listed in `x-whichapiapi-repaired`.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)
_KIMI = re.compile(
    r"<\|tool_call_begin\|>\s*(?:functions\.)?([\w\-.]+)(?::\d+)?\s*<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>",
    re.S,
)
_LLAMA = re.compile(r"<function=([\w\-.]+)>(.*?)</function>", re.S)
_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCED = re.compile(r"^\s*(?:```(?:json)?\s*)?(\{.*\})\s*(?:```)?\s*$", re.S)
_SECTION = re.compile(r"<\|tool_calls_section_begin\|>|<\|tool_calls_section_end\|>")


def _tool_names(body: dict[str, Any]) -> set[str]:
    names = set()
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    for f in body.get("functions") or []:
        if isinstance(f, dict) and f.get("name"):
            names.add(f["name"])
    return names


def _args(raw: Any) -> str | None:
    """Arguments as a JSON string, or None when they are not valid JSON."""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
        except ValueError:
            return None
        return json.dumps(val, ensure_ascii=False) if isinstance(val, dict) else None
    return None


def _call(name: str, args: str) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def parse_text_tool_calls(text: str, names: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Tool calls found in `text` (only for offered `names`) and the text left over."""
    calls: list[dict[str, Any]] = []
    rest = text
    for rx in (_KIMI, _LLAMA):
        for m in rx.finditer(text):
            name, args = m.group(1), _args(m.group(2).strip())
            if name in names and args is not None:
                calls.append(_call(name, args))
                rest = rest.replace(m.group(0), "")
    for m in _HERMES.finditer(text):
        with_name = _named_json(m.group(1), names)
        if with_name:
            calls.append(with_name)
            rest = rest.replace(m.group(0), "")
    if not calls:
        m = _FENCED.match(text)
        hit = _named_json(m.group(1), names) if m else None
        if hit:
            calls.append(hit)
            rest = ""
    return calls, _SECTION.sub("", rest).strip()


def _named_json(raw: str, names: set[str]) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("name") not in names:
        return None
    args = _args(obj.get("arguments", obj.get("parameters", {})))
    return _call(obj["name"], args) if args is not None else None


def _undouble(args: Any) -> str | None:
    """'"{\\"a\\":1}"' (a JSON string holding JSON) → '{"a":1}'. None when nothing to fix."""
    if not isinstance(args, str):
        return None
    try:
        val = json.loads(args)
    except ValueError:
        return None
    if isinstance(val, str):
        try:
            inner = json.loads(val)
        except ValueError:
            return None
        if isinstance(inner, dict):
            return json.dumps(inner, ensure_ascii=False)
    return None


def repair(body: dict[str, Any], data: Any) -> list[str]:
    """Repair `data` (a chat.completion) in place. Returns what was repaired."""
    done: list[str] = []
    if not isinstance(data, dict):
        return done
    names = _tool_names(body)
    for choice in data.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            m = _THINK.match(content)
            if m:
                msg["reasoning_content"] = (msg.get("reasoning_content") or "") + m.group(1).strip()
                content = msg["content"] = content[m.end() :]
                done.append("think_tags")
            if names and not msg.get("tool_calls") and content.strip():
                calls, rest = parse_text_tool_calls(content, names)
                if calls:
                    msg["tool_calls"] = calls
                    msg["content"] = rest or None
                    choice["finish_reason"] = "tool_calls"
                    done.append("text_tool_calls")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            fixed = _undouble(fn.get("arguments")) if isinstance(fn, dict) else None
            if fixed is not None:
                fn["arguments"] = fixed
                done.append("double_encoded_args")
    return sorted(set(done))
