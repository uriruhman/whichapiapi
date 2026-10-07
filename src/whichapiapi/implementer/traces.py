"""Production traces → eval suite: turn logged LLM calls into test cases, so a model switch is judged on real traffic.

Reads the common trace formats and normalises each LLM call to `Trace(messages, output, model, tokens, cost, ms)`:
- Langfuse observations (API v2 `GET /api/public/v2/observations?type=GENERATION` pages, or a saved JSON of them);
- OpenTelemetry spans in OTLP/JSON (`resourceSpans`), with either OpenInference attributes (Arize Phoenix:
  `llm.input_messages.N.message.role|content`, `llm.model_name`, `llm.token_count.*`) or OTel GenAI ones
  (`gen_ai.input.messages` / `gen_ai.output.messages`, `gen_ai.request.model`, `gen_ai.usage.*`);
- JSON lines / a list of `{"messages"|"input", "output", "model"}` records (Braintrust / Helicone exports reshaped,
  or your own logs).

The suite keeps the most common system prompt as the fixed prompt, the last user message as `{{input}}`, and the
answer production gave as `{{reference}}` for the judge. Traffic is personal data by default (`contains_pii`).
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from whichapiapi.schema.naming import model_key


@dataclass
class Trace:
    messages: list[dict[str, str]]
    output: str = ""
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def system(self) -> str:
        return "\n".join(m["content"] for m in self.messages if m["role"] == "system")

    @property
    def last_user(self) -> str:
        return next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), "")


# ---------------------------------------------------------------- normalising


def _text(content: Any) -> str:
    """Message content in any of the usual shapes → plain text (images and tool parts dropped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for p in content if (t := _text(p)))
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return _text(content[key])
    return ""


def _messages(value: Any) -> list[dict[str, str]]:
    """Chat input in any of the usual shapes → [{"role", "content"}]."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return [{"role": "user", "content": value}] if value else []
        return _messages(parsed) if not isinstance(parsed, str) else [{"role": "user", "content": parsed}]
    if isinstance(value, dict):
        for key in ("messages", "input", "prompt"):
            if key in value:
                return _messages(value[key])
        if "role" in value:
            value = [value]
        else:
            return [{"role": "user", "content": json.dumps(value, ensure_ascii=False)}]
    out = []
    for m in value or []:
        if isinstance(m, dict) and (text := _text(m.get("content", m.get("parts")))):
            out.append({"role": str(m.get("role") or "user"), "content": text})
    return out


def _output(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return _output(parsed) if not isinstance(parsed, str) else parsed
    if isinstance(value, dict):
        if "choices" in value:  # a raw OpenAI response
            return _output((value["choices"] or [{}])[0].get("message", {}))
        if "content" in value or "parts" in value:
            return _text(value.get("content", value.get("parts")))
        for key in ("output", "completion", "text"):
            if key in value:
                return _output(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(_output(v) for v in value)
    return "" if value is None else str(value)


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    n = _num(value)
    return int(n) if n is not None else None


def from_langfuse(obs: dict[str, Any]) -> Trace | None:
    if obs.get("type") not in (None, "GENERATION"):
        return None
    usage = obs.get("usageDetails") or obs.get("usage") or {}
    cost = obs.get("totalCost", obs.get("calculatedTotalCost"))
    if cost is None and isinstance(obs.get("costDetails"), dict):
        cost = obs["costDetails"].get("total")
    latency = _num(obs.get("latency"))  # seconds in Langfuse
    return Trace(
        messages=_messages(obs.get("input")),
        output=_output(obs.get("output")),
        model=obs.get("model") or obs.get("providedModelName"),
        input_tokens=_int(usage.get("input", usage.get("promptTokens", obs.get("inputUsage")))),
        output_tokens=_int(usage.get("output", usage.get("completionTokens", obs.get("outputUsage")))),
        cost_usd=_num(cost),
        latency_ms=latency * 1000 if latency is not None else None,
        meta={"id": obs.get("id"), "trace": obs.get("traceId"), "name": obs.get("name")},
    )


def _attr_value(v: dict[str, Any]) -> Any:
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in v:
            return v[key]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values") or []]
    return None


def from_otel_span(span: dict[str, Any]) -> Trace | None:
    a = {x["key"]: _attr_value(x.get("value") or {}) for x in span.get("attributes") or []}
    msgs: list[dict[str, str]] = []
    if "gen_ai.input.messages" in a:
        msgs = _messages(a["gen_ai.input.messages"])
    else:
        i = 0
        while f"llm.input_messages.{i}.message.role" in a:
            msgs.append(
                {
                    "role": str(a[f"llm.input_messages.{i}.message.role"]),
                    "content": str(a.get(f"llm.input_messages.{i}.message.content") or ""),
                }
            )
            i += 1
        if not msgs and a.get("openinference.span.kind") == "LLM" and a.get("input.value"):
            msgs = _messages(a["input.value"])
    if not msgs:
        return None
    output = (
        _output(a["gen_ai.output.messages"])
        if "gen_ai.output.messages" in a
        else str(a.get("llm.output_messages.0.message.content") or _output(a.get("output.value")))
    )
    start, end = _num(span.get("startTimeUnixNano")), _num(span.get("endTimeUnixNano"))
    return Trace(
        messages=msgs,
        output=output,
        model=a.get("gen_ai.response.model") or a.get("gen_ai.request.model") or a.get("llm.model_name"),
        input_tokens=_int(a.get("gen_ai.usage.input_tokens", a.get("llm.token_count.prompt"))),
        output_tokens=_int(a.get("gen_ai.usage.output_tokens", a.get("llm.token_count.completion"))),
        cost_usd=_num(a.get("llm.cost.total")),
        latency_ms=(end - start) / 1e6 if start and end else None,
        meta={"span": span.get("spanId"), "name": span.get("name")},
    )


def from_record(r: dict[str, Any]) -> Trace | None:
    msgs = _messages(r.get("messages") or r.get("input") or r.get("prompt") or r.get("request"))
    if not msgs:
        return None
    usage = r.get("usage") or r.get("metrics") or {}
    return Trace(
        messages=msgs,
        output=_output(r.get("output", r.get("response", r.get("completion")))),
        model=r.get("model") or (r.get("metadata") or {}).get("model"),
        input_tokens=_int(usage.get("prompt_tokens", usage.get("input_tokens"))),
        output_tokens=_int(usage.get("completion_tokens", usage.get("output_tokens"))),
        cost_usd=_num(r.get("cost", usage.get("cost"))),
        latency_ms=_num(r.get("latency_ms")),
    )


def parse(data: Any) -> list[Trace]:
    """Any supported export (already JSON-decoded, or JSON-lines text) → traces, format auto-detected."""
    if isinstance(data, str):
        text = data.strip()
        try:
            data = json.loads(text)
        except ValueError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and "resourceSpans" in data:
        spans = [
            s
            for rs in data["resourceSpans"]
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
            for s in ss.get("spans") or []
        ]
        return [t for s in spans if (t := from_otel_span(s))]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if isinstance(data, dict):
        data = [data]
    out = []
    for r in data or []:
        if not isinstance(r, dict):
            continue
        t = from_langfuse(r) if ("traceId" in r and "type" in r) else from_record(r)
        if t is not None:
            out.append(t)
    return out


def fetch_langfuse(
    host: str,
    public_key: str,
    secret_key: str,
    name: str | None = None,
    since: str | None = None,
    limit: int = 200,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """GENERATION observations from Langfuse API v2 (cursor pages of ≤1000), newest first."""
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    http = client or httpx.Client(timeout=60)
    params: dict[str, Any] = {
        "type": "GENERATION",
        "limit": min(limit, 1000),
        "fields": "core,basic,io,model,usage,metrics",
    }
    if name:
        params["name"] = name
    if since:
        params["fromStartTime"] = since
    rows: list[dict[str, Any]] = []
    while len(rows) < limit:
        r = http.get(
            f"{host.rstrip('/')}/api/public/v2/observations",
            params=params,
            headers={"Authorization": f"Basic {auth}"},
        )
        r.raise_for_status()
        body = r.json()
        rows += body.get("data") or []
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor or not body.get("data"):
            break
        params["cursor"] = cursor
    return rows[:limit]


# ---------------------------------------------------------------- traces → suite


def summarize(traces: list[Trace]) -> dict[str, Any]:
    models = Counter(t.model for t in traces if t.model)
    costs = [t.cost_usd for t in traces if t.cost_usd is not None]
    return {
        "calls": len(traces),
        "models": dict(models.most_common()),
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "avg_input_tokens": _avg([t.input_tokens for t in traces]),
        "avg_output_tokens": _avg([t.output_tokens for t in traces]),
    }


def _avg(xs: list[int | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals)) if vals else None


def to_cases(traces: list[Trace], sample: int = 20) -> tuple[str, list[dict[str, Any]]]:
    """(system prompt, cases): the most common system prompt among the calls (others are dropped so the suite has one
    prompt), unique last-user inputs, spread evenly over the export rather than the newest `sample`."""
    systems = Counter(t.system for t in traces)
    system = systems.most_common(1)[0][0] if systems else ""
    seen, pool = set(), []
    for t in traces:
        if t.system == system and t.last_user and t.last_user not in seen:
            seen.add(t.last_user)
            pool.append(t)
    step = max(1, len(pool) // sample) if sample else 1
    picked = pool[::step][:sample] if sample else pool
    return system, [
        {"description": f"trace {t.meta.get('id') or t.meta.get('span') or i + 1}",
         "vars": {"input": t.last_user, "reference": t.output}}
        for i, t in enumerate(picked)
    ]  # fmt: skip


RUBRIC = (
    "The answer handles this input correctly and completely, following every instruction. For reference, production "
    "answered: <<<{{reference}}>>> — the new answer must be at least as correct and useful (not necessarily the same "
    "words)."
)


def current_candidate(model: str | None, channels: list[str], store: Any) -> str | None:
    """The production model as a "channel:model" id on the first of `channels` that sells it."""
    if not model:
        return None
    key = model_key(model.split("/")[-1])
    for ch in channels:
        for o in store.find_offers(model=key, channel=ch, capability="llm.chat", limit=50):
            if model_key(o.model.split("/")[-1]) == key:
                return f"{ch}:{o.model}"
    return None


def build_suite(
    traces: list[Trace],
    directory: Path,
    title: str,
    task: str = "general",
    sample: int = 20,
    candidates: list[str] | None = None,
    criteria: str = "",
    budget_usd: float = 0.10,
) -> dict[str, Any]:
    """Write a runnable suite from traces: production model (when a reachable channel sells it) vs the best
    variants for `task`, judged against production's own answers."""

    from whichapiapi import core
    from whichapiapi.implementer.suite_draft import draft_suite

    if not traces:
        raise ValueError("no LLM calls found in the traces")
    system, cases = to_cases(traces, sample)
    stats = summarize(traces)
    if not candidates:
        usable = core.reachable_channels()
        top_model = next(iter(stats["models"]), None)
        current = current_candidate(top_model, usable, core._store(None))
        best = core.suggest_candidates(task, n=4)
        candidates = ([current] if current else []) + [
            c
            for c in best
            if not current or model_key(c.split(":", 1)[1]) != model_key(current.split(":", 1)[1])
        ]
    result = draft_suite(
        directory, title, "{{input}}", cases, candidates, criteria or RUBRIC, system,
        budget_usd=budget_usd, contains_pii=True, task=task,
    )  # fmt: skip
    return {**result, "traffic": stats}
