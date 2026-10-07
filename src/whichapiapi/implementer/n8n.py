"""n8n workflow JSON → the paid API calls inside it (Phase 4, read-only).

Finds language-model / embedding / transcription nodes (the n8n LangChain nodes and plain HTTP Request nodes that
call known AI APIs), the model each one uses (a literal, or the default inside a `{{ ... || 'model' }}` expression),
how often the workflow runs (schedule trigger), and whether a failure has a fallback (retry, error output, a second
language model on the same chain). Also flags API keys written into node parameters. Pure: no I/O, no pricing —
`implementer.audit` joins this with the offer catalog.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from whichapiapi.channels import channel_of_host

# LangChain model nodes: type suffix -> (capability, provider, parameter holding the model)
_LC_MODELS: dict[str, tuple[str, str, str]] = {
    "lmChatOpenRouter": ("llm.chat", "openrouter", "model"),
    "lmChatOpenAi": ("llm.chat", "openai", "model"),
    "lmChatAnthropic": ("llm.chat", "anthropic", "model"),
    "lmChatGoogleGemini": ("llm.chat", "google", "modelName"),
    "lmChatMistralCloud": ("llm.chat", "mistral", "model"),
    "lmChatGroq": ("llm.chat", "groq", "model"),
    "lmChatDeepSeek": ("llm.chat", "deepseek", "model"),
    "lmChatXAiGrok": ("llm.chat", "xai", "model"),
    "lmChatOllama": ("llm.chat", "ollama", "model"),
    "lmOpenAi": ("llm.chat", "openai", "model"),
    "embeddingsOpenAi": ("llm.embeddings", "openai", "model"),
    "embeddingsGoogleGemini": ("llm.embeddings", "google", "modelName"),
    "embeddingsMistralCloud": ("llm.embeddings", "mistral", "model"),
    "embeddingsOllama": ("llm.embeddings", "ollama", "model"),
    "openAi": ("llm.chat", "openai", "modelId"),  # the "OpenAI" app node (text resource)
}
# the node's own default when the model parameter is absent (n8n node schemas, verified via n8n-mcp 2026-09-29)
_LC_DEFAULTS = {"lmChatOpenRouter": "openai/gpt-4.1-mini", "embeddingsOpenAi": "text-embedding-3-small"}
# HTTP Request hosts of AI APIs -> provider; the path decides the capability
_HOSTS = {
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.groq.com": "groq",
    "api.deepseek.com": "deepseek",
    "api.mistral.ai": "mistral",
    "api.deepinfra.com": "deepinfra",
    "api.deepgram.com": "deepgram",
    "api.assemblyai.com": "assemblyai",
    "api.tavily.com": "tavily",
}
_LOCAL_HOSTS = ("ollama", "localhost", "127.0.0.1", "host.docker.internal")
_PATHS = (
    ("/audio/transcriptions", "speech.stt"),
    ("/listen", "speech.stt"),
    ("/transcript", "speech.stt"),
    ("/audio/speech", "speech.tts"),
    ("/embeddings", "llm.embeddings"),
    ("/images", "image.gen"),
    ("/search", "search.web"),
    ("/chat/completions", "llm.chat"),
    ("/messages", "llm.chat"),
    ("/completions", "llm.chat"),
    ("/api/chat", "llm.chat"),
    ("/api/generate", "llm.chat"),
)
_MODEL_IN_TEXT = re.compile(r"""\bmodel\W{1,6}['"]([\w.:/@-]{3,})['"]""")
_EXPR_DEFAULT = re.compile(r"""\|\|\s*['"]([\w.:/@-]{3,})['"]""")
# provider key shapes: a known prefix, a separator, then a long token that contains digits ("skill-gap-detector" is
# not a key); Google keys have no separator
_KEYLIKE = re.compile(
    r"\b(?:(?:sk|tvly|gsk|pk|rk|xai)[-_](?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{35})"
)


class ApiUse(BaseModel):
    """One node that calls a (usually paid) AI API."""

    workflow: str
    workflow_id: str | None = None
    node: str
    node_type: str
    capability: str
    provider: str
    local: bool = False  # self-hosted (Ollama etc.): no per-call price
    model: str | None = None  # literal model, or the default of an expression
    model_is_expression: bool = False  # chosen at runtime; `model` is the fallback value, if any
    retry: bool = False
    continues_on_error: bool = False
    fallback_model: bool = False  # another language model wired into the same chain/agent
    runs_per_month: float | None = None  # from the workflow's schedule trigger(s); None = webhook/manual
    prompt_chars: int | None = None  # static prompt text we could see, for a token estimate

    @property
    def has_fallback(self) -> bool:
        return self.retry or self.continues_on_error or self.fallback_model


class SecretLeak(BaseModel):
    workflow: str
    node: str
    field: str
    preview: str  # first/last characters only


class Extraction(BaseModel):
    workflows: int = 0
    uses: list[ApiUse] = Field(default_factory=list)
    secrets: list[SecretLeak] = Field(default_factory=list)


def load(data: Any) -> list[dict[str, Any]]:
    """A workflow export in any of n8n's shapes: one workflow, a list (`n8n export:workflow --all`), or
    `{"data": [...]}` (the public API)."""
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "nodes" in data:
        return [data]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [w for w in data["data"] if isinstance(w, dict) and "nodes" in w]
    if isinstance(data, list):
        return [w for w in data if isinstance(w, dict) and "nodes" in w]
    raise ValueError("not an n8n workflow export")


def extract(data: Any) -> Extraction:
    workflows = load(data)
    out = Extraction(workflows=len(workflows))
    for wf in workflows:
        name, runs = wf.get("name") or "?", runs_per_month(wf)
        fallbacks = _chains_with_fallback(wf)
        for node in wf.get("nodes") or []:
            out.secrets += _secrets(name, node)
            use = _api_use(node)
            if use is None:
                continue
            use.workflow, use.workflow_id, use.runs_per_month = name, wf.get("id"), runs
            use.fallback_model = node.get("name") in fallbacks
            out.uses.append(use)
    return out


def _model(value: Any) -> tuple[str | None, bool]:
    """(model, is_expression) from a parameter: plain string, resource locator {"value": ...} or an expression."""
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str) or not value.strip():
        return None, False
    if value.startswith("="):
        default = _EXPR_DEFAULT.search(value)
        literal = value[1:].strip()
        if "{{" not in literal:
            return literal, False
        return (default.group(1) if default else None), True
    return value.strip(), False


def _api_use(node: dict[str, Any]) -> ApiUse | None:
    ntype, params = node.get("type") or "", node.get("parameters") or {}
    common = {
        "workflow": "",
        "node": node.get("name") or "?",
        "node_type": ntype,
        "retry": bool(node.get("retryOnFail")),
        "continues_on_error": node.get("onError") not in (None, "stop", "stopWorkflow"),
    }
    suffix = ntype.rsplit(".", 1)[-1]
    if ntype.startswith("@n8n/n8n-nodes-langchain.") and suffix in _LC_MODELS:
        capability, provider, field = _LC_MODELS[suffix]
        if suffix == "openAi" and params.get("resource") not in (None, "text"):
            capability = {"audio": "speech.stt", "image": "image.gen"}.get(params.get("resource"), capability)
        model, is_expr = _model(params.get(field))
        if field not in params and suffix in _LC_DEFAULTS:
            model = _LC_DEFAULTS[suffix]
        prompt = sum(len(str(m.get("content", ""))) for m in (params.get("messages") or {}).get("values", []))
        return ApiUse(
            **common,
            capability=capability,
            provider=provider,
            local=provider == "ollama",
            model=model,
            model_is_expression=is_expr,
            prompt_chars=prompt or None,
        )
    if ntype == "n8n-nodes-base.httpRequest":
        url = str(params.get("url") or "")
        host = urlparse(url.lstrip("=")).hostname or ""
        local = host.startswith(_LOCAL_HOSTS) and ":11434" in url
        provider = _HOSTS.get(host) or channel_of_host(host) or ("ollama" if local else None)
        if provider is None:
            return None
        path = urlparse(url.lstrip("=")).path
        capability = next((cap for p, cap in _PATHS if p in path), "llm.chat")
        body = json.dumps(
            [params.get("jsonBody"), params.get("body"), params.get("bodyParameters")], ensure_ascii=False
        )
        literal = next(
            (
                str(p.get("value"))
                for p in (params.get("bodyParameters") or {}).get("parameters") or []
                if isinstance(p, dict) and p.get("name") == "model" and p.get("value")
            ),
            None,
        )
        found = _MODEL_IN_TEXT.search(body)
        model, is_expr = _model(literal) if literal else ((found.group(1), False) if found else (None, False))
        return ApiUse(
            **common,
            capability=capability,
            provider=provider,
            local=local,
            model=model,
            model_is_expression=is_expr or (model is None and "{{" in body),
            prompt_chars=len(body) if found else None,
        )
    return None


def _chains_with_fallback(wf: dict[str, Any]) -> set[str]:
    """Language-model nodes whose chain/agent has at least one other language model attached."""
    by_target: dict[str, list[str]] = {}
    for source, outputs in (wf.get("connections") or {}).items():
        for branch in outputs.get("ai_languageModel") or []:
            for link in branch or []:
                by_target.setdefault(link.get("node"), []).append(source)
    return {src for sources in by_target.values() if len(sources) > 1 for src in sources}


def _secrets(workflow: str, node: dict[str, Any]) -> list[SecretLeak]:
    out = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, str) and (m := _KEYLIKE.search(value)):
            key = m.group(0)
            out.append(
                SecretLeak(
                    workflow=workflow,
                    node=node.get("name") or "?",
                    field=path,
                    preview=f"{key[:6]}…{key[-3:]}",
                )
            )

    walk(node.get("parameters") or {}, "")
    return out


# ------------------------------------------------------------------ schedule → runs per month

_PER_MONTH = {
    "seconds": 30 * 86400,
    "minutes": 30 * 1440,
    "hours": 720,
    "days": 30,
    "weeks": 4.35,
    "months": 1,
}


def runs_per_month(wf: dict[str, Any]) -> float | None:
    """Sum over the workflow's schedule triggers; None if nothing is scheduled (webhook/manual/chat triggers)."""
    total, found = 0.0, False
    for node in wf.get("nodes") or []:
        if node.get("type") != "n8n-nodes-base.scheduleTrigger" or node.get("disabled"):
            continue
        for rule in ((node.get("parameters") or {}).get("rule") or {}).get("interval") or []:
            found = True
            field = rule.get("field") or "days"
            if field == "cronExpression":
                total += _cron_per_month(str(rule.get("expression") or ""))
                continue
            every = float(rule.get(f"{field}Interval") or 1)
            per = _PER_MONTH.get(field, 30) / max(every, 1.0)
            if field == "weeks":
                per *= max(len(rule.get("triggerAtDay") or [1]), 1)
            total += per
    return round(total, 1) if found else None


def _cron_field(expr: str, span: int) -> float:
    """How many values a cron field takes within its span ('*', '*/n', 'a,b', 'a-b', single)."""
    if expr in ("*", "?"):
        return span
    if expr.startswith("*/"):
        return span / max(int(expr[2:] or 1), 1)
    count = 0
    for part in expr.split(","):
        lo, _, hi = part.partition("-")
        count += (int(hi) - int(lo) + 1) if hi and lo.isdigit() and hi.isdigit() else 1
    return count


def _cron_per_month(expr: str) -> float:
    parts = expr.split()
    if len(parts) == 6:  # with seconds
        parts = parts[1:]
    if len(parts) != 5:
        return 30.0
    minute, hour, dom, _month, dow = parts
    per_day = _cron_field(minute, 60) * _cron_field(hour, 24)
    days = 4.35 * _cron_field(dow, 7) if dow not in ("*", "?") else _cron_field(dom, 30)
    return per_day * days
