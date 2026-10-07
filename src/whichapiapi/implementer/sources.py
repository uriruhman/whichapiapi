"""Audit inputs besides n8n: Make (Integromat) scenario blueprints and plain source code. Both produce the same
`ApiUse` records as the n8n extractor, so `implementer.audit` treats them alike. Read-only and pure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from whichapiapi.channels import channel_of_host
from whichapiapi.implementer.n8n import _HOSTS, _KEYLIKE, _PATHS, ApiUse, Extraction, SecretLeak

# Make app modules ("app:Action") that call AI models -> (capability, provider)
_MAKE_APPS = {
    "openai-gpt-3": ("llm.chat", "openai"),
    "openai": ("llm.chat", "openai"),
    "anthropic-claude": ("llm.chat", "anthropic"),
    "gemini-ai": ("llm.chat", "google"),
    "google-gemini-ai": ("llm.chat", "google"),
    "mistral-ai": ("llm.chat", "mistral"),
    "perplexity-ai": ("llm.chat", "perplexity"),
    "deepseek-ai": ("llm.chat", "deepseek"),
    "openrouter": ("llm.chat", "openrouter"),
}


def _url_use(url: str) -> tuple[str, str] | None:
    host = urlparse(url).hostname or ""
    provider = _HOSTS.get(host) or channel_of_host(host)
    if provider is None:
        return None
    path = urlparse(url).path
    return next((cap for p, cap in _PATHS if p in path), "llm.chat"), provider


def extract_make(data: Any) -> Extraction:
    """Make scenario blueprint JSON (Scenario → ⋯ → Export blueprint): walks `flow`, including router `routes`."""
    bp = json.loads(data) if isinstance(data, str) else data
    name = bp.get("name") or "Make scenario"
    out = Extraction(workflows=1)

    def walk(flow: list[dict[str, Any]]) -> None:
        for mod in flow or []:
            app, _, action = str(mod.get("module") or "").partition(":")
            mapper = mod.get("mapper") or {}
            label = (mod.get("metadata") or {}).get("designer", {}).get(
                "name"
            ) or f"{app}:{action} #{mod.get('id')}"
            hit = _MAKE_APPS.get(app)
            if app == "http" and (found := _url_use(str(mapper.get("url") or ""))):
                hit = found
            if hit:
                model = mapper.get("model") or mapper.get("modelId")
                if not model and (m := re.search(r'"model"\s*:\s*"([^"]+)"', str(mapper.get("data") or ""))):
                    model = m.group(1)
                if "transcri" in action.lower():
                    hit = ("speech.stt", hit[1])
                out.uses.append(
                    ApiUse(
                        workflow=name,
                        node=label,
                        node_type=str(mod.get("module")),
                        capability=hit[0],
                        provider=hit[1],
                        model=str(model) if model and "{{" not in str(model) else None,
                        model_is_expression=bool(model) and "{{" in str(model),
                        continues_on_error=bool(mod.get("onerror")),
                    )
                )
            for text in re.findall(_KEYLIKE, json.dumps(mapper)):
                out.secrets.append(
                    SecretLeak(workflow=name, node=label, field="mapper", preview=f"{text[:6]}…{text[-3:]}")
                )
            for route in mod.get("routes") or []:
                walk(route.get("flow") or [])

    walk(bp.get("flow") or [])
    return out


# source code: an AI endpoint URL or SDK constructor, and model literals near it
_CODE_EXT = {".py", ".js", ".ts", ".mjs", ".cjs", ".jsx", ".tsx", ".go", ".rb", ".php", ".java", ".kt", ".cs"}
_SDKS = {
    r"\bOpenAI\(|from openai import|require\(['\"]openai['\"]\)|from ['\"]openai['\"]": "openai",
    r"\bAnthropic\(|import anthropic|@anthropic-ai/sdk": "anthropic",
    r"google\.generativeai|@google/genai|from google import genai": "google",
    r"\bGroq\(|from groq import": "groq",
}
_URL = re.compile(r"""https?://[^\s'"`)]+""")
_MODEL = re.compile(r"""\bmodel(?:_name|Id)?\s*[:=]\s*['"]([\w.:/@-]{3,})['"]""")


def extract_code(root: str | Path, max_files: int = 2000) -> Extraction:
    """Scan source files under `root` (or one file) for AI API calls: known endpoint URLs or SDK imports, plus
    `model=...` literals in the same file. One use per (file, model); `base_url=` pointing at another host wins."""
    root = Path(root)
    files = (
        [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix in _CODE_EXT and p.is_file()]
    )
    skip = ("node_modules", ".venv", "venv", "site-packages", "dist", "build", ".git")
    out = Extraction()
    for path in files[:max_files]:
        if any(part in skip for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        providers: dict[str, str] = {}
        for url in _URL.findall(text):
            if found := _url_use(url):
                providers.setdefault(found[1], found[0])
        for pattern, provider in _SDKS.items():
            if re.search(pattern, text) and not providers:
                providers[provider] = "llm.chat"
        for m in _KEYLIKE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.secrets.append(
                SecretLeak(
                    workflow=rel,
                    node=f"line {line}",
                    field="source",
                    preview=f"{m.group(0)[:6]}…{m.group(0)[-3:]}",
                )
            )
        if not providers:
            continue
        out.workflows += 1
        models = {m.group(1): text.count("\n", 0, m.start()) + 1 for m in _MODEL.finditer(text)}
        provider, capability = next(iter(providers.items()))
        for model, line in (models or {None: 1}).items():
            out.uses.append(
                ApiUse(
                    workflow=rel,
                    node=f"line {line}",
                    node_type="code",
                    capability=capability,
                    provider=provider,
                    model=model,
                    retry=bool(re.search(r"max_retries|maxRetries|retry|backoff|tenacity", text)),
                )
            )
    return out
