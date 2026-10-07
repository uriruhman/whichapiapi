"""Channels the router, evals and dashboards can call: a few official APIs built in, plus your own custom providers
(resellers and gateways with their own pricing) described outside the repo in a YAML file —
`$WHICHAPIAPI_CHANNELS` or `~/.config/whichapiapi/channels.yaml`:

```yaml
myreseller:                       # the channel name used in "channel:model"
  base_url: https://api.example.com/v1
  key_env: MYRESELLER_API_KEY     # the key is read from this env variable, never from the file
  kind: newapi                    # new-api / one-api: price groups, per-call log, balance (optional)
  primary: true                   # default channel for judges, balance forecast and the dashboard (optional)
  judge_model: some-model         # the judge used by generated suites on this channel (optional)
  min_balance: 0.30               # evals stop when the balance would drop under this (optional)
```

The default judge is `$WHICHAPIAPI_JUDGE`, else `<primary>:<judge_model>`, else an OpenRouter model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BUILTIN: dict[str, dict[str, str]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "deepinfra": {"base_url": "https://api.deepinfra.com/v1/openai", "key_env": "DEEPINFRA_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY"},
}


def _path() -> Path:
    return Path(os.environ.get("WHICHAPIAPI_CHANNELS") or Path.home() / ".config/whichapiapi/channels.yaml")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict) and v.get("base_url")}


def custom() -> dict[str, dict[str, Any]]:
    """Your custom channels (empty when there is no channels file)."""
    p = _path()
    return _load(str(p), p.stat().st_mtime) if p.exists() else {}


def all_channels() -> dict[str, dict[str, Any]]:
    return {**BUILTIN, **custom()}


def key_env(channel: str) -> str | None:
    return (all_channels().get(channel) or {}).get("key_env")


def base_url(channel: str) -> str | None:
    url = (all_channels().get(channel) or {}).get("base_url")
    return url.rstrip("/") if url else None


def root_url(channel: str) -> str | None:
    """The base URL without the API version suffix (new-api's /api/* endpoints live there)."""
    url = base_url(channel)
    return url.removesuffix("/v1") if url else None


def primary() -> str | None:
    """The custom channel marked `primary` (else the first custom one)."""
    cs = custom()
    return next((n for n, c in cs.items() if c.get("primary")), next(iter(cs), None))


def newapi_channels() -> list[str]:
    return [n for n, c in custom().items() if c.get("kind") == "newapi"]


def with_key() -> list[str]:
    """Channels whose key env variable is set: custom ones first (they're why you configured them)."""
    out = [n for n, c in custom().items() if os.environ.get(c.get("key_env") or "")]
    return out + [n for n, c in BUILTIN.items() if n not in out and os.environ.get(c["key_env"])]


def suite_channel(channel: str) -> dict[str, Any] | None:
    """The `x-whichapiapi.channels` entry a suite needs for `channel` (None when unknown)."""
    c = all_channels().get(channel)
    if not c:
        return None
    out: dict[str, Any] = {"base_url": c["base_url"], "key_env": c["key_env"]}
    if c.get("kind") == "newapi":
        out["balance"] = "newapi"
    elif channel == "openrouter":
        out["balance"] = "openrouter"
    if c.get("min_balance") is not None:
        out["min_balance"] = c["min_balance"]
    return out


def default_judge() -> str:
    if os.environ.get("WHICHAPIAPI_JUDGE"):
        return os.environ["WHICHAPIAPI_JUDGE"]
    p = primary()
    model = (custom().get(p) or {}).get("judge_model") if p else None
    return f"{p}:{model}" if p and model else "openrouter:openai/gpt-5-mini"


def channel_of_host(host: str) -> str | None:
    """A custom channel whose base URL is on `host` (for audits of HTTP calls in automations)."""
    from urllib.parse import urlparse

    for name, c in custom().items():
        if urlparse(c["base_url"]).hostname == host:
            return name
    return None
