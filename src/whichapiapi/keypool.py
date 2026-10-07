"""Your own provider keys (several per provider), encrypted at rest, rotated by the router — borrowed from freellmapi
(MIT): AES-256-GCM with a key file (`$WHICHAPIAPI_HOME/.encryption-key`, 0600, or WHICHAPIAPI_ENCRYPTION_KEY as 64 hex
chars), provider detection from env-variable names when importing a `.env`/JSON file, round-robin per provider,
a key that keeps failing auth (3 × 401/403) is disabled. Mostly for free tiers (Groq, Google AI Studio, Cerebras,
Mistral, NVIDIA, …) so the router can use them before paid channels; any OpenAI-compatible provider works.

Store: `$WHICHAPIAPI_HOME/provider-keys.json` (0600, file-locked), values never written in clear.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

# OpenAI-compatible base URLs of providers with free tiers (and a few paid ones people bring keys for)
PROVIDERS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "cohere": "https://api.cohere.ai/compatibility/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "github": "https://models.github.ai/inference",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "modelscope": "https://api-inference.modelscope.cn/v1",
    "ovh": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
}
_PREFIX = [  # env-variable name → provider (longest match first)
    ("GOOGLE_GENERATIVE_AI", "google"), ("GEMINI", "google"), ("GOOGLE", "google"), ("GROQ", "groq"),
    ("CEREBRAS", "cerebras"), ("MISTRAL", "mistral"), ("NVIDIA", "nvidia"), ("NIM", "nvidia"),
    ("OPENROUTER", "openrouter"), ("HUGGINGFACE", "huggingface"), ("HF", "huggingface"), ("COHERE", "cohere"),
    ("SAMBANOVA", "sambanova"), ("GITHUB", "github"), ("DEEPINFRA", "deepinfra"), ("TOGETHER", "together"),
    ("DEEPSEEK", "deepseek"), ("OPENAI", "openai"), ("ZHIPU", "zhipu"), ("GLM", "zhipu"), ("MODELSCOPE", "modelscope"),
    ("OVH", "ovh"),
]  # fmt: skip
AUTH_FAILS_TO_DISABLE = 3
_rr: dict[str, int] = {}


def _home() -> Path:
    h = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    h.mkdir(parents=True, exist_ok=True, mode=0o700)
    return h


def _master() -> bytes:
    env = os.environ.get("WHICHAPIAPI_ENCRYPTION_KEY")
    if env:
        return bytes.fromhex(env)
    f = _home() / ".encryption-key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
        os.chmod(f, 0o600)
    return bytes.fromhex(f.read_text().strip())


def _seal(value: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    return base64.b64encode(nonce + AESGCM(_master()).encrypt(nonce, value.encode(), None)).decode()


def _open(blob: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = base64.b64decode(blob)
    return AESGCM(_master()).decrypt(raw[:12], raw[12:], None).decode()


@contextlib.contextmanager
def _locked():
    p = _home() / "provider-keys.json"
    with open(p.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(p.read_text()) if p.exists() else {"keys": []}
        yield data
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(p)


def _read() -> dict[str, Any]:
    p = _home() / "provider-keys.json"
    return json.loads(p.read_text()) if p.exists() else {"keys": []}


def add(provider: str, key: str, label: str = "", base_url: str | None = None) -> dict[str, Any]:
    """Store a key (deduplicated by provider + value). `base_url` for providers not in PROVIDERS."""
    provider = provider.lower()
    if provider not in PROVIDERS and not base_url:
        raise ValueError(
            f"unknown provider {provider!r}: pass base_url, or one of {', '.join(sorted(PROVIDERS))}"
        )
    key = key.strip()
    with _locked() as data:
        for k in data["keys"]:
            if k["provider"] == provider and _open(k["secret"]) == key:
                return _public(k)
        rec = {"id": secrets.token_hex(4), "provider": provider, "label": label or f"{provider}-{len(data['keys']) + 1}",
               "secret": _seal(key), "hint": key[:4] + "…" + key[-3:], "base_url": base_url, "enabled": True,
               "auth_fails": 0, "added_at": time.time()}  # fmt: skip
        data["keys"].append(rec)
        return _public(rec)


def _public(k: dict[str, Any]) -> dict[str, Any]:
    return {
        x: k.get(x)
        for x in ("id", "provider", "label", "hint", "base_url", "enabled", "auth_fails", "last_error")
    }


def listing() -> list[dict[str, Any]]:
    return [_public(k) for k in _read()["keys"]]


def remove(key_id: str) -> bool:
    with _locked() as data:
        before = len(data["keys"])
        data["keys"] = [k for k in data["keys"] if k["id"] != key_id]
        return len(data["keys"]) < before


def providers() -> set[str]:
    """Providers with at least one enabled key."""
    return {k["provider"] for k in _read()["keys"] if k.get("enabled")}


def base_url(provider: str) -> str | None:
    for k in _read()["keys"]:
        if k["provider"] == provider and k.get("base_url"):
            return k["base_url"]
    return PROVIDERS.get(provider)


def pick(provider: str) -> tuple[str, str] | None:
    """(key id, secret) for the next request to `provider`, round-robin over its enabled keys."""
    keys = [k for k in _read()["keys"] if k["provider"] == provider and k.get("enabled")]
    if not keys:
        return None
    i = _rr.get(provider, 0) % len(keys)
    _rr[provider] = i + 1
    return keys[i]["id"], _open(keys[i]["secret"])


def report(key_id: str, status: int, text: str = "") -> None:
    """Feed back an answer: auth failures count toward disabling the key; a success resets the count."""
    if status not in (401, 403) and status >= 400:
        return
    with _locked() as data:
        for k in data["keys"]:
            if k["id"] != key_id:
                continue
            if status in (401, 403):
                k["auth_fails"] = k.get("auth_fails", 0) + 1
                k["last_error"] = f"{status} {text[:80]}"
                if k["auth_fails"] >= AUTH_FAILS_TO_DISABLE:
                    k["enabled"] = False
            elif k.get("auth_fails"):
                k["auth_fails"] = 0


def detect(name: str) -> str | None:
    """Provider from an env-variable name: GROQ_API_KEY → groq, GEMINI_API_KEY → google."""
    up = name.upper()
    for prefix, prov in _PREFIX:
        if up == prefix or up.startswith(prefix + "_"):
            return prov
    return None


def parse_file(text: str) -> list[tuple[str, str]]:
    """(provider, key) pairs from a .env file (`NAME=value`, comments, quotes, `export`) or a JSON object/list."""
    pairs: list[tuple[str, str]] = []
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        data = json.loads(stripped)
        items = data.items() if isinstance(data, dict) else ((d.get("provider"), d.get("key")) for d in data)
        for name, val in items:
            if not isinstance(val, str) or not name:
                continue
            prov = name.lower() if name.lower() in PROVIDERS else detect(name)
            if prov:
                pairs.append((prov, val))
        return pairs
    for line in text.splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m or not re.search(r"KEY|TOKEN", m.group(1).upper()):
            continue
        val = re.sub(r"\s+#.*$", "", m.group(2)).strip().strip("'\"")
        prov = detect(m.group(1))
        if prov and len(val) >= 8:
            pairs.append((prov, val))
    return pairs
