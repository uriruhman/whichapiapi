"""Point coding agents at the router: write their config (`setup`), start them with it injected (`launch`), and check
where each one really sends its requests (`doctor`). Patterned on freellmapi's CLI (MIT): timestamped backups,
0600 files for secrets, a unified diff with `dry_run`, and the key kept in env where the tool supports it.

| tool | file | what is written |
|---|---|---|
| claude | `<project>/.claude/settings.local.json` (default) or `~/.claude/settings.json` (`global_`) | `env`: ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL + the haiku/sonnet/opus defaults |
| codex | `~/.codex/config.toml` | a marked block: `[model_providers.whichapiapi]` (wire_api = "responses", env_key) + `[profiles.whichapiapi]` → `codex --profile whichapiapi` |
| aider | `~/.aider.conf.yml` | openai-api-base, openai-api-key, model `openai/<model>` |
| opencode | `~/.config/opencode/opencode.json` | provider `whichapiapi` (@ai-sdk/openai-compatible) with the auto models |
| continue | `~/.continue/config.yaml` | a model entry (provider openai, apiBase) |
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

TOOLS = ("claude", "codex", "aider", "opencode", "continue")
MODELS = {"main": "auto:coding", "fast": "auto:general@fast", "best": "auto:coding@best"}
KEY_ENV = "WHICHAPIAPI_KEY"
_MARK_START, _MARK_END = "# whichapiapi:start", "# whichapiapi:end"


def _root(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def target_path(tool: str, project: Path | None = None, global_: bool = False) -> Path:
    home = Path.home()
    if tool == "claude":
        return (
            home / ".claude/settings.json"
            if global_
            else (project or Path.cwd()) / ".claude/settings.local.json"
        )
    return {
        "codex": home / ".codex/config.toml",
        "aider": home / ".aider.conf.yml",
        "opencode": home / ".config/opencode/opencode.json",
        "continue": home / ".continue/config.yaml",
    }[tool]


def render(tool: str, current: str, url: str, key: str, model: str = MODELS["main"]) -> str:
    """The tool's config file content after adding the router, given its current content ('' if none)."""
    base = _root(url)
    if tool == "claude":
        data = json.loads(current) if current.strip() else {}
        env = data.setdefault("env", {})
        env.update({
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": MODELS["best"],
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODELS["fast"],
        })  # fmt: skip
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if tool == "codex":
        block = "\n".join([
            _MARK_START,
            "[model_providers.whichapiapi]",
            'name = "Which API API"',
            f'base_url = "{base}/v1"',
            f'env_key = "{KEY_ENV}"',
            'wire_api = "responses"',
            "",
            "[profiles.whichapiapi]",
            f'model = "{model}"',
            'model_provider = "whichapiapi"',
            _MARK_END,
        ])  # fmt: skip
        rest = re.sub(
            rf"\n?{re.escape(_MARK_START)}.*?{re.escape(_MARK_END)}\n?", "\n", current, flags=re.S
        ).rstrip()
        return (rest + "\n\n" if rest else "") + block + "\n"
    if tool == "aider":
        data = yaml.safe_load(current) or {} if current.strip() else {}
        data.update({"openai-api-base": f"{base}/v1", "openai-api-key": key, "model": f"openai/{model}"})
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if tool == "opencode":
        data = json.loads(current) if current.strip() else {"$schema": "https://opencode.ai/config.json"}
        data.setdefault("provider", {})["whichapiapi"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Which API API",
            "options": {"baseURL": f"{base}/v1", "apiKey": key},
            "models": {m: {"name": m} for m in dict.fromkeys([model, *MODELS.values()])},
        }
        data["model"] = f"whichapiapi/{model}"
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if tool == "continue":
        data = (
            yaml.safe_load(current) or {}
            if current.strip()
            else {"name": "local", "version": "1.0.0", "schema": "v1"}
        )
        models = [m for m in data.get("models") or [] if m.get("name") != "Which API API"]
        models.insert(0, {"name": "Which API API", "provider": "openai", "model": model, "apiBase": f"{base}/v1",
                          "apiKey": key, "roles": ["chat", "edit", "apply"]})  # fmt: skip
        data["models"] = models
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    raise ValueError(f"unknown tool {tool!r}; one of {', '.join(TOOLS)}")


def setup(
    tool: str,
    url: str,
    key: str,
    model: str = MODELS["main"],
    dry_run: bool = False,
    project: Path | None = None,
    global_: bool = False,
) -> dict[str, Any]:
    """Write (or with `dry_run` only diff) the router into a tool's config. Backs the old file up next to it."""
    path = target_path(tool, project, global_)
    current = path.read_text() if path.exists() else ""
    new = render(tool, current, url, key, model)
    diff = "".join(
        difflib.unified_diff(
            _redact(current, key).splitlines(True), _redact(new, key).splitlines(True), str(path), str(path)
        )
    )
    out: dict[str, Any] = {"tool": tool, "path": str(path), "changed": new != current, "diff": diff}
    if dry_run or new == current:
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.name}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        os.chmod(backup, 0o600)
        out["backup"] = str(backup)
    path.write_text(new)
    os.chmod(path, 0o600)
    if tool == "codex":
        out["next"] = f"export {KEY_ENV}=<your key>; codex --profile whichapiapi"
    return out


def _redact(text: str, key: str) -> str:
    return text.replace(key, key[:5] + "…") if key and len(key) > 8 else text


def launch_env(
    tool: str, url: str, key: str, model: str = MODELS["main"]
) -> tuple[dict[str, str], list[str]]:
    """(environment, extra argv) to start `tool` against the router without touching any config file."""
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    base = _root(url)
    if tool == "claude":
        env.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": MODELS["best"],
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODELS["fast"]})  # fmt: skip
        return env, []
    if tool == "codex":
        env[KEY_ENV] = key
        args = ["-c", 'model_provider="whichapiapi"', "-c", f'model="{model}"',
                "-c", 'model_providers.whichapiapi.name="Which API API"',
                "-c", f'model_providers.whichapiapi.base_url="{base}/v1"',
                "-c", f'model_providers.whichapiapi.env_key="{KEY_ENV}"',
                "-c", 'model_providers.whichapiapi.wire_api="responses"']  # fmt: skip
        return env, args
    if tool == "aider":
        env.update({"OPENAI_API_BASE": f"{base}/v1", "OPENAI_API_KEY": key})
        return env, ["--model", f"openai/{model}"]
    raise ValueError(f"launch supports claude, codex, aider (got {tool!r})")


def _claude_layers(project: Path) -> list[tuple[str, str | None, str | None]]:
    """Where Claude Code takes ANTHROPIC_BASE_URL (and its token) from, highest precedence first."""
    layers: list[tuple[str, str | None, str | None]] = []
    managed = [Path("/etc/claude-code/managed-settings.json")]
    managed += sorted(Path("/etc/claude-code/managed-settings.d").glob("*.json"), reverse=True)
    files = [*managed, project / ".claude/settings.local.json", project / ".claude/settings.json",
             Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "settings.json"]  # fmt: skip
    for f in files:
        if f.exists():
            try:
                env = json.loads(f.read_text()).get("env") or {}
            except (ValueError, AttributeError):
                env = {}
            layers.append((str(f), env.get("ANTHROPIC_BASE_URL"), env.get("ANTHROPIC_AUTH_TOKEN")))
    layers.append(
        ("process env", os.environ.get("ANTHROPIC_BASE_URL"), os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    )
    return layers


def _probe(url: str, key: str | None) -> str:
    """routed (answers with our routing policies), degraded (our router refuses the key), elsewhere, unreachable."""
    try:
        r = httpx.get(
            f"{_root(url)}/v1/models", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=10
        )
    except httpx.HTTPError:
        return "unreachable"
    if r.status_code == 200 and '"auto' in r.text:
        return "routed"
    return "degraded" if r.status_code in (401, 402, 403) else "elsewhere"


def _key_in(tool: str, text: str) -> str | None:
    m = re.search(r'(?:apiKey|api-key|openai-api-key|"apiKey")\W+([\w.-]{12,})', text)
    return m.group(1) if m else os.environ.get(KEY_ENV) if tool == "codex" else None


def doctor(router_url: str, project: Path | None = None) -> list[dict[str, Any]]:
    """Per tool: where its requests go — routed (our router answers with the configured key), degraded (our router
    but the key is refused), shadowed (we're configured in a lower layer that a higher one overrides), elsewhere,
    unreachable, unknown (not configured)."""
    project = project or Path.cwd()
    ours = _root(router_url)
    out = []
    layers = _claude_layers(project)
    winner = next(((layer, u, k) for layer, u, k in layers if u), None)
    mentioned = [layer for layer, u, _ in layers if u and _root(u) == ours]
    if winner is None:
        out.append(
            {"tool": "claude", "verdict": "unknown", "detail": "no ANTHROPIC_BASE_URL (Anthropic's own API)"}
        )
    elif _root(winner[1]) == ours:
        out.append({"tool": "claude", "verdict": _probe(ours, winner[2]), "detail": f"from {winner[0]}"})
    elif mentioned:
        out.append(
            {"tool": "claude", "verdict": "shadowed", "detail": f"{winner[0]} overrides {mentioned[0]}"}
        )
    else:
        out.append({"tool": "claude", "verdict": "elsewhere", "detail": f"{winner[1]} from {winner[0]}"})
    for tool in ("codex", "aider", "opencode", "continue"):
        path = target_path(tool)
        text = path.read_text() if path.exists() else ""
        if not text:
            out.append({"tool": tool, "verdict": "unknown", "detail": f"no {path}"})
        elif ours in text:
            out.append({"tool": tool, "verdict": _probe(ours, _key_in(tool, text)), "detail": str(path)})
        else:
            out.append({"tool": tool, "verdict": "elsewhere", "detail": str(path)})
    return out
