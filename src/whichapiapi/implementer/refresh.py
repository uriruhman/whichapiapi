"""Leaderboard auto-refresh for your own benchmarks: when new models ship, add the current best ones for the suite's
task to the suite, re-run it (cached results make the old candidates free) and flag when the recommended primary
changes. Artificial Analysis Optima re-runs a custom benchmark on new models; this does the same for local suites,
limited to models your keys can buy and capped by the suite's budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from itertools import takewhile
from pathlib import Path
from typing import Any

import yaml

from whichapiapi import core
from whichapiapi.implementer.suite_gen import known_channels
from whichapiapi.schema.naming import model_key


def new_candidates(
    suite_path: Path, n: int = 4, preset: str = "optimal", store: Any = None
) -> dict[str, Any]:
    """{"task", "top", "new"}: today's best `n` variants for the suite's task (`x-whichapiapi.task`, default general)
    and those of them the suite doesn't have yet (same model on any channel counts as present)."""
    raw = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8")) or {}
    ext = raw.get("x-whichapiapi") or {}
    capability = ext.get("capability", "llm.chat")
    task = ext.get("task") or "general"
    if capability != "llm.chat":
        return {"task": task, "top": [], "new": [], "skipped": f"capability {capability} has no task ranking"}
    have = {model_key(_id(p).split(":", 1)[-1]) for p in raw.get("providers") or []}
    top = core.suggest_candidates(task, n=n, preset=preset, store=store)
    return {"task": task, "top": top, "new": [c for c in top if model_key(c.split(":", 1)[1]) not in have]}


def _id(p: Any) -> str:
    return p if isinstance(p, str) else str(p.get("id", ""))


def add_providers(suite_path: Path, ids: list[str]) -> list[str]:
    """Append `ids` to the suite's providers (and their channels if missing), with the current provider's `config`.
    Leading comment lines survive; other comments and YAML anchors don't (the suite is rewritten, aliases expanded).
    Returns channels that still need base_url/key_env."""
    path = Path(suite_path)
    text = path.read_text(encoding="utf-8")
    header = "".join(takewhile(lambda line: line.startswith("#"), text.splitlines(keepends=True)))
    raw = yaml.safe_load(text) or {}
    today = datetime.now(UTC).date().isoformat()
    providers = raw.setdefault("providers", [])
    dicts = [p for p in providers if isinstance(p, dict)]
    current = next(
        (p for p in dicts if (p.get("x-whichapiapi") or {}).get("current")), dicts[0] if dicts else {}
    )
    config = current.get("config")  # same prompt → same call settings (response_format, max_tokens, …)
    providers.extend(
        {"id": c, "label": c.split(":", 1)[1], **({"config": config} if config else {}),
         "x-whichapiapi": {"added_by_refresh": today}}
        for c in ids
    )  # fmt: skip
    channels = raw.setdefault("x-whichapiapi", {}).setdefault("channels", {})
    unknown = []
    for ch in sorted({c.split(":", 1)[0] for c in ids}):
        if ch not in channels:
            if known_channels().get(ch):
                channels[ch] = known_channels()[ch]
            else:
                unknown.append(ch)
    path.write_text(header + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return unknown


def _state(suite_path: Path) -> Path:
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    digest = hashlib.sha1(str(Path(suite_path).resolve()).encode(), usedforsecurity=False).hexdigest()[:12]
    return home / "refresh" / f"{digest}.json"


def record_winner(suite_path: Path, primary: str | None, fallback: str | None) -> dict[str, Any] | None:
    """Remember the recommendation; return the previous one when the primary changed (None on first run or no
    change)."""
    state = _state(suite_path)
    before = json.loads(state.read_text()) if state.exists() else None
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "suite": str(Path(suite_path).resolve()),
                "primary": primary,
                "fallback": fallback,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
    )
    return before if before and before.get("primary") != primary else None
