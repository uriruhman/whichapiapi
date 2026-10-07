"""Full traces for tester keys: every routed chat (messages, answer, route, tokens, cost, latency) and every hosted
MCP tool call (arguments, result) of a key with role "tester" is written to
`$WHICHAPIAPI_HOME/traces/<key name>/<YYYY-MM-DD>.jsonl` (dir 700, files 600), so the owner can see exactly how
testers use the project. Testers are told so on the docs page. Chat rows use the `{messages, output, model}` shape
that `audit traces-suite` reads, so tester traffic can become a benchmark directly. Guest keys are never traced
beyond the activity log (no content).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_FIELD = 20_000  # chars per text field; a trace is for reading, not an archive


def _dir(name: str) -> Path:
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    (home / "traces").mkdir(parents=True, exist_ok=True, mode=0o700)
    d = home / "traces" / "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    d.mkdir(exist_ok=True, mode=0o700)
    return d


def _cap(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value
            if len(value) <= MAX_FIELD
            else value[:MAX_FIELD] + f"… [{len(value) - MAX_FIELD} chars cut]"
        )
    if isinstance(value, list):
        return [_cap(v) for v in value[:200]]
    if isinstance(value, dict):
        return {k: _cap(v) for k, v in list(value.items())[:200]}
    return value


def write(name: str, record: dict[str, Any]) -> None:
    now = datetime.now(UTC)
    path = _dir(name) / f"{now:%Y-%m-%d}.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        row = {"ts": now.isoformat(timespec="milliseconds"), "key": name, **_cap(record)}
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read(name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Newest `limit` rows, of one key or of all."""
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi")) / "traces"
    files = sorted((home / name).glob("*.jsonl") if name else home.glob("*/*.jsonl"), key=lambda p: p.name)
    rows: list[dict[str, Any]] = []
    for f in reversed(files):
        for line in reversed(f.read_text(encoding="utf-8").splitlines()):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
            if len(rows) >= limit:
                return sorted(rows, key=lambda r: r.get("ts", ""))
    return sorted(rows, key=lambda r: r.get("ts", ""))
