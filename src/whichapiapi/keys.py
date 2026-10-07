"""API keys for people the owner invites to the router (friends testing `/v1/chat/completions`).

Each key has a name, a USD limit and a running spend; only a SHA-256 of the key is stored, in
`$WHICHAPIAPI_HOME/api-keys.json` (mode 600, file-locked so the internal and the public server can share it). The key
itself is shown once, at creation. Role "tester" (default) also gets the hosted MCP tools and full traces of what
it sends and gets back (see tracing.py); role "guest" only the router, with no content logged. Spend is the list price of the route that answered × the learned real/list ratio,
charged after each call; a key at its limit gets HTTP 402.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFIX = "wa-"


def _path() -> Path:
    home = Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
    return home / "api-keys.json"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@contextmanager
def _locked(write: bool = False) -> Iterator[dict[str, Any]]:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        text = f.read()
        data: dict[str, Any] = json.loads(text) if text.strip() else {}
        yield data
        if write:
            f.seek(0)
            f.truncate()
            f.write(json.dumps(data, indent=1))


ROLES = ("tester", "guest")


def add(name: str, limit_usd: float, role: str = "tester") -> str:
    """Create a key for `name` (replacing an old one of that name) and return it — the only time it's visible."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    key = PREFIX + secrets.token_urlsafe(24)
    with _locked(write=True) as data:
        prev = data.get(name) or {}
        data[name] = {
            "hash": _hash(key),
            "hint": key[:7] + "…",
            "limit_usd": float(limit_usd),
            "spent_usd": float(prev.get("spent_usd", 0.0)),
            "calls": int(prev.get("calls", 0)),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "revoked": False,
            "role": role,
        }
    return key


def find(key: str) -> tuple[str, dict[str, Any]] | None:
    """(name, record) for an active key, else None."""
    if not key.startswith(PREFIX):
        return None
    h = _hash(key)
    with _locked() as data:
        for name, rec in data.items():
            if hmac.compare_digest(rec.get("hash", ""), h) and not rec.get("revoked"):
                return name, rec
    return None


def remaining(rec: dict[str, Any]) -> float:
    return rec.get("limit_usd", 0.0) - rec.get("spent_usd", 0.0)


def charge(name: str, usd: float) -> None:
    with _locked(write=True) as data:
        if name in data:
            data[name]["spent_usd"] = round(data[name].get("spent_usd", 0.0) + max(usd, 0.0), 8)
            data[name]["calls"] = data[name].get("calls", 0) + 1
            data[name]["last_used"] = datetime.now(UTC).isoformat(timespec="seconds")


def set_limit(name: str, limit_usd: float) -> bool:
    with _locked(write=True) as data:
        if name not in data:
            return False
        data[name]["limit_usd"] = float(limit_usd)
        return True


def revoke(name: str) -> bool:
    with _locked(write=True) as data:
        if name not in data:
            return False
        data[name]["revoked"] = True
        return True


def list_keys() -> dict[str, dict[str, Any]]:
    with _locked() as data:
        return {n: {k: v for k, v in r.items() if k != "hash"} for n, r in data.items()}
