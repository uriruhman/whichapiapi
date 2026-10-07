"""Measured spend (ADR-0008): read the channel's own accounting.

new-api resellers expose a per-call log for a key (`/api/log/token?key=`): charged quota, the group the call was
routed to and its price ratio. That gives *exact* per-call cost. The cumulative token counter can lag
(batch updates), so the log is preferred. OpenRouter exposes cumulative key usage only.
"""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel


class CallLogEntry(BaseModel):
    ts: int
    model: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    group: str | None = None
    group_ratio: float | None = None
    use_time_s: float | None = None
    retries: int = 0
    request_id: str | None = None
    cached_tokens: int = 0


class BalanceProbe:
    """Returns cumulative USD spent on this key, as the channel reports it."""

    def __init__(self, kind: str, base_url: str, api_key: str, client: httpx.Client | None = None):
        self.kind = kind
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.key = api_key
        self.client = client or httpx.Client(timeout=20)
        self._quota_per_unit: float | None = None

    def used_usd(self) -> float | None:
        try:
            if self.kind == "newapi":
                return self._newapi()
            if self.kind == "openrouter":
                r = self.client.get(
                    "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {self.key}"}
                )
                r.raise_for_status()
                return float(r.json()["data"]["usage"])
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            return None
        return None

    @property
    def has_call_log(self) -> bool:
        return self.kind == "newapi"

    def calls_since(self, ts: float, model: str | None = None, max_pages: int = 20) -> list[CallLogEntry]:
        """Consume-log entries at or after `ts` (unix seconds), newest first. Empty list on any error."""
        if self.kind != "newapi":
            return []
        out: list[CallLogEntry] = []
        try:
            qpu = self._qpu()
            for page in range(1, max_pages + 1):
                r = self.client.get(
                    self.root + "/api/log/token",
                    params={"key": self.key, "p": page, "page_size": 100, "start_timestamp": int(ts)},
                    headers={"Authorization": f"Bearer {self.key}"},
                )
                r.raise_for_status()
                items = r.json()["data"]["items"] or []
                for it in items:
                    if it.get("type") != 2 or (model and it.get("model_name") != model):
                        continue
                    other = json.loads(it.get("other") or "{}")
                    out.append(
                        CallLogEntry(
                            ts=it["created_at"],
                            model=it["model_name"],
                            cost_usd=it["quota"] / qpu,
                            prompt_tokens=it.get("prompt_tokens") or 0,
                            completion_tokens=it.get("completion_tokens") or 0,
                            group=it.get("group"),
                            group_ratio=other.get("group_ratio"),
                            use_time_s=it.get("use_time"),
                            retries=other.get("retry_count") or 0,
                            request_id=other.get("request_id"),
                            cached_tokens=other.get("cache_tokens") or 0,
                        )
                    )
                if len(items) < 100:
                    break
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            return out
        return out

    def remaining_usd(self) -> float | None:
        if self.kind != "newapi":
            return None
        try:
            d = self._token_usage()
            return None if d.get("unlimited_quota") else d["total_available"] / self._qpu()
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            return None

    def _qpu(self) -> float:
        if self._quota_per_unit is None:
            try:
                r = self.client.get(self.root + "/api/status")
                self._quota_per_unit = float(r.json()["data"].get("quota_per_unit") or 500000)
            except (httpx.HTTPError, KeyError, ValueError):
                self._quota_per_unit = 500000.0
        return self._quota_per_unit

    def _token_usage(self) -> dict:
        r = self.client.get(self.root + "/api/usage/token/", headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return r.json()["data"]

    def _newapi(self) -> float:
        return self._token_usage()["total_used"] / self._qpu()
