"""InferIndex public pricing cross-check API.

InferIndex provides an informational price sanity check and promo detection, not a
source of truth or a primary offer source. The API is public and unauthenticated,
with a rate limit of 60 requests per minute per IP.
"""

from __future__ import annotations

from typing import Any

import httpx


class InferIndexAdapter:
    id = "inferindex"
    attribution = (
        "Price cross-check via the InferIndex API (https://api.inferindex.dev, public, unauthenticated)"
    )

    def __init__(
        self,
        client: httpx.Client | None = None,
        base_url: str = "https://api.inferindex.dev",
    ):
        self.client = client or httpx.Client(timeout=60)
        self.base_url = base_url.rstrip("/")

    def cheapest(self, model: str) -> dict[str, Any] | None:
        r = self.client.get(
            f"{self.base_url}/cheapest",
            params={"model": model},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        cheapest = data.get("cheapest")
        if not isinstance(cheapest, dict) or not cheapest:
            return {}

        model_data = data.get("model")
        if not isinstance(model_data, dict):
            model_data = {}

        promo = bool(cheapest.get("promo", False))
        return {
            "resolved_model_id": model_data.get("id"),
            "resolved_model_name": model_data.get("name"),
            "provider": cheapest.get("provider"),
            "via": cheapest.get("via"),
            "blended_per_1m": cheapest.get("blended_per_1M"),
            "input_per_1m": cheapest.get("input_per_1M"),
            "output_per_1m": cheapest.get("output_per_1M"),
            "cache_read_per_1m": cheapest.get("cache_read_per_1M"),
            "currency": cheapest.get("currency"),
            "context_length": cheapest.get("context_length"),
            "promo": promo,
            "price_before_promo": cheapest.get("price_before_promo"),
            "promo_confidence": cheapest.get("confidence") if promo else None,
            "checked_at": cheapest.get("checked_at") or data.get("fetched_at"),
            "stale": bool(cheapest.get("stale", False)),
        }
