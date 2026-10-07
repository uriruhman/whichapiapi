"""new-api / one-api resellers (custom providers with their own pricing). Public `/api/pricing` + `/api/status`, no key needed.

Verified on a live new-api reseller 2026-09-28 against official list prices (ADR-0005):
    usd_per_1M_input  = model_ratio * 2 * group_ratio
    usd_per_1M_output = usd_per_1M_input * completion_ratio
    usd_per_1M_cached = usd_per_1M_input * cache_ratio
`quota_type == 1` means a fixed price per request: model_price * group_ratio.
We only read the HTTP API; no new-api (AGPL) code is used.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx

from whichapiapi.schema.capability import infer_capability
from whichapiapi.schema.offer import Billing, Endpoint, Offer, Price, Provenance

RATIO_USD_PER_1M = 2.0


class NewApiAdapter:
    id = "newapi"
    license = "reseller public API"
    attribution = "Prices from the reseller's public /api/pricing endpoint"

    def __init__(
        self,
        base_url: str,
        channel: str,
        *,
        groups: list[str] | None = None,
        key_env: str | None = None,
        visibility: str = "user",
        client: httpx.Client | None = None,
    ):
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.channel = channel
        self.groups = groups
        self.key_env = key_env
        self.visibility = visibility
        self.client = client or httpx.Client(timeout=30)

    def _get(self, path: str) -> dict[str, Any]:
        r = self.client.get(self.root + path)
        r.raise_for_status()
        return r.json()

    def fetch(self) -> Iterable[Offer]:
        pricing = self._get("/api/pricing")
        try:
            status = self._get("/api/status").get("data", {})
        except httpx.HTTPError:
            status = {}
        return list(self.parse(pricing, status))

    def parse(self, pricing: dict[str, Any], status: dict[str, Any]) -> Iterable[Offer]:
        group_ratio: dict[str, float] = pricing.get("group_ratio") or {}
        vendors = {v.get("id"): v.get("name") for v in pricing.get("vendors") or []}
        billing = self._billing(status)
        src = f"{self.root}/api/pricing"
        for m in pricing.get("data") or []:
            name = m["model_name"]
            groups = [g for g in m.get("enable_groups") or [] if self.groups is None or g in self.groups]
            vendor = vendors.get(m.get("vendor_id"))
            # base offer at ratio 1: what a key in an unknown/auto group is *nominally* charged
            yield self._offer(m, name, None, 1.0, vendor, billing, src, "list price before group ratio")
            for g in groups:
                if g in group_ratio:
                    yield self._offer(m, name, g, float(group_ratio[g]), vendor, billing, src, None)

    def _offer(self, m, name, group, ratio, vendor, billing, src, note) -> Offer:
        if m.get("quota_type") == 1:
            price = Price(per_request=round(float(m.get("model_price") or 0) * ratio, 6))
        else:
            inp = float(m.get("model_ratio") or 0) * RATIO_USD_PER_1M * ratio
            price = Price(
                input_per_1m=round(inp, 6),
                output_per_1m=round(inp * float(m.get("completion_ratio") or 1), 6),
                cache_read_per_1m=round(inp * float(m["cache_ratio"]), 6) if m.get("cache_ratio") else None,
            )
        tags = str(m.get("tags") or "")
        features = [f for f, t in (("tools", "工具"), ("vision", "识图"), ("reasoning", "思考")) if t in tags]
        return Offer(
            capability=infer_capability(name),
            model=name,
            vendor=(vendor or "").lower() or None,
            channel=self.channel,
            channel_type="user_added",
            group=group,
            price=price,
            billing=billing,
            endpoint=Endpoint(protocol="openai", base_url=self.root + "/v1", key_env=self.key_env),
            features=features,
            provenance=Provenance(source=src, confidence="medium" if group else "low", note=note),
            visibility=self.visibility,  # type: ignore[arg-type]
            verified=False,
        )

    @staticmethod
    def _billing(status: dict[str, Any]) -> Billing:
        methods: list[str] = []
        raw = status.get("pay_methods")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        for pm in raw or []:
            if isinstance(pm, dict) and (pm.get("name") or pm.get("type")):
                methods.append(str(pm.get("name") or pm["type"]))
        if status.get("enable_stripe_topup"):
            methods.append("Stripe")
        return Billing(
            min_topup=status.get("min_topup"),
            payment_methods=sorted(set(methods)),
            currency=status.get("currency_type") or "USD",
            fx_note=(
                f"top-up price {status.get('price')} per $1 of quota"
                if status.get("price") not in (None, 1)
                else None
            ),
        )
