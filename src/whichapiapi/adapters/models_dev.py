"""models.dev open database (MIT): ~225 providers x models with prices. Seed for public offers (ADR-0005)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from whichapiapi.schema.capability import infer_capability
from whichapiapi.schema.offer import Endpoint, Offer, Price, Provenance

API = "https://models.dev/api.json"
# Routers/marketplaces that resell other vendors' models. Everything else is treated as selling directly.
AGGREGATORS = {
    "openrouter",
    "requesty",
    "nano-gpt",
    "poe",
    "aihubmix",
    "vercel",
    "helicone",
    "zenmux",
    "302ai",
    "github-models",
    "fastrouter",
    "inference",
    "llmgateway",
    "cloudflare-ai-gateway",
    "kilo",
    "bothub",
}


FIRST_PARTY_ADAPTERS = {"openrouter", "deepinfra", "huggingface"}


def is_aggregator(provider_id: str) -> bool:
    pid = provider_id.lower()
    return pid in AGGREGATORS or any(t in pid for t in ("gateway", "router", "-providers", "hub"))


class ModelsDevAdapter:
    id = "models_dev"
    license = "MIT"
    attribution = "Model and price data from models.dev (MIT)"

    def __init__(self, client: httpx.Client | None = None, url: str = API):
        self.client = client or httpx.Client(timeout=60)
        self.url = url

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(self.url)
        r.raise_for_status()
        return list(self.parse(r.json()))

    def parse(self, data: dict[str, Any]) -> Iterable[Offer]:
        for pid, prov in data.items():
            if pid in FIRST_PARTY_ADAPTERS:
                continue  # a dedicated adapter reads this channel directly (fresher, per-upstream detail)
            api = prov.get("api")
            npm = str(prov.get("npm") or "")
            protocol = "openai" if (api and "openai" in npm) or pid == "openai" else "none"
            key_env = (prov.get("env") or [None])[0]
            for mid, m in (prov.get("models") or {}).items():
                cost = m.get("cost") or {}
                if "input" not in cost and "output" not in cost:
                    continue
                limit = m.get("limit") or {}
                feats = [
                    f
                    for f, k in (
                        ("tools", "tool_call"),
                        ("json_schema", "structured_output"),
                        ("reasoning", "reasoning"),
                    )
                    if m.get(k)
                ]
                mods = m.get("modalities") or {}
                if "image" in mods.get("input", []):
                    feats.append("vision")
                yield Offer(
                    capability=infer_capability(mid, mods.get("input"), mods.get("output")),
                    model=mid,
                    vendor=m.get("family") or (mid.split("/")[0] if "/" in mid else None),
                    channel=pid,
                    channel_type="aggregator" if is_aggregator(pid) else "official",
                    price=Price(
                        input_per_1m=cost.get("input"),
                        output_per_1m=cost.get("output"),
                        cache_read_per_1m=cost.get("cache_read"),
                        cache_write_per_1m=cost.get("cache_write"),
                    ),
                    endpoint=Endpoint(protocol=protocol, base_url=api, model_id=mid, key_env=key_env),  # type: ignore[arg-type]
                    context_window=limit.get("context"),
                    max_output=limit.get("output"),
                    features=feats,
                    provenance=Provenance(source=f"{API}#{pid}/{mid}", confidence="medium"),
                )
