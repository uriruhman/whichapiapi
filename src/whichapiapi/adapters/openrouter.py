"""OpenRouter (aggregator). Public `/api/v1/models` (prices per token) and, per model, `/endpoints`
(price, uptime and quantization of every upstream provider). No key needed for either."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from whichapiapi.schema.capability import infer_capability
from whichapiapi.schema.offer import Endpoint, Offer, Price, Provenance, Reliability

API = "https://openrouter.ai/api/v1"
BASE_URL = "https://openrouter.ai/api/v1"


def _per_1m(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f < 0 else round(f * 1e6, 6)


def _positive(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _features(params: list[str], modalities: list[str]) -> list[str]:
    feats = []
    if "tools" in params:
        feats.append("tools")
    if "structured_outputs" in params or "response_format" in params:
        feats.append("json_schema")
    if "reasoning" in params or "reasoning_effort" in params:
        feats.append("reasoning")
    if "image" in modalities:
        feats.append("vision")
    return feats


class OpenRouterAdapter:
    id = "openrouter"
    license = "OpenRouter API terms"
    attribution = "Prices and uptime from OpenRouter's public API"

    def __init__(self, client: httpx.Client | None = None, endpoints_for: list[str] | None = None):
        self.client = client or httpx.Client(timeout=60)
        self.endpoints_for = endpoints_for or []  # model ids to expand into per-upstream offers

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(f"{API}/models")
        r.raise_for_status()
        offers = list(self.parse_models(r.json()))
        for mid in self.endpoints_for:
            er = self.client.get(f"{API}/models/{mid}/endpoints")
            if er.status_code == 200:
                offers.extend(self.parse_endpoints(er.json()))
        return offers

    def parse_models(self, data: dict[str, Any]) -> Iterable[Offer]:
        for m in data.get("data") or []:
            mid = m.get("id", "")
            if not mid or mid.startswith("~") or "/" not in mid:
                continue  # aliases ("latest") and routers
            p = m.get("pricing") or {}
            price = Price(
                input_per_1m=_per_1m(p.get("prompt")),
                output_per_1m=_per_1m(p.get("completion")),
                cache_read_per_1m=_per_1m(p.get("input_cache_read")),
                per_request=_positive(p.get("request")),
            )
            if price.input_per_1m is None and price.output_per_1m is None:
                continue
            arch = m.get("architecture") or {}
            top = m.get("top_provider") or {}
            yield Offer(
                capability=infer_capability(mid, arch.get("input_modalities"), arch.get("output_modalities")),
                model=mid,
                vendor=mid.split("/")[0],
                channel="openrouter",
                channel_type="aggregator",
                price=price,
                endpoint=Endpoint(
                    protocol="openai", base_url=BASE_URL, model_id=mid, key_env="OPENROUTER_API_KEY"
                ),
                context_window=m.get("context_length"),
                max_output=top.get("max_completion_tokens"),
                features=_features(m.get("supported_parameters") or [], arch.get("input_modalities") or []),
                provenance=Provenance(source=f"{API}/models#{mid}", confidence="high"),
            )

    def parse_endpoints(self, data: dict[str, Any]) -> Iterable[Offer]:
        d = data.get("data") or {}
        mid = d.get("id", "")
        for e in d.get("endpoints") or []:
            p = e.get("pricing") or {}
            tag = e.get("tag") or e.get("provider_name")
            uptime = e.get("uptime_last_1d")
            yield Offer(
                model=mid,
                vendor=mid.split("/")[0] if "/" in mid else None,
                channel="openrouter",
                channel_type="aggregator",
                group=tag,
                price=Price(
                    input_per_1m=_per_1m(p.get("prompt")),
                    output_per_1m=_per_1m(p.get("completion")),
                    cache_read_per_1m=_per_1m(p.get("input_cache_read")),
                ),
                reliability=Reliability(uptime_30d=round(uptime / 100, 4) if uptime is not None else None),
                endpoint=Endpoint(
                    protocol="openai", base_url=BASE_URL, model_id=mid, key_env="OPENROUTER_API_KEY"
                ),
                context_window=e.get("context_length"),
                max_output=e.get("max_completion_tokens"),
                features=_features(e.get("supported_parameters") or [], []),
                provenance=Provenance(
                    source=f"{API}/models/{mid}/endpoints#{tag}",
                    confidence="high",
                    note=f"upstream {e.get('provider_name')}, quantization {e.get('quantization') or 'unknown'}",
                ),
            )
