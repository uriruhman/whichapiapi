"""Hugging Face Inference Providers: one OpenAI-compatible router over many inference providers. Public
`GET /v1/models` (no key) lists every model with its live providers, their prices (USD per 1M tokens), context,
time to first token and throughput. One offer per live provider (group = provider) plus an ungrouped offer for the
router's `:cheapest` policy, approximated here as the lowest input+output price."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from whichapiapi.schema.capability import infer_capability
from whichapiapi.schema.offer import Endpoint, Offer, Price, Provenance, Reliability

API = "https://router.huggingface.co/v1"


class HFRouterAdapter:
    id = "huggingface"
    license = "Hugging Face API terms"
    attribution = "Provider availability, pricing and performance from Hugging Face Inference Providers"

    def __init__(self, client: httpx.Client | None = None, url: str = API):
        self.client = client or httpx.Client(timeout=60)
        self.url = url

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(f"{self.url}/models")
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, data: dict[str, Any]) -> Iterable[Offer]:
        for model_data in data.get("data") or []:
            model_id = model_data.get("id")
            if not model_id:
                continue

            architecture = model_data.get("architecture") or {}
            input_modalities = architecture.get("input_modalities") or []
            output_modalities = architecture.get("output_modalities") or []
            vendor = (model_data.get("owned_by") or model_id.split("/")[0]).lower()
            providers = model_data.get("providers") or []
            live = [
                provider
                for provider in providers
                if provider.get("status") == "live"
                and not (provider.get("is_free") is True and not provider.get("pricing"))
            ]

            vision = ["vision"] if "image" in input_modalities else []
            common = {
                "capability": infer_capability(model_id, input_modalities, output_modalities),
                "model": model_id,
                "vendor": vendor,
                "channel": "huggingface",
                "channel_type": "aggregator",
                "provenance": Provenance(source=f"{API}#{model_id}", confidence="high"),
            }

            offers = []
            for provider in live:
                pricing = provider.get("pricing") or {}
                features = list(vision)
                if provider.get("supports_tools"):
                    features.append("tools")
                if provider.get("supports_structured_output"):
                    features.append("json_schema")
                offers.append(
                    Offer(
                        **common,
                        group=provider.get("provider"),
                        price=Price(input_per_1m=pricing.get("input"), output_per_1m=pricing.get("output")),
                        features=features,
                        reliability=Reliability(
                            ttft_ms=provider.get("first_token_latency_ms"),
                            throughput_tps=provider.get("throughput"),
                        ),
                        context_window=provider.get("context_length"),
                        endpoint=Endpoint(
                            protocol="openai",
                            base_url=API,
                            model_id=f"{model_id}:{provider.get('provider')}",
                            key_env="HF_TOKEN",
                        ),
                    )
                )
            yield from offers
            priced = [
                o for o in offers if o.price.input_per_1m is not None and o.price.output_per_1m is not None
            ]
            if priced:  # the router's `:cheapest` policy, approximated as the lowest input+output price
                best = min(priced, key=lambda o: (o.price.input_per_1m or 0) + (o.price.output_per_1m or 0))
                endpoint = best.endpoint.model_copy(update={"model_id": f"{model_id}:cheapest"})
                yield best.model_copy(update={"group": None, "endpoint": endpoint})
