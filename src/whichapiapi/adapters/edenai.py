"""Eden AI: one API over many non-LLM providers (speech-to-text, TTS, OCR, document parsing, translation, image
generation). Public `GET /v2/info/provider_subfeatures` (no key) lists every provider x feature with its price per
N units (seconds, characters, pages, files, images, requests), converted here to per-minute / per-1k-chars / per-page
prices. LLMs are skipped: they come from the LLM catalogs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from whichapiapi.schema.offer import Endpoint, Offer, Price, Provenance

API = "https://api.edenai.run/v2/info/provider_subfeatures"

_CAPABILITIES = {
    ("audio", "speech_to_text_async"): "speech.stt",
    ("audio", "tts"): "speech.tts",
    ("audio", "text_to_speech"): "speech.tts",
    ("audio", "text_to_speech_async"): "speech.tts",
    ("ocr", "ocr"): "doc.ocr",
    ("ocr", "ocr_async"): "doc.ocr",
    ("ocr", "ocr_tables_async"): "doc.parse",
    ("ocr", "invoice_parser"): "doc.parse",
    ("ocr", "receipt_parser"): "doc.parse",
    ("ocr", "financial_parser"): "doc.parse",
    ("ocr", "identity_parser"): "doc.parse",
    ("ocr", "resume_parser"): "doc.parse",
    ("translation", "automatic_translation"): "text.translate",
    ("translation", "document_translation"): "text.translate",
    ("image", "generation"): "image.gen",
}

_DEFAULT_UNITS = {
    "speech.stt": "minute",
    "speech.tts": "1k_chars",
    "text.translate": "1k_chars",
    "doc.ocr": "page",
    "doc.parse": "page",
    "image.gen": "image",
}


class EdenAIAdapter:
    id = "edenai"
    license = "Eden AI API terms"
    attribution = "Prices from Eden AI's public provider subfeatures API"

    def __init__(self, client: httpx.Client | None = None, url: str = API):
        self.client = client or httpx.Client(timeout=60)
        self.url = url

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(self.url)
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, data: list[dict[str, Any]]) -> Iterable[Offer]:
        best: dict[tuple[str, str], Offer] = {}

        for entry in data:
            if not entry.get("is_working"):
                continue

            feature = (entry.get("feature") or {}).get("name")
            subfeature = (entry.get("subfeature") or {}).get("name")
            capability = _CAPABILITIES.get((feature, subfeature))
            if capability is None:
                continue

            provider = (entry.get("provider") or {}).get("name")
            if not provider:
                continue

            model_endpoint = f"https://api.edenai.run/v2/{feature}/{subfeature}"
            model_base = f"{provider}/"
            for pricing in entry.get("pricings") or []:
                model_name = pricing.get("model_name")
                # Offer ids are channel/model, so "default" (Eden's name for a provider's only model) becomes the
                # subfeature: openai/speech_to_text_async and openai/tts must not collide.
                model = model_base + (model_name if model_name not in (None, "", "default") else subfeature)
                price = self._price(pricing, capability)
                if price is None:
                    continue

                offer = Offer(
                    capability=capability,
                    model=model,
                    vendor=provider,
                    channel="edenai",
                    channel_type="aggregator",
                    group=None,
                    price=price,
                    endpoint=Endpoint(
                        protocol="http",
                        base_url=model_endpoint,
                        model_id=model,
                        key_env="EDENAI_API_KEY",
                    ),
                    provenance=Provenance(
                        source=f"{API}#{provider}.{feature}.{subfeature}",
                        confidence="high",
                    ),
                )
                key = (capability, model)
                previous = best.get(key)
                if previous is None or self._cost(offer) < self._cost(previous):
                    best[key] = offer

        return best.values()

    @staticmethod
    def _cost(offer: Offer) -> float:
        price = offer.price
        return price.per_request if price.per_request is not None else price.per_unit or 0.0

    @staticmethod
    def _price(pricing: dict[str, Any], capability: str) -> Price | None:
        unit = pricing.get("price_unit_type")
        try:
            value = float(pricing.get("price"))
            quantity = float(pricing.get("price_unit_quantity") or 1)
        except (TypeError, ValueError):
            return None
        if value < 0 or quantity <= 0:
            return None

        if unit == "free":
            return Price(per_unit=0.0, unit=_DEFAULT_UNITS[capability])
        if unit == "seconde":
            return Price(per_unit=round(value / quantity * 60, 8), unit="minute")
        if unit == "minute":
            return Price(per_unit=round(value / quantity, 8), unit="minute")
        if unit == "char":
            return Price(per_unit=round(value / quantity * 1000, 8), unit="1k_chars")
        if unit in {"page", "file", "image"}:
            return Price(per_unit=round(value / quantity, 8), unit=unit)
        if unit == "request":
            return Price(per_request=round(value / quantity, 8))
        return None
