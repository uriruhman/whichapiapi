"""DeepInfra (inference provider: LLMs, embeddings, STT, TTS, images). Public `GET /models/list` (no key) carries
every model's type and price in US cents per token / audio second / character / image. OpenAI-compatible API at
`/v1/openai` for chat, embeddings and transcription. `time`-priced models bill GPU seconds, not audio length."""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import httpx

from whichapiapi.schema.offer import Endpoint, Offer, Price, Provenance

API = "https://api.deepinfra.com/models/list"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _rounded(value: float | None) -> float | None:
    return round(value, 8) if value is not None else None


class DeepInfraAdapter:
    id = "deepinfra"
    license = "DeepInfra API terms"
    attribution = "Prices from DeepInfra's public model catalog"

    def __init__(self, client: httpx.Client | None = None, url: str = API):
        self.client = client or httpx.Client(timeout=60)
        self.url = url

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(self.url)
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, data: list[dict[str, Any]]) -> Iterable[Offer]:
        capabilities = {
            "text-generation": "llm.chat",
            "embeddings": "llm.embeddings",
            "automatic-speech-recognition": "speech.stt",
            "text-to-speech": "speech.tts",
            "text-to-image": "image.gen",
        }
        for record in data:
            if record.get("private"):
                continue
            # `deprecated` is the unix time the model is switched off (often a few days ahead), not a flag
            sunset = _number(record.get("deprecated"))
            if sunset is not None and sunset <= time.time():
                continue
            note = None
            if sunset is not None:
                when = datetime.fromtimestamp(sunset, UTC).date().isoformat()
                note = f"deprecated, switched off {when}" + (
                    f"; replaced by {record['replaced_by']}" if record.get("replaced_by") else ""
                )

            model_name = record.get("model_name")
            capability = capabilities.get(record.get("type"))
            pricing = record.get("pricing") or {}
            pricing_type = pricing.get("type")
            if not model_name or capability is None:
                continue

            input_per_1m = output_per_1m = cache_read_per_1m = None
            per_unit = None
            unit = None

            if pricing_type == "tokens":
                value = _number(pricing.get("cents_per_input_token"))
                if value is not None:
                    input_per_1m = value * 1e4
                value = _number(pricing.get("cents_per_output_token"))
                if value is not None:
                    output_per_1m = value * 1e4
                # a multiplier on the input price (0.2 = cached input costs 20 %), not cents per token
                value = _number(pricing.get("rate_per_input_token_cached"))
                if value is not None and input_per_1m is not None:
                    cache_read_per_1m = input_per_1m * value
            elif pricing_type == "input_tokens":
                value = _number(pricing.get("cents_per_input_token"))
                if value is not None:
                    input_per_1m = value * 1e4
            elif pricing_type == "input_length":
                value = _number(pricing.get("cents_per_input_sec"))
                if value is not None:
                    per_unit, unit = value * 60 / 100, "minute"
            elif pricing_type == "time":
                value = _number(pricing.get("cents_per_sec"))
                if value is not None:
                    per_unit, unit = value / 100, "exec_second"
            elif pricing_type == "input_character_length":
                value = _number(pricing.get("cents_per_input_chars"))
                if value is not None:
                    per_unit, unit = value * 1000 / 100, "1k_chars"
            elif pricing_type == "image_units":
                value = _number(pricing.get("cents_per_image_unit"))
                if value is not None:
                    per_unit, unit = value / 100, "image"
            else:
                continue

            price = Price(
                input_per_1m=_rounded(input_per_1m),
                output_per_1m=_rounded(output_per_1m),
                cache_read_per_1m=_rounded(cache_read_per_1m),
                per_unit=_rounded(per_unit),
                unit=unit,
            )
            if not price.is_known:
                continue

            yield Offer(
                capability=capability,
                model=model_name,
                vendor=model_name.split("/")[0].lower() if "/" in model_name else None,
                channel="deepinfra",
                channel_type="official",
                price=price,
                endpoint=Endpoint(
                    protocol="openai"
                    if capability in ("llm.chat", "llm.embeddings", "speech.stt")
                    else "http",
                    base_url="https://api.deepinfra.com/v1/openai",
                    model_id=model_name,
                    key_env="DEEPINFRA_API_KEY",
                ),
                context_window=record.get("max_tokens")
                if isinstance(record.get("max_tokens"), int)
                else None,
                provenance=Provenance(source=f"{API}#{model_name}", confidence="high", note=note),
            )
