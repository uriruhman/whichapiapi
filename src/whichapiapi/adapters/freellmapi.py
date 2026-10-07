"""Free tiers of ~30 providers with their limits, from freellmapi's public catalog (github.com/tashfeenahmed/freellmapi,
MIT). `GET https://api.freellmapi.co/v1/latest` serves the free "monthly" snapshot (their paid tier is same-day); the
body is Ed25519-signed (`x-catalog-signature`) and verified here against the key pinned in their source, so a
tampered or spoofed feed is rejected.

Each model becomes a zero-priced offer in group `free` (`<platform>/<model>@free`) with `conditions.free_tier`
(rpm / rpd / tpm / tpd, the stated monthly token budget, reset time, provider quirks). Zero-priced offers stay
hidden in `find_offers` unless `include_free` — free tiers come with caps, no SLA and often training on prompts.
Their ranks (intelligence/speed) are not imported: our own benchmarks rank models.
Licence and credit: THIRD_PARTY_NOTICES.md (MIT, Copyright (c) 2026 Tashfeen Ahmed).
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import httpx

from whichapiapi.schema.capability import infer_capability
from whichapiapi.schema.offer import Conditions, Endpoint, FreeTier, Offer, Price, Provenance

API = "https://api.freellmapi.co/v1/latest"
PINNED_KEY = (
    "MCowBQYDK2VwAyEAq9yv4+3EeyMHKsfVYBhkcz1lYgIXSUeHNnN6tNgYX3k="  # from server/src/services/catalog-sync.ts
)
RESETS = {"google": "pacific_midnight"}


def verify(body: bytes, signature_b64: str, key_b64: str | None = None) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    try:
        key = load_der_public_key(base64.b64decode(key_b64 or PINNED_KEY))
        key.verify(base64.b64decode(signature_b64), body)  # type: ignore[call-arg, union-attr]
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


class FreeLLMAPIAdapter:
    id = "freellmapi"
    license = "MIT (github.com/tashfeenahmed/freellmapi); catalog: freellmapi.co free monthly snapshot"
    attribution = "Free-tier limits from the FreeLLMAPI catalog (freellmapi.co)"

    def __init__(self, client: httpx.Client | None = None, url: str = API, require_signature: bool = True):
        self.client = client or httpx.Client(timeout=60)
        self.url = url
        self.require_signature = require_signature

    def fetch(self) -> Iterable[Offer]:
        r = self.client.get(self.url)
        r.raise_for_status()
        sig = r.headers.get("x-catalog-signature")
        if self.require_signature and not (sig and verify(r.content, sig)):
            raise ValueError("freellmapi catalog signature missing or invalid — refusing the feed")
        data = r.json()
        when = datetime.now(UTC)
        version = data.get("version")
        source = f"{self.url} (v{version}, {data.get('tier')})"
        names = {p["id"]: p.get("name") for p in data.get("platforms") or []}
        for m in data.get("models") or []:
            if not m.get("enabled", True) or not m.get("platform") or not m.get("modelId"):
                continue
            yield self._offer(m, "llm.chat", source, when, names)
        for m in data.get("embeddings") or []:
            if m.get("enabled", True) and m.get("platform") and m.get("modelId"):
                yield self._offer(m, "llm.embeddings", source, when, names)
        for m in data.get("transcriptionModels") or []:
            if m.get("enabled", True) and m.get("platform") and m.get("modelId"):
                yield self._offer(m, "speech.stt", source, when, names)

    def _offer(self, m: dict[str, Any], default_cap: str, source: str, when: datetime, names: dict) -> Offer:
        lim = m.get("limits") or {}
        platform = m["platform"]
        notes = [f"{q.get('title')}: {q.get('body')}" for q in m.get("quirks") or [] if q.get("title")]
        if m.get("quotaLabel"):
            notes.append(f"quota: {m['quotaLabel']}")
        features = [
            f for f, flag in (("vision", m.get("supportsVision")), ("tools", m.get("supportsTools"))) if flag
        ]
        cap = default_cap
        if default_cap == "llm.chat":
            cap = infer_capability(m["modelId"])
        return Offer(
            capability=cap,
            model=m["modelId"],
            channel=platform,
            channel_type="official" if platform in ("google", "groq", "mistral", "cohere", "nvidia", "cloudflare")
            else "aggregator",
            group="free",
            price=Price(input_per_1m=0.0, output_per_1m=0.0),
            conditions=Conditions(
                free_tier=FreeTier(
                    rpm=_int(lim.get("rpm")), rpd=_int(lim.get("rpd")), tpm=_int(lim.get("tpm")),
                    tokens_per_day=_int(lim.get("tpd")), monthly_tokens=m.get("monthlyTokenBudget"),
                    resets=RESETS.get(platform, "utc_midnight"), notes=notes[:6],
                )
            ),
            endpoint=Endpoint(protocol="openai", model_id=m["modelId"]),
            context_window=_int(m.get("contextWindow")) or _int(m.get("maxInputTokens")),
            features=features,
            provenance=Provenance(source=source, verified_at=when, confidence="medium",
                                  note=f"{names.get(platform) or platform} free tier via FreeLLMAPI catalog"),
        )  # fmt: skip
