import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from whichapiapi.adapters import freellmapi as fl

FEED = {
    "version": "2026.10.07",
    "tier": "monthly",
    "platforms": [{"id": "google", "name": "Google AI Studio"}],
    "models": [
        {"platform": "google", "modelId": "gemini-3.5-flash", "limits": {"rpm": 10, "rpd": 20, "tpm": 250000,
         "tpd": None}, "monthlyTokenBudget": "~3M", "contextWindow": 1048576, "enabled": True,
         "supportsVision": True, "supportsTools": True,
         "quirks": [{"title": "Thinking tokens use output cap", "body": "avoid tiny max_tokens"}]},
        {"platform": "groq", "modelId": "old", "enabled": False},
    ],
    "embeddings": [{"platform": "google", "modelId": "gemini-embedding-001", "maxInputTokens": 2048, "enabled": True,
                    "quotaLabel": "100 rpm · 1K req/day"}],
}  # fmt: skip


def _client(body: bytes, sig: str | None):
    headers = {"x-catalog-signature": sig} if sig else {}
    return httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body, headers=headers))
    )


def test_signed_feed_becomes_free_offers(monkeypatch):
    key = Ed25519PrivateKey.generate()
    pub = base64.b64encode(
        key.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    ).decode()
    monkeypatch.setattr(fl, "PINNED_KEY", pub)
    body = json.dumps(FEED).encode()
    sig = base64.b64encode(key.sign(body)).decode()
    offers = list(fl.FreeLLMAPIAdapter(client=_client(body, sig)).fetch())
    assert [o.id for o in offers] == ["google/gemini-3.5-flash@free", "google/gemini-embedding-001@free"]
    ft = offers[0].conditions.free_tier
    assert (ft.rpm, ft.rpd, ft.tpm, ft.monthly_tokens, ft.resets) == (
        10,
        20,
        250000,
        "~3M",
        "pacific_midnight",
    )
    assert offers[0].price.input_per_1m == 0 and offers[0].features == ["vision", "tools"]
    assert (
        offers[1].capability == "llm.embeddings"
        and "quota: 100 rpm" in offers[1].conditions.free_tier.notes[0]
    )


def test_signature_is_checked():
    key = Ed25519PrivateKey.generate()
    pub = base64.b64encode(
        key.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    ).decode()
    body = json.dumps(FEED).encode()
    good = base64.b64encode(key.sign(body)).decode()
    assert fl.verify(body, good, pub) and not fl.verify(body + b" ", good, pub)
    with pytest.raises(ValueError, match="signature"):
        list(
            fl.FreeLLMAPIAdapter(client=_client(body, good)).fetch()
        )  # signed by a key that isn't the pinned one
