"""Deepgram pre-recorded speech-to-text (`speech.stt`): POST {base_url}/listen?model=..., raw audio body,
`Authorization: Token <key>` — verified against the live API 2026-09-29.

The rendered user message is the audio path, relative to the suite directory (paths escaping it are refused, as in
the OpenAI STT transport). Language is auto-detected unless the provider config sets `language`. Usage carries the
billed audio length from `metadata.duration` in minutes (`Usage.units`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from whichapiapi.transports.base import CallResult, Usage, audio_file, elapsed_ms, http_error_kind, redact

_PASSTHROUGH = {"language", "detect_language", "smart_format", "punctuate", "numerals", "keyterm", "diarize"}
_CONTENT_TYPES = {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


class DeepgramTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        base_dir: Path = Path("."),
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.base_dir = base_dir
        self.client = client or httpx.AsyncClient(timeout=timeout)

    async def call(self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> CallResult:
        path = audio_file(messages, self.base_dir)
        if isinstance(path, str):
            return CallResult(error=path)
        params: dict[str, Any] = {"model": model, "smart_format": "true"}
        if "language" not in config:
            params["detect_language"] = "true"
        params |= {
            k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items() if k in _PASSTHROUGH
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        }
        t0 = time.perf_counter()
        try:
            r = await self.client.post(
                f"{self.base_url}/listen", params=params, headers=headers, content=path.read_bytes()
            )
        except httpx.HTTPError as e:
            return CallResult(error=f"APIConnectionError: {type(e).__name__}", latency_ms=elapsed_ms(t0))
        latency = elapsed_ms(t0)
        if r.status_code != 200:
            return CallResult(
                error=redact(
                    f"{http_error_kind(r.status_code)}: HTTP {r.status_code} {r.text[:200]}", self.api_key
                ),
                latency_ms=latency,
            )
        try:
            data = r.json()
            channel = data["results"]["channels"][0]
            text = channel["alternatives"][0]["transcript"]
            meta = data.get("metadata") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            return CallResult(error="APIStatusError: unexpected response shape", latency_ms=latency)
        info = next(iter((meta.get("model_info") or {}).values()), {})
        return CallResult(
            output=text,
            usage=Usage(units=float(meta.get("duration") or 0.0) / 60),
            latency_ms=latency,
            model_reported=info.get("name"),
            raw={"detected_language": channel.get("detected_language"), "request_id": meta.get("request_id")},
            error=None if text else "empty transcript",
        )
