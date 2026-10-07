"""OpenAI-compatible speech-to-text (`speech.stt`): POST {base_url}/audio/transcriptions, multipart upload.

Works for OpenAI, Groq (`https://api.groq.com/openai/v1`) and new-api resellers that relay audio. The rendered user
message is the audio file path, relative to the suite directory (paths escaping it are refused: a suite must not be
able to upload arbitrary local files). Usage carries the audio length in minutes (`Usage.units`) for per-minute prices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI
from tinytag import TinyTag

from whichapiapi.transports.base import CallResult, Usage, audio_file, elapsed_ms, redact


class OpenAISTTTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300.0,
        max_retries: int = 1,
        base_dir: Path = Path("."),
        http_client: httpx.AsyncClient | None = None,
    ):
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if http_client is not None:
            kwargs["http_client"] = http_client
        self.client = AsyncOpenAI(**kwargs)
        self.base_dir = base_dir

    async def call(self, model: str, messages: list[dict[str, Any]], config: dict[str, Any]) -> CallResult:
        path = audio_file(messages, self.base_dir)
        if isinstance(path, str):
            return CallResult(error=path)

        try:
            duration = TinyTag.get(path).duration
        except Exception:
            duration = None
        units = duration / 60 if duration is not None else 0.0
        params = {k: v for k, v in config.items() if k in {"language", "prompt", "temperature"}}

        t0 = time.perf_counter()
        try:
            resp = await self.client.audio.transcriptions.create(
                model=model,
                file=(path.name, path.read_bytes()),
                response_format="json",
                **params,
            )
        except openai.APIError as e:
            return CallResult(
                error=redact(f"{type(e).__name__}: {str(e)[:300]}", self.client.api_key),
                latency_ms=elapsed_ms(t0),
            )

        latency = elapsed_ms(t0)
        text = resp.text or ""
        u = getattr(resp, "usage", None)
        usage_type = getattr(u, "type", None)
        if usage_type == "tokens":
            usage = Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                units=units,
            )
        elif usage_type == "duration":
            usage = Usage(units=(u.seconds or 0) / 60)
        else:
            usage = Usage(units=units)
        return CallResult(
            output=text,
            usage=usage,
            latency_ms=latency,
            raw={"audio_seconds": duration},
            error=None if text else "empty transcript",
        )
