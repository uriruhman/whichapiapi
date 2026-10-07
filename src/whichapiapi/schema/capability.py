"""Capability taxonomy. A capability is *what* is needed, independent of who sells it."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Capability(BaseModel):
    id: str  # dotted: "llm.chat", "speech.stt"
    title: str
    unit: str = "token"  # pricing unit family: token | minute | request | page | 1k_requests
    scorers: list[str] = Field(default_factory=list)  # default assertion types for this capability


TAXONOMY: dict[str, Capability] = {
    c.id: c
    for c in [
        Capability(
            id="llm.chat", title="Chat / text generation", scorers=["llm-rubric", "is-json", "python"]
        ),
        Capability(id="llm.embeddings", title="Text embeddings", scorers=["python"]),
        Capability(id="llm.vision", title="Image understanding", scorers=["llm-rubric"]),
        Capability(id="search.web", title="Web search", unit="1k_requests", scorers=["llm-rubric"]),
        Capability(id="search.news", title="News search", unit="1k_requests", scorers=["llm-rubric"]),
        Capability(id="scrape.page", title="Fetch and extract a page", unit="request", scorers=["contains"]),
        Capability(id="speech.stt", title="Speech to text", unit="minute", scorers=["wer"]),
        Capability(id="speech.tts", title="Text to speech", unit="1k_chars", scorers=["llm-rubric"]),
        Capability(id="text.translate", title="Machine translation", unit="1k_chars", scorers=["llm-rubric"]),
        Capability(id="image.gen", title="Image generation", unit="image", scorers=["llm-rubric"]),
        Capability(id="video.gen", title="Video generation", unit="second", scorers=["llm-rubric"]),
        Capability(id="doc.ocr", title="OCR", unit="page", scorers=["cer"]),
        Capability(id="doc.parse", title="Document parsing", unit="page", scorers=["python"]),
        Capability(id="geo.geocode", title="Geocoding", unit="1k_requests", scorers=["distance"]),
        Capability(id="geo.routing", title="Routing", unit="1k_requests", scorers=["python"]),
        Capability(
            id="transit.realtime", title="Public transit realtime", unit="request", scorers=["python"]
        ),
        Capability(
            id="transit.schedule", title="Public transit schedules", unit="request", scorers=["python"]
        ),
        Capability(id="messaging.telegram", title="Telegram messaging", unit="request"),
        Capability(id="messaging.email", title="Email sending", unit="request"),
    ]
}


_STT_NAME = re.compile(r"whisper|transcri|\basr\b|-asr|parakeet|canary-(?:\d|qwen)|\bstt\b", re.I)
_TTS_NAME = re.compile(r"\btts\b|-tts|text-to-speech", re.I)
_EMBED_NAME = re.compile(r"embed", re.I)
_VIDEO_NAME = re.compile(r"video|\bveo|kling|seedance|hailuo|vidu|\bsora\b|runway|pika", re.I)
_IMAGE_NAME = re.compile(
    r"[-_]image\b|image[-_]|imagine|seedream|\bmj_|dall-e|\bflux\b|midjourney|imagen|upscale", re.I
)


def infer_capability(model: str, inputs: list[str] | None = None, outputs: list[str] | None = None) -> str:
    """Best-effort capability of a catalog model from its declared modalities, else its name. Catalogs list STT,
    TTS and embedding models next to chat models; without this they would all be offered as `llm.chat`."""
    inputs, outputs = inputs or [], outputs or []
    if "audio" in inputs and "text" not in inputs and "text" in outputs:
        return "speech.stt"
    if "audio" in outputs and "text" not in outputs:
        return "speech.tts"
    if "video" in outputs and "text" not in outputs:
        return "video.gen"
    if "image" in outputs and "text" not in outputs:
        return "image.gen"
    if _EMBED_NAME.search(model):
        return "llm.embeddings"
    if not inputs and _STT_NAME.search(model):
        return "speech.stt"
    if not outputs and _TTS_NAME.search(model):
        return "speech.tts"
    if not outputs and _VIDEO_NAME.search(model):
        return "video.gen"
    if not outputs and _IMAGE_NAME.search(model):
        return "image.gen"
    return "llm.chat"
