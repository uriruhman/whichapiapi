"""Artificial Analysis data API (free tier, key required, 1000 requests/day, attribution to artificialanalysis.ai
required): `GET /api/v2/data/llms/models` with `x-api-key` — verified live 2026-09-29.

Per model: the Intelligence / Coding / Math indices (0-100), per-benchmark scores (HLE, GPQA, SciCode, LiveCodeBench,
IFBench, Terminal-Bench, tau2, ...; published as fractions, stored here as %), and measured median output speed and
time to first token. Models are keyed by the API's `slug` ("claude-opus-5-5"); several reasoning settings of one
model can share a slug, `pull_benchmarks` then keeps the best score per board.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from whichapiapi.schema.benchmark import BenchmarkScore

API = "https://artificialanalysis.ai/api/v2/data/llms/models"
MEDIA = "https://artificialanalysis.ai/api/v2/data/media"
MEDIA_KINDS = ("text-to-image", "image-editing", "text-to-speech", "text-to-video")  # Elo arenas, no prices
KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"
_INDICES = {
    "artificial_analysis_intelligence_index": "intelligence_index",
    "artificial_analysis_coding_index": "coding_index",
    "artificial_analysis_math_index": "math_index",
}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


class ArtificialAnalysisAdapter:
    id = "artificial_analysis"
    license = "Artificial Analysis free API terms (attribution required)"
    attribution = "Benchmarks and speed from Artificial Analysis (https://artificialanalysis.ai/)"

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None):
        self.client = client or httpx.Client(timeout=60)
        self.api_key = api_key if api_key is not None else os.environ.get(KEY_ENV)
        self.raw: dict[str, Any] = {}  # last LLM payload, kept for model cards (per reasoning variant)
        self.errors: dict[str, str] = {}

    def fetch(self) -> list[BenchmarkScore]:
        if not self.api_key:
            raise RuntimeError(f"{KEY_ENV} is not set (free key: artificialanalysis.ai)")
        headers = {"x-api-key": self.api_key}
        r = self.client.get(API, headers=headers)
        r.raise_for_status()
        self.raw = r.json()
        out = self.parse(self.raw)
        for kind in MEDIA_KINDS:  # 4 more requests of the 1000/day free quota
            try:
                m = self.client.get(f"{MEDIA}/{kind}", headers=headers)
                m.raise_for_status()
                out += self.parse_media(m.json(), kind)
            except (httpx.HTTPError, ValueError) as e:
                self.errors[kind] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    def parse_media(self, data: dict[str, Any], kind: str) -> list[BenchmarkScore]:
        out = []
        for m in data.get("data") or []:
            if (elo := _num(m.get("elo"))) is None or not m.get("slug"):
                continue
            ci = str(m.get("ci95") or "")
            half = _num(float(ci.split("/")[-1])) if ci.split("/")[-1].replace(".", "").isdigit() else None
            out.append(
                BenchmarkScore(
                    source=self.id,
                    board=f"media/{kind}",
                    model_name=m.get("name") or m["slug"],
                    score=elo,
                    rank=m.get("rank"),
                    ci_low=elo - half if half else None,
                    ci_high=elo + half if half else None,
                    votes=m.get("appearances"),
                    date=m.get("release_date"),
                    organization=(m.get("model_creator") or {}).get("name"),
                )
            )
        return out

    def parse(self, data: dict[str, Any]) -> list[BenchmarkScore]:
        out: list[BenchmarkScore] = []
        for m in data.get("data") or []:
            name = m.get("slug") or m.get("name")
            if not name:
                continue
            org = (m.get("model_creator") or {}).get("slug")
            date = m.get("release_date")
            values: list[tuple[str, float | None, str, bool]] = []
            for field, v in (m.get("evaluations") or {}).items():
                if field in _INDICES:
                    values.append((_INDICES[field], _num(v), "index", True))
                elif (x := _num(v)) is not None:
                    values.append((f"bench/{field}", round(x * 100, 2) if x <= 1 else x, "pct", True))
            values.append(("speed_tps", _num(m.get("median_output_tokens_per_second")), "tokens_per_s", True))
            values.append(("ttft_s", _num(m.get("median_time_to_first_token_seconds")), "s", False))
            out += [
                BenchmarkScore(
                    source=self.id,
                    board=board,
                    model_name=name,
                    score=round(v, 3),
                    unit=unit,
                    higher_is_better=higher,
                    date=date,
                    organization=org,
                )
                for board, v, unit, higher in values
                if v is not None and not (board in ("speed_tps", "ttft_s") and v <= 0)
            ]
        return out
